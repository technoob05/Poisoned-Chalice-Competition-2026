"""
EXPERIMENT 54: SPV-MIA — Self-Prompted Calibration MIA via Probabilistic Variation

Paper: "Membership Inference Attacks against Fine-tuned Large Language Models
        via Self-prompt Calibration"
       Fu, Wang, Gao, Liu, Li, Jiang (NeurIPS 2024)

Survey reference: Wu & Cao (arXiv:2503.19338v3, Aug 2025), Section 4.2 [17]

Core idea:
    Traditional reference-based MIAs need external reference data.
    SPV-MIA generates reference data FROM THE TARGET MODEL ITSELF.
    The model's own generations have similar distribution to training data
    but are NOT exact training samples (due to stochastic generation).

    Process:
    1. For each target sample x, use the first half as prompt → generate
       continuation with the target model → this is the "self-reference"
    2. Compute probabilistic variation metric between original and generated:
       pv(x) ≈ (1/2N) * sum(p(x+Z_n) + p(x-Z_n)) - p(x)
       where Z_n are small perturbations
    3. Do the same for the generated reference: pv(x_gen)
    4. Membership signal = pv(x) - pv(x_gen)

    For pre-training MIA, we simplify:
    - Instead of probabilistic variation (computationally expensive),
      we use the LIKELIHOOD RATIO between original and self-generated
      continuations as the calibration signal
    - The model should assign HIGHER likelihood to training data than
      to its own stochastic generations (memorization > generation)

Adaptation for Poisoned Chalice:
    - Split each code sample: first 50% as prompt, second 50% as target
    - Generate continuation from prompt using target model (greedy/sample)
    - Compare: LL(original_suffix | prefix) vs LL(generated_suffix | prefix)
    - Membership signal: LL_orig - LL_gen (higher = more likely member)
    - Also: generate multiple continuations → variance of LL as signal
      (members should have more consistent continuations)

    ~3 forward passes per sample (1 for loss, 1 for generation, 1 for
    generated text loss). Forward-only, 10% sample.

Expected runtime: ~15-20 min on A100
Expected AUC: 0.52-0.62 (self-calibration may help on code since model
    generates structured code; original paper targets fine-tuned LLMs)
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  EXP54: SPV-MIA — Self-Prompted Calibration")
    print("  Paper: Fu et al. (NeurIPS 2024)")
    print("=" * 70)
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("posioned")
        from huggingface_hub import login
        login(token=hf_token)
    except Exception as e:
        print(f"[HF] Note: {e}")


def load_model(model_path: str):
    print(f"[*] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    print(f"  Loaded. dtype={dtype}")
    return model, tokenizer


class SPVScorer:
    """Self-Prompted Calibration: compare original vs self-generated continuations."""

    def __init__(self, model, tokenizer, max_length: int = 512,
                 prefix_ratio: float = 0.5, n_generations: int = 3,
                 gen_max_new_tokens: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix_ratio = prefix_ratio
        self.n_generations = n_generations
        self.gen_max_new_tokens = gen_max_new_tokens
        self._err_count = 0

    @torch.no_grad()
    def _compute_suffix_ll(self, prefix_ids: torch.Tensor,
                           suffix_ids: torch.Tensor) -> float:
        """Compute mean log-likelihood of suffix conditioned on prefix."""
        full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
        if full_ids.shape[1] > self.max_length:
            full_ids = full_ids[:, :self.max_length]

        outputs = self.model(input_ids=full_ids)
        logits = outputs.logits[0, :-1, :].float()
        labels = full_ids[0, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_ll = log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)

        prefix_len = prefix_ids.shape[1]
        suffix_ll = token_ll[prefix_len - 1:]  # log-probs for suffix tokens
        if len(suffix_ll) == 0:
            return 0.0
        return float(suffix_ll.mean().item())

    @torch.no_grad()
    def _generate_continuation(self, prefix_ids: torch.Tensor,
                               temperature: float = 0.8) -> torch.Tensor:
        """Generate continuation from prefix using target model."""
        gen_output = self.model.generate(
            input_ids=prefix_ids,
            max_new_tokens=self.gen_max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_suffix = gen_output[:, prefix_ids.shape[1]:]
        return generated_suffix

    def score(self, text: str) -> Dict[str, float]:
        result = {}
        if not text or len(text) < 50:
            return result

        try:
            tokens = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            )
            input_ids = tokens["input_ids"].to(self.model.device)
            seq_len = input_ids.shape[1]
            if seq_len < 20:
                return result

            # Split into prefix and suffix
            split_point = int(seq_len * self.prefix_ratio)
            split_point = max(10, min(split_point, seq_len - 10))
            prefix_ids = input_ids[:, :split_point]
            suffix_ids = input_ids[:, split_point:]
            suffix_len = suffix_ids.shape[1]

            # 1. Compute LL of ORIGINAL suffix given prefix
            orig_ll = self._compute_suffix_ll(prefix_ids, suffix_ids)
            result["orig_suffix_ll"] = orig_ll

            # 2. Generate continuations and compute their LL
            gen_lls = []
            for _ in range(self.n_generations):
                gen_suffix = self._generate_continuation(prefix_ids)
                if gen_suffix.shape[1] < 3:
                    continue
                # Truncate generated suffix to same length as original
                gen_suffix = gen_suffix[:, :suffix_len]
                gen_ll = self._compute_suffix_ll(prefix_ids, gen_suffix)
                gen_lls.append(gen_ll)

            if not gen_lls:
                return result

            gen_lls = np.array(gen_lls)
            result["gen_suffix_ll_mean"] = float(gen_lls.mean())
            result["gen_suffix_ll_std"] = float(gen_lls.std())

            # 3. SPV signal: original LL minus generated LL
            # Higher = model "prefers" the original continuation more
            result["spv_diff"] = orig_ll - float(gen_lls.mean())

            # 4. SPV ratio
            if abs(gen_lls.mean()) > 1e-10:
                result["spv_ratio"] = orig_ll / float(gen_lls.mean())
            else:
                result["spv_ratio"] = 1.0

            # 5. Generation consistency (members may produce more consistent code)
            if len(gen_lls) >= 2:
                result["gen_consistency"] = -float(gen_lls.std())
            else:
                result["gen_consistency"] = 0.0

            # 6. Full-sequence loss (baseline comparison)
            full_outputs = self.model(input_ids=input_ids)
            logits = full_outputs.logits[0, :-1, :].float()
            labels = input_ids[0, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_ll = log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            result["neg_mean_loss"] = float(token_ll.mean().item())

            result["seq_len"] = float(seq_len)
            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP54 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.model, self.tokenizer = load_model(args.model_name)

    def load_data(self) -> pd.DataFrame:
        subsets = ["Go", "Java", "Python", "Ruby", "Rust"]
        dfs = []
        is_local = os.path.exists(self.args.dataset)
        for subset in subsets:
            if is_local:
                path = os.path.join(self.args.dataset, subset)
                if not os.path.exists(path):
                    continue
                ds = load_from_disk(path)
                if hasattr(ds, "keys") and "test" in ds.keys():
                    ds = ds["test"]
            else:
                ds = load_dataset(self.args.dataset, subset, split="test")
            sub_df = ds.to_pandas()
            sub_df["subset"] = subset
            dfs.append(sub_df)
        df = pd.concat(dfs, ignore_index=True)
        df["is_member"] = df["membership"].apply(lambda x: 1 if x == "member" else 0)
        if self.args.sample_fraction < 1.0:
            df = df.sample(
                frac=self.args.sample_fraction, random_state=self.args.seed,
            ).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        scorer = SPVScorer(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
            prefix_ratio=self.args.prefix_ratio,
            n_generations=self.args.n_generations,
            gen_max_new_tokens=self.args.gen_max_new_tokens,
        )

        n_fwd = 1 + self.args.n_generations * 2 + 1  # orig + gen*2 + full
        print(f"\n[EXP54] Scoring {len(df)} samples...")
        print(f"  prefix_ratio={self.args.prefix_ratio}")
        print(f"  n_generations={self.args.n_generations}")
        print(f"  ~{n_fwd} forward passes per sample")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP54]"):
            rows.append(scorer.score(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[EXP54] Valid: {n_valid}/{len(df)}")

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- AUC Results ---
        print("\n" + "=" * 70)
        print("   EXP54: SPV-MIA RESULTS")
        print("=" * 70)

        score_cols = ["orig_suffix_ll", "gen_suffix_ll_mean", "spv_diff",
                      "spv_ratio", "gen_consistency", "neg_mean_loss"]
        aucs = {}
        for col in score_cols:
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            vals = v[col].values
            if np.std(vals) < 1e-15:
                continue
            auc = roc_auc_score(v["is_member"], vals)
            auc_neg = roc_auc_score(v["is_member"], -vals)
            best = max(auc, auc_neg)
            direction = "+" if auc >= auc_neg else "-"
            aucs[col] = (best, direction)
            print(f"  {direction}{col:<30} AUC = {best:.4f}")

        if aucs:
            best_signal = max(aucs.items(), key=lambda x: x[1][0])
            print(f"\n  BEST: {best_signal[1][1]}{best_signal[0]} = {best_signal[1][0]:.4f}")

        print(f"  vs EXP41 -grad_z_lang: 0.6539 (current best)")
        print(f"  vs EXP01 raw loss:     0.5807")

        # SPV signal analysis
        for col in ["spv_diff", "spv_ratio"]:
            v = df.dropna(subset=[col])
            m = v[v["is_member"] == 1][col]
            nm = v[v["is_member"] == 0][col]
            if len(m) > 10 and len(nm) > 10:
                print(f"\n  {col}: M={m.mean():.4f}±{m.std():.4f}, "
                      f"NM={nm.mean():.4f}±{nm.std():.4f}")

        # Per-subset for spv_diff
        if "spv_diff" in aucs:
            print(f"\n{'Subset':<10} | {'spv_diff':<10} | {'neg_loss':<10} | N")
            print("-" * 45)
            for subset in sorted(df["subset"].unique()):
                sub = df[df["subset"] == subset]
                r = {}
                for col in ["spv_diff", "neg_mean_loss"]:
                    v = sub.dropna(subset=[col])
                    if not v.empty and len(v["is_member"].unique()) > 1:
                        auc_p = roc_auc_score(v["is_member"], v[col])
                        auc_n = roc_auc_score(v["is_member"], -v[col])
                        r[col] = max(auc_p, auc_n)
                    else:
                        r[col] = float("nan")
                print(f"  {subset:<10} | {r.get('spv_diff', float('nan')):.4f}    "
                      f"| {r.get('neg_mean_loss', float('nan')):.4f}    "
                      f"| {len(sub)}")

        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP54_{timestamp}.parquet", index=False)
        print(f"\n[EXP54] Results saved.")


if __name__ == "__main__":
    setup_environment()

    class Args:
        model_name = "bigcode/starcoder2-3b"
        if os.path.exists("/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"):
            dataset = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
        else:
            dataset = "AISE-TUDelft/Poisoned-Chalice"
        sample_fraction = 0.10
        max_length = 512
        prefix_ratio = 0.50
        n_generations = 3
        gen_max_new_tokens = 128
        output_dir = "results"
        seed = 42

    print(f"[EXP54] SPV-MIA: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  prefix_ratio={Args.prefix_ratio}, n_gen={Args.n_generations}")
    Experiment(Args).run()
