"""
EXPERIMENT 50: memTrace — Neural Breadcrumbs MIA via Hidden States & Attention

Paper: "Neural Breadcrumbs: Membership Inference Attacks on LLMs Through
        Hidden State and Attention Pattern Analysis"
       Makhija, Arivazhagan, Kumar, Gangadharaiah
       AWS AI Labs (arXiv:2509.05449v1, Sep 2025)

Core insight:
    Membership signals are strongly encoded in a model's INTERNAL
    representations (hidden states + attention patterns), not just outputs.
    The paper calls these traces "neural breadcrumbs" — distinctive
    processing pathways for member vs non-member sequences.

    Key findings:
    1. Middle layers show STRONGEST membership signal (not final layer!)
    2. Members exhibit higher VARIANCE in prediction confidence across
       token positions (recognition hot-spots)
    3. Layer transition patterns (how representations change between layers)
       carry strong signal
    4. Attention entropy/concentration patterns differ for member vs non-member
    5. AUC 0.85 average across MIMIR benchmarks; 0.73-0.77 on GitHub code

Feature families:
    1. LAYER TRANSITION FEATURES: L2 distance and cosine similarity between
       consecutive layer hidden states, per token → stats (mean, std, max, min)
    2. PREDICTION CONFIDENCE: entropy and confidence of logits per layer,
       plus confidence VARIANCE across tokens (key discriminator!)
    3. ATTENTION PATTERNS: attention entropy, concentration, sparsity per layer
    4. CONTEXT EVOLUTION: how mean representation changes as tokens accumulate
    5. POSITION FEATURES: first-last token similarity per layer

    All features aggregated into fixed-length vector → Random Forest classifier

Paper results:
    - Wikipedia/Pythia-6.9B: AUC 0.89
    - GitHub/Pythia-6.9B: AUC 0.77 (high n-gram overlap makes it harder)
    - BookMIA: AUC 0.95-0.99
    - Outperforms all baselines (Perplexity, Min-K%, Zlib, FSD, MIATuner)

Adaptation for Poisoned Chalice:
    - StarCoder2-3b has 32 transformer layers → extract from all
    - Attention extraction needs eager mode (memory intensive) → sample layers
    - Use 5-fold stratified CV like the paper, or probe+score approach
    - Code domain has high n-gram overlap (like GitHub in paper) →
      expect AUC 0.65-0.80

Related to our prior experiments:
    - EXP14 (Internal Entropy): AUC 0.4397 — used attention entropy alone
    - EXP29 (Attention Early-Settling): AUC 0.3840 — used limited attention features
    - EXP43 (AttenMIA): pending — attention-based but different feature set
    memTrace is MORE COMPREHENSIVE: combines hidden state transitions +
    attention + confidence + position features into a single classifier

Compute: 1 forward pass per sample (with output_hidden_states=True)
    + attention from sampled layers (eager mode for those layers)
    Forward-only but memory-intensive. 10% sample.
Expected runtime: ~15-25 min on A100 (10% sample)
Expected AUC: 0.60-0.80 (GitHub code achieved 0.77 in paper with Pythia-6.9B;
    our setup is StarCoder2-3b code-specific → may be stronger or weaker
    depending on memorization patterns)
"""
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


def setup_environment():
    print("\n" + "=" * 70)
    print("  EXP50: memTrace — Neural Breadcrumbs MIA")
    print("  Paper: Makhija et al. (arXiv:2509.05449v1, Sep 2025)")
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
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Loaded. dtype={dtype}, layers={n_layers}, hidden={hidden_dim}")
    return model, tokenizer


class MemTraceExtractor:
    """Extract memTrace features from transformer hidden states.

    Focuses on the most discriminative feature families from the paper:
    1. Layer transition features (surprise + stability)
    2. Prediction confidence features (entropy, confidence variance)
    3. Hidden state norm statistics per layer
    4. Position-based features (first-last similarity)
    """

    def __init__(self, model, tokenizer, max_length: int = 512,
                 n_sample_layers_attn: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._err_count = 0

        self.n_layers = model.config.num_hidden_layers
        self.hidden_dim = model.config.hidden_size

    def _compute_transition_features(self, hidden_states: List[torch.Tensor],
                                     seq_len: int) -> Dict[str, float]:
        """Layer transition features: L2 surprise and cosine stability."""
        features = {}
        n_transitions = len(hidden_states) - 1

        all_surprises_mean = []
        all_stability_mean = []

        for i in range(n_transitions):
            h_curr = hidden_states[i][0, :seq_len, :].float()   # (T, D)
            h_next = hidden_states[i+1][0, :seq_len, :].float()

            diff = h_next - h_curr
            surprise = diff.norm(dim=-1)  # (T,)

            cos_sim = F.cosine_similarity(h_curr, h_next, dim=-1)  # (T,)

            s_mean = surprise.mean().item()
            c_mean = cos_sim.mean().item()

            all_surprises_mean.append(s_mean)
            all_stability_mean.append(c_mean)

            # Per-transition stats for key layers only (to limit feature count)
            if i in [0, self.n_layers // 4, self.n_layers // 2,
                     3 * self.n_layers // 4, self.n_layers - 1]:
                tag = f"L{i}"
                features[f"surprise_mean_{tag}"] = s_mean
                features[f"surprise_std_{tag}"] = surprise.std().item()
                features[f"surprise_max_{tag}"] = surprise.max().item()
                features[f"stability_mean_{tag}"] = c_mean
                features[f"stability_std_{tag}"] = cos_sim.std().item()
                features[f"stability_min_{tag}"] = cos_sim.min().item()

        # Cross-layer transition statistics
        arr_s = np.array(all_surprises_mean)
        arr_c = np.array(all_stability_mean)
        features["surprise_global_mean"] = float(arr_s.mean())
        features["surprise_global_std"] = float(arr_s.std())
        features["surprise_global_max"] = float(arr_s.max())
        features["surprise_argmax"] = float(arr_s.argmax())
        features["stability_global_mean"] = float(arr_c.mean())
        features["stability_global_std"] = float(arr_c.std())
        features["stability_global_min"] = float(arr_c.min())
        features["stability_argmin"] = float(arr_c.argmin())

        # Middle vs edge transition ratio (paper: middle layers have strongest signal)
        mid_start = self.n_layers // 3
        mid_end = 2 * self.n_layers // 3
        if mid_end > mid_start:
            mid_surprise = arr_s[mid_start:mid_end].mean()
            edge_surprise = np.concatenate([arr_s[:mid_start], arr_s[mid_end:]]).mean()
            features["surprise_mid_edge_ratio"] = float(
                mid_surprise / (edge_surprise + 1e-10)
            )

        return features

    def _compute_confidence_features(self, logits: torch.Tensor,
                                     input_ids: torch.Tensor,
                                     seq_len: int) -> Dict[str, float]:
        """Prediction confidence and entropy features from final logits."""
        features = {}

        shift_logits = logits[0, :seq_len - 1, :].float()  # (T-1, V)
        shift_labels = input_ids[0, 1:seq_len]              # (T-1,)
        T = shift_logits.shape[0]
        if T < 3:
            return features

        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Per-token entropy
        entropy = -(probs * log_probs).sum(dim=-1)  # (T,)
        features["entropy_mean"] = entropy.mean().item()
        features["entropy_std"] = entropy.std().item()
        features["entropy_min"] = entropy.min().item()
        features["entropy_max"] = entropy.max().item()

        # Per-token confidence (max probability)
        confidence = probs.max(dim=-1).values  # (T,)
        features["confidence_mean"] = confidence.mean().item()
        features["confidence_std"] = confidence.std().item()
        features["confidence_min"] = confidence.min().item()
        features["confidence_max"] = confidence.max().item()

        # Confidence STABILITY (paper Figure 2: KEY discriminator)
        # Higher variance = member (recognition hot-spots)
        if confidence.std().item() > 1e-10:
            features["confidence_stability"] = float(
                confidence.mean().item() / confidence.std().item()
            )
        else:
            features["confidence_stability"] = 100.0

        # Confidence gap (top1 - top2)
        top2 = torch.topk(probs, k=2, dim=-1).values  # (T, 2)
        gap = top2[:, 0] - top2[:, 1]
        features["confidence_gap_mean"] = gap.mean().item()
        features["confidence_gap_std"] = gap.std().item()

        # Token-level loss
        token_ll = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
        features["neg_mean_loss"] = token_ll.mean().item()
        features["loss_std"] = (-token_ll).std().item()

        return features

    def _compute_hidden_norm_features(self, hidden_states: List[torch.Tensor],
                                      seq_len: int) -> Dict[str, float]:
        """Per-layer hidden state norm statistics."""
        features = {}
        layer_norms_mean = []

        for i, hs in enumerate(hidden_states):
            h = hs[0, :seq_len, :].float()
            norms = h.norm(dim=-1)  # (T,)
            n_mean = norms.mean().item()
            layer_norms_mean.append(n_mean)

            if i in [0, self.n_layers // 2, self.n_layers]:
                tag = f"L{i}"
                features[f"hnorm_mean_{tag}"] = n_mean
                features[f"hnorm_std_{tag}"] = norms.std().item()

        arr = np.array(layer_norms_mean)
        features["hnorm_global_mean"] = float(arr.mean())
        features["hnorm_growth_ratio"] = float(arr[-1] / (arr[0] + 1e-10))

        return features

    def _compute_position_features(self, hidden_states: List[torch.Tensor],
                                   seq_len: int) -> Dict[str, float]:
        """Position-based features: first-last token similarity per layer."""
        features = {}
        similarities = []

        for i in [0, self.n_layers // 4, self.n_layers // 2,
                  3 * self.n_layers // 4, self.n_layers]:
            if i >= len(hidden_states):
                continue
            h = hidden_states[i][0, :seq_len, :].float()
            if seq_len < 2:
                continue
            sim = F.cosine_similarity(
                h[0].unsqueeze(0), h[seq_len - 1].unsqueeze(0)
            ).item()
            features[f"first_last_sim_L{i}"] = sim
            similarities.append(sim)

        if similarities:
            features["first_last_sim_mean"] = float(np.mean(similarities))
            features["first_last_sim_std"] = float(np.std(similarities))

        return features

    def _compute_context_evolution(self, hidden_states: List[torch.Tensor],
                                   seq_len: int,
                                   layer_idx: int = -1) -> Dict[str, float]:
        """Context evolution: how mean representation changes with each new token."""
        features = {}

        if layer_idx == -1:
            layer_idx = self.n_layers // 2  # middle layer (strongest signal per paper)

        h = hidden_states[layer_idx][0, :seq_len, :].float()  # (T, D)
        if seq_len < 5:
            return features

        # Sample evolution at a few positions to avoid O(T²)
        sample_positions = [seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1]
        evolutions = []
        prev_mean = h[0].unsqueeze(0)

        for pos in sample_positions:
            if pos < 1 or pos >= seq_len:
                continue
            curr_mean = h[:pos + 1].mean(dim=0, keepdim=True)
            delta = (curr_mean - prev_mean).norm().item()
            evolutions.append(delta)
            prev_mean = curr_mean

        if evolutions:
            features["ctx_evolution_mean"] = float(np.mean(evolutions))
            features["ctx_evolution_std"] = float(np.std(evolutions))

        return features

    @torch.no_grad()
    def extract(self, text: str) -> Dict[str, float]:
        """Extract memTrace feature vector for a single sample."""
        result = {}

        if not text or len(text) < 20:
            return result

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True,
            ).to(self.model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            if seq_len < 5:
                return result

            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states  # tuple of (1, T, D)
            logits = outputs.logits               # (1, T, V)

            # 1. Layer transition features (THE key feature family)
            result.update(self._compute_transition_features(hidden_states, seq_len))

            # 2. Prediction confidence features
            result.update(self._compute_confidence_features(logits, input_ids, seq_len))

            # 3. Hidden state norm features
            result.update(self._compute_hidden_norm_features(hidden_states, seq_len))

            # 4. Position-based features
            result.update(self._compute_position_features(hidden_states, seq_len))

            # 5. Context evolution
            result.update(self._compute_context_evolution(hidden_states, seq_len))

            # Add sequence length as feature
            result["seq_len"] = float(seq_len)

            return result

        except Exception as e:
            if self._err_count < 3:
                print(f"\n[EXP50 WARN] {type(e).__name__}: {e}")
            self._err_count += 1
            return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
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
                frac=self.args.sample_fraction, random_state=self.args.seed
            ).reset_index(drop=True)
        print(f"[*] Dataset: {len(df)} samples ({self.args.sample_fraction*100:.0f}%)")
        return df

    def run(self):
        df = self.load_data()
        extractor = MemTraceExtractor(
            self.model, self.tokenizer,
            max_length=self.args.max_length,
        )

        print(f"\n[EXP50] Extracting memTrace features for {len(df)} samples...")
        print(f"  Features: transition + confidence + norms + position + evolution")
        print(f"  1 forward pass/sample with output_hidden_states=True")

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="[EXP50]"):
            rows.append(extractor.extract(row["content"]))
        feat_df = pd.DataFrame(rows)

        n_valid = feat_df.dropna(how="all").shape[0]
        print(f"\n[EXP50] Valid: {n_valid}/{len(df)}")
        if extractor._err_count > 0:
            print(f"[EXP50] Errors: {extractor._err_count}")

        feature_cols = [c for c in feat_df.columns if c != "seq_len"]
        print(f"[EXP50] Total features extracted: {len(feature_cols)}")

        for col in feat_df.columns:
            if col not in df.columns:
                df[col] = feat_df[col].values

        # --- Unsupervised signal AUCs (no classifier) ---
        print("\n" + "=" * 70)
        print("   EXP50: memTrace — UNSUPERVISED SIGNAL AUCs")
        print("=" * 70)

        unsup_aucs = {}
        for col in sorted(feature_cols):
            v = df.dropna(subset=[col])
            if len(v) < 50 or len(v["is_member"].unique()) < 2:
                continue
            auc = roc_auc_score(v["is_member"], v[col])
            auc_neg = roc_auc_score(v["is_member"], -v[col])
            best_auc = max(auc, auc_neg)
            direction = "+" if auc >= auc_neg else "-"
            unsup_aucs[col] = (best_auc, direction)

        top_features = sorted(unsup_aucs.items(), key=lambda x: x[1][0], reverse=True)[:15]
        print("\nTop 15 individual features (unsupervised):")
        for col, (auc, direction) in top_features:
            print(f"  {direction}{col:<50} AUC = {auc:.4f}")

        # --- Random Forest Classifier (5-fold CV like paper) ---
        print("\n" + "=" * 70)
        print("   EXP50: memTrace — RANDOM FOREST CLASSIFIER (5-fold CV)")
        print("=" * 70)

        valid_mask = feat_df.dropna(how="all").index
        X_all = feat_df.loc[valid_mask].copy()
        y_all = df.loc[valid_mask, "is_member"].values

        X_all = X_all.fillna(0)
        X_all = X_all.replace([np.inf, -np.inf], 0)

        feature_names = list(X_all.columns)
        X_np = X_all.values.astype(np.float64)
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.args.seed)

        fold_aucs = []
        all_scores = np.full(len(X_np), np.nan)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_np, y_all)):
            X_train, X_test = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = RandomForestClassifier(
                n_estimators=200, max_depth=8,
                min_samples_split=10, min_samples_leaf=5,
                max_features="sqrt", class_weight="balanced",
                random_state=self.args.seed, n_jobs=-1,
            )
            clf.fit(X_train_s, y_train)

            proba = clf.predict_proba(X_test_s)[:, 1]
            auc = roc_auc_score(y_test, proba)
            fold_aucs.append(auc)
            all_scores[test_idx] = proba

            print(f"  Fold {fold_idx+1}/{n_folds}: AUC = {auc:.4f} "
                  f"(train={len(train_idx)}, test={len(test_idx)})")

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"\n  memTrace RF CV Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

        # Write RF scores back
        df.loc[valid_mask, "memtrace_rf_score"] = all_scores

        # Feature importance from last fold
        print("\n--- Top 15 Feature Importances (last fold RF) ---")
        importances = clf.feature_importances_
        imp_idx = np.argsort(importances)[::-1][:15]
        for rank, idx in enumerate(imp_idx):
            print(f"  {rank+1:2d}. {feature_names[idx]:<45} imp = {importances[idx]:.4f}")

        # --- Comparison ---
        print(f"\n--- COMPARISON ---")
        print(f"  memTrace RF (5-fold CV): {mean_auc:.4f} +/- {std_auc:.4f}")

        neg_loss_auc = unsup_aucs.get("neg_mean_loss", (0.5, "+"))[0]
        print(f"  neg_mean_loss (unsupervised): {neg_loss_auc:.4f}")
        print(f"  vs EXP41 -grad_z_lang:  0.6539 (current best)")
        print(f"  vs EXP39 Ridge stacker:  0.6490")
        print(f"  vs EXP11 -grad_embed:    0.6472")

        if mean_auc > 0.6539:
            print(f"\n  NEW BEST! memTrace RF beats -grad_z_lang ({mean_auc:.4f} > 0.6539)")
        elif mean_auc > 0.6472:
            print(f"\n  STRONG: memTrace RF beats -grad_embed ({mean_auc:.4f} > 0.6472)")
        elif mean_auc > 0.60:
            print(f"\n  PROMISING: memTrace adds signal ({mean_auc:.4f})")
        else:
            print(f"\n  WEAK: memTrace RF = {mean_auc:.4f}")

        # Per-subset breakdown
        print(f"\n{'Subset':<10} | {'RF_AUC':<10} | {'neg_loss':<10} | N")
        print("-" * 45)
        for subset in sorted(df["subset"].unique()):
            sub = df[df["subset"] == subset]
            v = sub.dropna(subset=["memtrace_rf_score"])
            r = {}
            if not v.empty and len(v["is_member"].unique()) > 1:
                r["rf"] = roc_auc_score(v["is_member"], v["memtrace_rf_score"])
            else:
                r["rf"] = float("nan")
            v2 = sub.dropna(subset=["neg_mean_loss"]) if "neg_mean_loss" in sub.columns else pd.DataFrame()
            if not v2.empty and len(v2["is_member"].unique()) > 1:
                r["loss"] = roc_auc_score(v2["is_member"], v2["neg_mean_loss"])
            else:
                r["loss"] = float("nan")
            print(
                f"{subset:<10} | {r.get('rf', float('nan')):.4f}     "
                f"| {r.get('loss', float('nan')):.4f}     "
                f"| {len(sub)}"
            )

        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(self.output_dir / f"EXP50_{timestamp}.parquet", index=False)
        print(f"\n[EXP50] Results saved.")


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
        output_dir = "results"
        seed = 42

    print(f"[EXP50] memTrace: {Args.model_name}")
    print(f"  sample={Args.sample_fraction*100:.0f}%, max_len={Args.max_length}")
    print(f"  1 fwd pass/sample (output_hidden_states=True)")
    Experiment(Args).run()
