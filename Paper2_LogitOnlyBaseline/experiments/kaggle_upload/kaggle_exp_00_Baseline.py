#!/usr/bin/env python3
"""
Kaggle Script — exp_00_Baseline (Logit-Only Baselines, No ESP)
================================================================
Self-contained: paste entire file into a single Kaggle notebook cell.

Methods: Loss, Min-K% (Shi 2024), Min-K%++ (Zhang ICLR 2025), SURP, Zlib
No entropy slope, no calibration — pure loss-based baselines.

Requirements:
  - GPU: T4 or P100
  - Kaggle Secret: "posioned" → HuggingFace token
  - Dataset: minh2duy/poisoned-chalice-dataset (attach as input)
"""

# ── Install & Auth ──────────────────────────────────────────────────────────
import subprocess, sys, os

_shm_free = 0
try:
    _st = os.statvfs("/dev/shm")
    _shm_free = (_st.f_bavail * _st.f_frsize) / 1e9
except Exception:
    pass
_hf_root = "/dev/shm/hf_cache" if _shm_free > 30 else "/tmp/hf_cache"
os.makedirs(os.path.join(_hf_root, "hub"), exist_ok=True)
os.environ["HF_HOME"] = _hf_root
os.environ["HF_HUB_CACHE"] = f"{_hf_root}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_hf_root}/hub"
print(f"✓ HF cache → {_hf_root}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.40", "accelerate", "datasets",
                "scikit-learn", "scipy", "huggingface_hub>=0.23"],
               capture_output=True)

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    for sn in ["HF_TOKEN", "posioned"]:
        try:
            t = UserSecretsClient().get_secret(sn)
            if t: login(token=t, add_to_git_credential=True); print(f"✓ HF auth: {sn}"); break
        except Exception: continue
except Exception as e:
    print(f"○ No Kaggle secrets: {e}")

# ── CONFIG ──────────────────────────────────────────────────────────────────
CONFIG = {
    "model_name":       "bigcode/starcoder2-3b",
    "max_length":       512,
    "sample_fraction":  0.10,   # ← set to 1.0 for full run
    "split":            "test",
    "seed":             42,
    "min_tokens":       8,
}

# ── Imports ─────────────────────────────────────────────────────────────────
import gc, json, time, warnings, zlib
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# ── Model Loader ─────────────────────────────────────────────────────────────
def load_model(name):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"✓ {name}: {model.config.num_hidden_layers}L, "
          f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tok

# ── Extractor: Baselines ONLY (no entropy slope) ─────────────────────────────
class BaselineExtractor:
    """Loss, Min-K%, Min-K%++ (ICLR 2025), SURP, Zlib. No ESP."""
    def __init__(self, model, tok, max_len=512, min_tok=8):
        self.model, self.tok = model, tok
        self.max_len, self.min_tok = max_len, min_tok
        self.dev = next(model.parameters()).device

    @torch.no_grad()
    def extract(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_len, padding=False)
        ids = enc["input_ids"].to(self.dev)
        n   = ids.shape[1]
        if n < self.min_tok:
            return {k: np.nan for k in [
                "neg_mean_loss","minkprob_20","minkpp_20","minkpp_10","minkpp_50",
                "surp","zlib_ratio","neg_mean_rank",
                "signal_loss","signal_mink","signal_minkpp","signal_zlib",
                "seq_len","n_tokens"]}

        out       = self.model(input_ids=ids)
        logits    = out.logits[:, :-1, :].float()
        labels    = ids[:, 1:]
        n         = logits.shape[1]
        probs     = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        token_lp   = log_probs.squeeze(0).gather(
                         1, labels.squeeze(0).unsqueeze(1)).squeeze(1)
        token_loss = -token_lp.cpu().numpy()
        mean_loss  = token_loss.mean()

        # Min-K% (Shi et al., 2024)
        lp_np = token_lp.cpu().numpy()
        minkprob_20 = float(np.sort(lp_np)[:max(1, int(n*0.2))].mean())

        # Min-K%++ (Zhang et al., ICLR 2025)
        mu_v  = (probs * log_probs).sum(-1).squeeze(0)
        var_v = (probs * log_probs.pow(2)).sum(-1).squeeze(0) - mu_v.pow(2)
        z_tok = (token_lp - mu_v) / var_v.clamp(min=1e-20).sqrt()
        z_np  = z_tok.cpu().numpy()
        minkpp_20 = float(np.sort(z_np)[:max(1, int(n*0.2))].mean())
        minkpp_10 = float(np.sort(z_np)[:max(1, int(n*0.1))].mean())
        minkpp_50 = float(np.sort(z_np)[:max(1, int(n*0.5))].mean())

        ranks = (logits.squeeze(0).argsort(-1, descending=True).argsort(-1)
                 .gather(1, labels.squeeze(0).unsqueeze(1)).squeeze(1)
                 .float().cpu().numpy())

        surp_val   = mean_loss - token_loss.std()
        zl_bytes   = len(zlib.compress(text.encode("utf-8")))
        zlib_ratio = (-mean_loss) / zl_bytes if zl_bytes > 0 else 0.0

        del out; torch.cuda.empty_cache()
        return {
            "neg_mean_loss": -mean_loss,
            "minkprob_20":   minkprob_20,
            "minkpp_20":     minkpp_20,
            "minkpp_10":     minkpp_10,
            "minkpp_50":     minkpp_50,
            "surp":          -surp_val,
            "zlib_ratio":    zlib_ratio,
            "neg_mean_rank": -ranks.mean(),
            "signal_loss":   -mean_loss,
            "signal_mink":   minkprob_20,
            "signal_minkpp": minkpp_20,
            "signal_zlib":   zlib_ratio,
            "seq_len": ids.shape[1], "n_tokens": n,
        }

# ── Data Loader ──────────────────────────────────────────────────────────────
def load_pc_data(split="test", frac=1.0, seed=42):
    from datasets import load_dataset, load_from_disk
    kp = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset/poisoned_chalice_dataset"
    rows = []
    for lang in ["Go", "Java", "Python", "Ruby", "Rust"]:
        try:
            lp = os.path.join(kp, lang, split)
            ds = load_from_disk(lp) if os.path.exists(lp) \
                 else load_dataset("AISE-TUDelft/Poisoned-Chalice", lang, split=split)
            for r in ds:
                text = r.get("content") or ""
                if not text.strip(): continue
                mem = r["membership"]
                rows.append({"text": text,
                             "is_member": 1 if (mem=="member" or mem==1) else 0,
                             "subset": lang})
            print(f"    {lang}: {len(ds)}")
        except Exception as e:
            print(f"    {lang}: ERR {e}")
    df = pd.DataFrame(rows)
    if frac < 1:
        df = df.groupby(["subset","is_member"]).apply(
            lambda x: x.sample(frac=frac, random_state=seed)).reset_index(drop=True)
    print(f"  Total: {len(df)} ({df.is_member.sum()} members)")
    return df

# ── Evaluation ───────────────────────────────────────────────────────────────
def eval_auc(df, cols, label="is_member"):
    res = []
    for c in cols:
        v = df[c].notna() & df[label].notna()
        if v.sum() < 10 or len(np.unique(df.loc[v, label])) < 2: continue
        auc = roc_auc_score(df.loc[v, label], df.loc[v, c])
        res.append({"score": c, "auc": max(auc,1-auc), "pol": "+" if auc>=0.5 else "-"})
    return pd.DataFrame(res).sort_values("auc", ascending=False)

def eval_per_subset(df, col, label="is_member"):
    res = []
    for sub, g in df.groupby("subset"):
        v = g[col].notna() & g[label].notna()
        if v.sum() < 10 or len(np.unique(g.loc[v, label])) < 2: continue
        auc = roc_auc_score(g.loc[v, label], g.loc[v, col])
        res.append({"subset": sub, "auc": max(auc,1-auc), "n": int(v.sum())})
    return pd.DataFrame(res)

# ── Run Pipeline ─────────────────────────────────────────────────────────────
def run_benchmark(model, tok, df, tag, max_len=512):
    ext = BaselineExtractor(model, tok, max_len)
    t0  = time.time()
    feats = []
    for i, row in df.iterrows():
        if i > 0 and i % 500 == 0:
            r = i / (time.time()-t0)
            print(f"  [{i}/{len(df)}] {r:.1f}/s ETA {(len(df)-i)/r:.0f}s")
        feats.append(ext.extract(row["text"]))
    fdf = pd.DataFrame(feats)
    df  = pd.concat([df.reset_index(drop=True), fdf], axis=1)
    print(f"  ✓ {len(df)} rows in {time.time()-t0:.0f}s")

    score_cols = [c for c in fdf.columns if c not in ["seq_len","n_tokens"]]
    print(f"\n{'─'*50}\n  RESULTS: {tag}\n{'─'*50}")
    res = eval_auc(df, score_cols)
    for _, r in res.iterrows():
        m = "★" if "minkpp" in r["score"] else " "
        print(f"  {m} {r['score']:25s}  AUC={r['auc']:.4f} ({r['pol']})")

    best = res.iloc[0]["score"] if len(res) > 0 else "signal_minkpp"
    print(f"\n  Per-language ({best}):")
    for _, sr in eval_per_subset(df, best).iterrows():
        print(f"    {sr['subset']:10s}  AUC={sr['auc']:.4f} (n={sr['n']})")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    odir = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "./results"
    os.makedirs(odir, exist_ok=True)
    df.to_parquet(f"{odir}/baseline_{tag}_{ts}.parquet", index=False)
    with open(f"{odir}/baseline_{tag}_{ts}.json", "w") as f:
        json.dump({"tag": tag, "results": res.to_dict("records")}, f, indent=2)
    print(f"  Saved → {odir}/baseline_{tag}_{ts}.*")
    return df, res

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    print("\n" + "█"*60 + "\n  exp_00_Baseline: Poisoned Chalice\n" + "█"*60)
    model, tok = load_model(CONFIG["model_name"])
    df_pc = load_pc_data(CONFIG["split"], CONFIG["sample_fraction"], CONFIG["seed"])
    run_benchmark(model, tok, df_pc, "PoisonedChalice", CONFIG["max_length"])
    del model, tok; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "═"*60 + "\n  DONE — exp_00_Baseline\n" + "═"*60)
