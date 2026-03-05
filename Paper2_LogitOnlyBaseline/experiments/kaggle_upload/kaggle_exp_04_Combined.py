#!/usr/bin/env python3
"""
Kaggle Script — exp_04_Combined (All Features + Ensembles)
==========================================================
Self-contained: paste entire file into a single Kaggle notebook cell.

Combines EVERY signal from exp_02 + exp_03, applies 3-scale calibration,
then builds mean ensembles to find the best combination.

Key questions:
  1. What is the single best logit-only feature?
  2. ESP + SurpriseTraj vs either alone?
  3. Does any ensemble beat Min-K%++ (ICLR 2025 SOTA)?

Ensemble groups:
  ens_esp_traj    = [signal_esp,  signal_surprise_drop]
  ens_esp_minkpp  = [signal_esp,  signal_minkpp]
  ens_traj_minkpp = [signal_surprise_drop, signal_minkpp]
  ens_all         = [signal_esp,  signal_surprise_drop, signal_minkpp,
                     signal_loss, signal_zlib]

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
    "position_buckets": 16,
}

ENSEMBLE_GROUPS = {
    "ens_esp_traj":    ["signal_esp", "signal_surprise_drop"],
    "ens_esp_minkpp":  ["signal_esp", "signal_minkpp"],
    "ens_traj_minkpp": ["signal_surprise_drop", "signal_minkpp"],
    "ens_esp_traj_minkpp": ["signal_esp", "signal_surprise_drop", "signal_minkpp"],
    "ens_all":         ["signal_esp", "signal_surprise_drop", "signal_minkpp",
                        "signal_loss", "signal_zlib"],
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

# ── Extractor: ALL features (ESP + SurpriseTraj + MinK%++) ───────────────────
class CombinedExtractor:
    """All logit-level features in a single forward pass."""
    def __init__(self, model, tok, max_len=512, min_tok=8):
        self.model, self.tok = model, tok
        self.max_len, self.min_tok = max_len, min_tok
        self.dev = next(model.parameters()).device

    @torch.no_grad()
    def extract(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_len, padding=False)
        ids = enc["input_ids"].to(self.dev)
        seq = ids.shape[1]
        if seq < self.min_tok:
            return {k: np.nan for k in [
                "esp_slope","z_esp_slope","h_drop","h_curvature",
                "neg_mean_loss","minkprob_20","minkpp_20","minkpp_10","minkpp_50",
                "surp","zlib_ratio",
                "surprise_drop","surprise_accel","neg_surprise_vol",
                "neg_loss_q_range","max_loss_drop","settle_frac",
                "signal_esp","signal_h_drop","signal_loss",
                "signal_mink","signal_minkpp","signal_zlib","signal_surprise_drop",
                "seq_len","n_tokens"]}

        out       = self.model(input_ids=ids)
        logits    = out.logits[:, :-1, :].float()
        labels    = ids[:, 1:]
        n         = logits.shape[1]
        probs     = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        H          = -(probs * log_probs).sum(-1).squeeze(0).cpu().numpy()
        token_lp   = log_probs.squeeze(0).gather(
                         1, labels.squeeze(0).unsqueeze(1)).squeeze(1)
        token_loss = -token_lp.cpu().numpy()
        lp_np      = token_lp.cpu().numpy()
        pos        = np.arange(n)
        mid        = n // 2
        ml         = token_loss.mean()

        # ── Entropy (ESP) ──
        esp_slope, _ = np.polyfit(pos, H, 1)
        h_mean, h_std = H.mean(), H.std()
        h_drop = H[:mid].mean() - H[mid:].mean() if mid > 0 else 0.0
        h_curv = np.polyfit(pos, H, 2)[0] if n >= 6 else 0.0
        z_esp  = np.polyfit(pos, (H - h_mean) / max(h_std, 1e-10), 1)[0] if h_std > 0 else 0.0

        # ── Loss ──
        surp_val = ml - token_loss.std()
        zl_bytes = len(zlib.compress(text.encode("utf-8")))
        zlib_ratio = (-ml) / zl_bytes if zl_bytes > 0 else 0.0

        # ── Min-K% ──
        minkprob_20 = float(np.sort(lp_np)[:max(1, int(n*0.2))].mean())

        # ── Min-K%++ (ICLR 2025) ──
        mu_v      = (probs * log_probs).sum(-1).squeeze(0)
        var_v     = (probs * log_probs.pow(2)).sum(-1).squeeze(0) - mu_v.pow(2)
        z_tok     = (token_lp - mu_v) / var_v.clamp(min=1e-20).sqrt()
        z_np      = z_tok.cpu().numpy()
        minkpp_20 = float(np.sort(z_np)[:max(1, int(n*0.2))].mean())
        minkpp_10 = float(np.sort(z_np)[:max(1, int(n*0.1))].mean())
        minkpp_50 = float(np.sort(z_np)[:max(1, int(n*0.5))].mean())

        # ── Surprise trajectory ──
        loss_first  = token_loss[:mid].mean() if mid > 0 else ml
        loss_second = token_loss[mid:].mean() if mid < n else ml
        surprise_drop  = float(loss_first - loss_second)
        surprise_accel = float(np.polyfit(pos, token_loss, 2)[0]) if n >= 6 else 0.0
        diffs = np.diff(token_loss)
        neg_surprise_vol  = float(-diffs.std()) if len(diffs) > 1 else 0.0
        neg_loss_q_range  = float(-(np.quantile(token_loss,0.9) - np.quantile(token_loss,0.1))) if n>2 else 0.0
        max_loss_drop     = float(-diffs.min()) if len(diffs) > 0 else 0.0
        settle_frac       = float((token_loss[mid:] < np.median(token_loss)).mean()) if mid < n else 0.5

        del out; torch.cuda.empty_cache()
        return {
            "esp_slope":          esp_slope,
            "z_esp_slope":        z_esp,
            "h_drop":             h_drop,
            "h_curvature":        h_curv,
            "neg_mean_loss":      -ml,
            "minkprob_20":        minkprob_20,
            "minkpp_20":          minkpp_20,
            "minkpp_10":          minkpp_10,
            "minkpp_50":          minkpp_50,
            "surp":               -surp_val,
            "zlib_ratio":         zlib_ratio,
            "surprise_drop":      surprise_drop,
            "surprise_accel":     surprise_accel,
            "neg_surprise_vol":   neg_surprise_vol,
            "neg_loss_q_range":   neg_loss_q_range,
            "max_loss_drop":      max_loss_drop,
            "settle_frac":        settle_frac,
            # Primary signals
            "signal_esp":           -esp_slope,
            "signal_h_drop":        h_drop,
            "signal_loss":          -ml,
            "signal_mink":          minkprob_20,
            "signal_minkpp":        minkpp_20,
            "signal_zlib":          zlib_ratio,
            "signal_surprise_drop": surprise_drop,
            "seq_len": seq, "n_tokens": n,
        }

# ── 3-Scale Calibration ──────────────────────────────────────────────────────
def calibrate_3scale(df, cols, n_buckets=16):
    df = df.copy()
    for c in cols:
        if c not in df.columns: continue
        df[f"{c}_raw"] = df[c]
        if "n_tokens" in df.columns:
            nb = min(n_buckets, df["n_tokens"].nunique())
            if nb > 1:
                df["_lb"] = pd.qcut(df["n_tokens"].fillna(0), q=nb, duplicates="drop", labels=False)
                m = df.groupby("_lb")[c].transform("mean")
                s = df.groupby("_lb")[c].transform("std").replace(0, 1)
                df[c] = (df[c] - m) / s
                df.drop(columns=["_lb"], inplace=True)
        if "subset" in df.columns and df["subset"].nunique() > 1:
            m = df.groupby("subset")[c].transform("mean")
            s = df.groupby("subset")[c].transform("std").replace(0, 1)
            df[c] = (df[c] - m) / s
    return df

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
    ext = CombinedExtractor(model, tok, max_len)
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

    # Calibrate all signal columns
    cal_cols = [c for c in fdf.columns if c not in ["seq_len","n_tokens"]]
    df = calibrate_3scale(df, cal_cols, CONFIG["position_buckets"])
    print(f"  ✓ 3-scale calibration applied ({len(cal_cols)} columns)")

    # Build ensemble scores (mean of z-normalised signals)
    for ens_name, members in ENSEMBLE_GROUPS.items():
        valid = [c for c in members if c in df.columns]
        if valid:
            df[ens_name] = df[valid].mean(axis=1)
            print(f"  + {ens_name} = mean({', '.join(valid)})")

    # Evaluate everything
    score_cols = [c for c in df.columns if c not in
                  ["text","is_member","subset","seq_len","n_tokens"] and not c.endswith("_raw")]
    print(f"\n{'─'*55}\n  TOP RESULTS: {tag}\n{'─'*55}")
    res = eval_auc(df, score_cols)
    for _, r in res.head(20).iterrows():
        tag_str = "ENS" if r["score"].startswith("ens_") else "   "
        m = "★" if r["score"] in ("signal_esp","signal_minkpp") or r["score"].startswith("ens_") else " "
        print(f"  {m} [{tag_str}] {r['score']:35s}  AUC={r['auc']:.4f}")

    best = res.iloc[0]["score"] if len(res) > 0 else "signal_minkpp"
    print(f"\n  Per-language  best={best}:")
    for _, sr in eval_per_subset(df, best).iterrows():
        print(f"    {sr['subset']:10s}  AUC={sr['auc']:.4f} (n={sr['n']})")

    # Ensemble summary
    ens_cols = [k for k in ENSEMBLE_GROUPS if k in df.columns]
    if ens_cols:
        print("\n  ENSEMBLE TABLE:")
        for _, r in eval_auc(df, ens_cols).iterrows():
            print(f"    {r['score']:35s}  AUC={r['auc']:.4f}")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    odir = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "./results"
    os.makedirs(odir, exist_ok=True)
    df.drop(columns=["text"], errors="ignore").to_parquet(
        f"{odir}/combined_{tag}_{ts}.parquet", index=False)
    with open(f"{odir}/combined_{tag}_{ts}.json", "w") as f:
        json.dump({"tag": tag, "best_feature": best,
                   "results_top20": res.head(20).to_dict("records")}, f, indent=2)
    print(f"  Saved → {odir}/combined_{tag}_{ts}.*")
    return df, res

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    print("\n" + "█"*60 + "\n  exp_04_Combined: Poisoned Chalice\n" + "█"*60)
    model, tok = load_model(CONFIG["model_name"])
    df_pc = load_pc_data(CONFIG["split"], CONFIG["sample_fraction"], CONFIG["seed"])
    run_benchmark(model, tok, df_pc, "PoisonedChalice", CONFIG["max_length"])
    del model, tok; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "═"*60 + "\n  DONE — exp_04_Combined\n" + "═"*60)
