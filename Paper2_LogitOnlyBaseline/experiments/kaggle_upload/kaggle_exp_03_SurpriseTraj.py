#!/usr/bin/env python3
"""
Kaggle Script — exp_03_SurpriseTraj (Surprise Trajectory Features)
===================================================================
Self-contained: paste entire file into a single Kaggle notebook cell.

Novel: characterise HOW per-token loss (surprise) evolves over the sequence.
  surprise_drop      — first - second half mean loss  (members: model recognises → loss drops)
  surprise_accel     — quadratic curvature of loss    (members: concave = fast initial drop)
  neg_surprise_vol   — -std(Δloss)                    (members: more predictable)
  neg_loss_q_range   — -(q90 - q10)                   (members: more uniform)
  max_loss_drop      — largest single-token drop       (members: recognition moments)
  settle_frac        — frac of 2nd-half below median   (members: loss settles low)

3-scale calibration applied to all features.

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

# ── Extractor: Surprise Trajectory Features ──────────────────────────────────
class SurpriseTrajExtractor:
    """
    Characterise HOW per-token loss evolves across the sequence.
    Also includes ESP and loss baseline for fair comparison.
    """
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
                "surprise_drop","surprise_accel","neg_surprise_vol",
                "neg_loss_q_range","max_loss_drop","settle_frac",
                "signal_surprise_drop",
                "esp_slope","neg_esp_slope","neg_mean_loss","minkprob_20",
                "seq_len","n_tokens"]}

        out       = self.model(input_ids=ids)
        logits    = out.logits[:, :-1, :].float()
        labels    = ids[:, 1:]
        n         = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=-1)

        token_lp   = log_probs.squeeze(0).gather(
                         1, labels.squeeze(0).unsqueeze(1)).squeeze(1).cpu().numpy()
        token_loss = -token_lp
        pos        = np.arange(n)
        mid        = n // 2
        ml         = token_loss.mean()

        # ── Surprise trajectory features ──
        loss_first  = token_loss[:mid].mean() if mid > 0 else ml
        loss_second = token_loss[mid:].mean() if mid < n else ml
        surprise_drop  = float(loss_first - loss_second)
        surprise_accel = float(np.polyfit(pos, token_loss, 2)[0]) if n >= 6 else 0.0

        diffs = np.diff(token_loss)
        neg_surprise_vol = float(-diffs.std()) if len(diffs) > 1 else 0.0

        neg_loss_q_range = float(
            -(np.quantile(token_loss,0.9) - np.quantile(token_loss,0.1))
        ) if n > 2 else 0.0

        max_loss_drop = float(-diffs.min()) if len(diffs) > 0 else 0.0

        median_loss = np.median(token_loss)
        settle_frac = float((token_loss[mid:] < median_loss).mean()) if mid < n else 0.5

        # ── ESP slope (comparison baseline) ──
        probs = F.softmax(logits, dim=-1)
        H     = -(probs * F.log_softmax(logits, dim=-1)).sum(-1).squeeze(0).cpu().numpy()
        esp_slope, _ = np.polyfit(pos, H, 1)

        # ── Min-K% (comparison) ──
        mkp = float(np.sort(token_lp)[:max(1, int(n*0.2))].mean())

        del out; torch.cuda.empty_cache()
        return {
            "surprise_drop":      surprise_drop,
            "surprise_accel":     surprise_accel,
            "neg_surprise_vol":   neg_surprise_vol,
            "neg_loss_q_range":   neg_loss_q_range,
            "max_loss_drop":      max_loss_drop,
            "settle_frac":        settle_frac,
            "signal_surprise_drop": surprise_drop,
            "esp_slope":          esp_slope,
            "neg_esp_slope":      -esp_slope,
            "neg_mean_loss":      -ml,
            "minkprob_20":        mkp,
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
                df["_lb"] = pd.qcut(df["n_tokens"].fillna(0), q=nb,
                                    duplicates="drop", labels=False)
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
    ext = SurpriseTrajExtractor(model, tok, max_len)
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

    cal_cols = [c for c in fdf.columns if c not in ["seq_len","n_tokens"]]
    df = calibrate_3scale(df, cal_cols, CONFIG["position_buckets"])
    print("  ✓ 3-scale calibration applied")

    score_cols = [c for c in cal_cols if c in df.columns]
    print(f"\n{'─'*50}\n  RESULTS: {tag}\n{'─'*50}")
    res = eval_auc(df, score_cols)
    for _, r in res.iterrows():
        m = "★" if "surprise" in r["score"] or "settle" in r["score"] else " "
        print(f"  {m} {r['score']:28s}  AUC={r['auc']:.4f} ({r['pol']})")

    best = res.iloc[0]["score"] if len(res) > 0 else "surprise_drop"
    print(f"\n  Per-language ({best}):")
    for _, sr in eval_per_subset(df, best).iterrows():
        print(f"    {sr['subset']:10s}  AUC={sr['auc']:.4f} (n={sr['n']})")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    odir = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "./results"
    os.makedirs(odir, exist_ok=True)
    df.to_parquet(f"{odir}/surptraj_{tag}_{ts}.parquet", index=False)
    with open(f"{odir}/surptraj_{tag}_{ts}.json", "w") as f:
        json.dump({"tag": tag, "results": res.to_dict("records")}, f, indent=2)
    print(f"  Saved → {odir}/surptraj_{tag}_{ts}.*")
    return df, res

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    print("\n" + "█"*60 + "\n  exp_03_SurpriseTraj: Poisoned Chalice\n" + "█"*60)
    model, tok = load_model(CONFIG["model_name"])
    df_pc = load_pc_data(CONFIG["split"], CONFIG["sample_fraction"], CONFIG["seed"])
    run_benchmark(model, tok, df_pc, "PoisonedChalice", CONFIG["max_length"])
    del model, tok; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "═"*60 + "\n  DONE — exp_03_SurpriseTraj\n" + "═"*60)
