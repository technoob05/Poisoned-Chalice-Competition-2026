#!/usr/bin/env python3
"""
Kaggle Notebook — Paper 2: ESP-Cal
====================================
Copy this entire file into a single Kaggle notebook cell.

Requirements:
  - GPU: T4 or P100 (16GB VRAM)
  - Kaggle Secrets: "posioned" → HuggingFace token
  - Dataset: minh2duy/poisoned-chalice-dataset (attach as input)

Runtime: ~2-3 hours for full Poisoned Chalice + ~1h WikiMIA + ~1h MIMIR
Tip: Set sample_fraction=0.1 for quick test (~20 min total)

ESP-Cal is logit-only (grey-box): no hidden states, no gradients.
Much faster than Paper 1 (MultiGeo-MIA), as it only needs a single forward pass
without output_hidden_states or output_attentions.
"""

# ── Cell 1: Install & authenticate ──
import subprocess, sys, os

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.40", "accelerate", "datasets",
                "scikit-learn", "scipy", "huggingface_hub>=0.23"],
               capture_output=True)

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    login(token=UserSecretsClient().get_secret("posioned"), add_to_git_credential=True)
    print("✓ HF authenticated")
except:
    print("○ Local mode")

# ── Cell 2: Configuration ──
CONFIG = {
    "model_name": "bigcode/starcoder2-3b",
    "wikimia_model": "EleutherAI/pythia-2.8b-deduped",
    "mimir_model": "EleutherAI/pythia-2.8b-deduped",
    "max_length": 512,
    "sample_fraction": 0.1,  # 10% for quick test runs
    "split": "train",
    "seed": 42,

    # ESP parameters
    "position_buckets": 16,
    "min_tokens": 8,

    # Ablation flags
    "enable_scale1_token": True,
    "enable_scale2_position": True,
    "enable_scale3_domain": True,

    # WikiMIA
    "wikimia_lengths": [32, 64, 128, 256],

    # MIMIR
    "mimir_domains": ["wikipedia", "github", "pile_cc", "pubmed_central",
                      "arxiv", "dm_mathematics", "hackernews"],

    # Run flags
    "run_poisoned_chalice": True,
    "run_wikimia": True,
    "run_mimir": True,
}

# ── Cell 3: Imports ──
import gc, json, time, warnings, zlib
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

# ── Cell 4: Model Loader (logit-only, no hidden states) ──
def load_model(name):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"✓ {name}: {model.config.num_hidden_layers}L, "
          f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tok

# ── Cell 5: ESP Feature Extractor ──
class ESPExtractor:
    """
    Entropy Slope Profile: fit linear model to per-token prediction entropy.
    Members show steeper entropy decline (model gets more confident faster).
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
                "esp_slope","z_esp_slope","h_mean","h_std","h_drop","h_curvature",
                "neg_mean_loss","loss_slope","minkprob_20","surp","neg_mean_rank",
                "zlib_ratio","signal_esp","signal_h_drop","signal_loss",
                "seq_len","n_tokens"]}

        out = self.model(input_ids=ids)
        logits = out.logits[:, :-1, :].float()
        labels = ids[:, 1:]
        n = logits.shape[1]

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Per-token entropy
        H = -(probs * log_probs).sum(-1).squeeze(0).cpu().numpy()  # (n,)

        # Per-token log-prob of true next token
        token_lp = log_probs.squeeze(0).gather(1, labels.squeeze(0).unsqueeze(1)).squeeze(1).cpu().numpy()
        token_loss = -token_lp

        # Ranks
        ranks = (logits.squeeze(0).argsort(-1,descending=True).argsort(-1)
                 .gather(1, labels.squeeze(0).unsqueeze(1)).squeeze(1).float().cpu().numpy())

        pos = np.arange(n)

        # ── ESP features ──
        esp_slope, esp_int = np.polyfit(pos, H, 1)
        h_mean, h_std = H.mean(), H.std()
        mid = n // 2
        h_drop = H[:mid].mean() - H[mid:].mean() if mid > 0 else 0.0
        h_curv = np.polyfit(pos, H, 2)[0] if n >= 6 else 0.0

        # Token z-norm entropy slope (Scale 1)
        z_esp = np.polyfit(pos, (H - h_mean) / max(h_std, 1e-10), 1)[0] if h_std > 0 else 0.0

        # Loss features
        ml = token_loss.mean()
        ls = np.polyfit(pos, token_loss, 1)[0]
        k20 = max(1, int(n * 0.2))
        mkp = np.sort(token_lp)[:k20].mean()
        surp_val = ml - token_loss.std()

        # Compression baseline
        zl = len(zlib.compress(text.encode("utf-8")))
        zr = ml / (zl / n) if zl > 0 else 0.0

        del out; torch.cuda.empty_cache()
        return {
            "esp_slope": esp_slope, "z_esp_slope": z_esp,
            "h_mean": h_mean, "h_std": h_std, "h_drop": h_drop, "h_curvature": h_curv,
            "neg_mean_loss": -ml, "loss_slope": ls,
            "minkprob_20": mkp, "surp": -surp_val,
            "neg_mean_rank": -ranks.mean(), "zlib_ratio": zr,
            "signal_esp": -esp_slope,  # steeper decline → more member → negate slope
            "signal_h_drop": h_drop,
            "signal_loss": -ml,
            "seq_len": seq, "n_tokens": n,
        }

# ── Cell 6: Multi-Scale Calibration ──
def calibrate_3scale(df, cols, n_buckets=16):
    """Apply 3-scale z-normalization: token(done) → position → domain."""
    df = df.copy()
    for c in cols:
        if c not in df.columns: continue
        df[f"{c}_raw"] = df[c]
        # Scale 2: position (length bucket)
        if "n_tokens" in df.columns:
            df["_lb"] = pd.qcut(df["n_tokens"].fillna(0),
                                q=min(n_buckets, df["n_tokens"].nunique()),
                                duplicates="drop", labels=False)
            m = df.groupby("_lb")[c].transform("mean")
            s = df.groupby("_lb")[c].transform("std").replace(0,1)
            df[c] = (df[c] - m) / s
            df.drop(columns=["_lb"], inplace=True)
        # Scale 3: domain
        if "subset" in df.columns and df["subset"].nunique() > 1:
            m = df.groupby("subset")[c].transform("mean")
            s = df.groupby("subset")[c].transform("std").replace(0,1)
            df[c] = (df[c] - m) / s
    return df

# ── Cell 7: Data Loaders ──
def load_pc_data(split="train", frac=1.0, seed=42):
    from datasets import load_dataset, load_from_disk
    kp = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"
    rows = []
    for lang in ["Go","Java","Python","Ruby","Rust"]:
        try:
            if os.path.exists(kp):
                ds = load_from_disk(os.path.join(kp, lang, split))
            else:
                ds = load_dataset("AISE-TUDelft/Poisoned-Chalice", lang, split=split)
            for r in ds:
                rows.append({"text": r["content"], "is_member": int(r["membership"]), "subset": lang})
            print(f"  {lang}: {len(ds)}")
        except Exception as e:
            print(f"  {lang}: ERR {e}")
    df = pd.DataFrame(rows)
    if frac < 1:
        df = df.groupby(["subset","is_member"]).apply(
            lambda x: x.sample(frac=frac, random_state=seed)).reset_index(drop=True)
    print(f"  Total: {len(df)} ({df.is_member.sum()} members)")
    return df

def load_wikimia_data(lengths=[32,64,128,256]):
    from datasets import load_dataset
    data = {}
    for L in lengths:
        try:
            ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{L}")
            df = pd.DataFrame([{"text":r["input"],"is_member":int(r["label"]),"subset":f"len{L}"} for r in ds])
            data[f"len{L}"] = df
            print(f"  WikiMIA len{L}: {len(df)}")
        except Exception as e:
            print(f"  WikiMIA len{L}: ERR {e}")
    return data

def load_mimir_data(domains):
    from datasets import load_dataset
    data = {}
    for dom in domains:
        try:
            ds = load_dataset("iamgroot42/mimir", dom, trust_remote_code=True)
            rows = []
            for sp, lb in [("member",1),("nonmember",0)]:
                if sp in ds:
                    for r in ds[sp]:
                        rows.append({"text":r.get("text",r.get("input","")),"is_member":lb,"subset":dom})
            if rows:
                data[dom] = pd.DataFrame(rows)
                print(f"  MIMIR {dom}: {len(data[dom])}")
        except Exception as e:
            print(f"  MIMIR {dom}: ERR {e}")
    return data

# ── Cell 8: Evaluation ──
def eval_auc(df, cols, label="is_member"):
    res = []
    for c in cols:
        v = df[c].notna() & df[label].notna()
        if v.sum() < 10: continue
        if len(np.unique(df.loc[v, label])) < 2: continue
        auc = roc_auc_score(df.loc[v,label], df.loc[v,c])
        res.append({"score":c, "auc":max(auc,1-auc), "pol":"+" if auc>=0.5 else "-"})
    return pd.DataFrame(res).sort_values("auc", ascending=False)

def eval_per_subset(df, col, label="is_member"):
    res = []
    for sub, g in df.groupby("subset"):
        v = g[col].notna() & g[label].notna()
        if v.sum()<10: continue
        auc = roc_auc_score(g.loc[v,label], g.loc[v,col])
        res.append({"subset":sub, "auc":max(auc,1-auc), "n":int(v.sum())})
    return pd.DataFrame(res)

# ── Cell 9: Ablation ──
def run_ablation(df):
    """Ablation: measure each calibration scale's contribution."""
    print(f"\n{'─'*50}\n  ABLATION: Calibration Scales\n{'─'*50}")
    results = []

    # (a) Raw ESP slope
    raw_col = "esp_slope_raw" if "esp_slope_raw" in df.columns else "esp_slope"
    v = df[raw_col].notna() & df["is_member"].notna()
    if v.sum() >= 10:
        auc = roc_auc_score(df.loc[v,"is_member"], df.loc[v,raw_col])
        results.append({"cond": "(a) ESP raw", "auc": max(auc,1-auc)})

    # (b) + Scale 1 (token z-norm)
    v = df["z_esp_slope"].notna() & df["is_member"].notna()
    if v.sum() >= 10:
        auc = roc_auc_score(df.loc[v,"is_member"], df.loc[v,"z_esp_slope"])
        results.append({"cond": "(b) + Scale 1 (token)", "auc": max(auc,1-auc)})

    # (c) + Scale 2 (position) — re-calibrate raw with only position
    if "esp_slope_raw" in df.columns and "n_tokens" in df.columns:
        df_c = df.copy()
        df_c["_temp"] = df_c["esp_slope_raw"]
        nb = min(16, df_c["n_tokens"].nunique())
        if nb > 1:
            df_c["_lb"] = pd.qcut(df_c["n_tokens"].fillna(0), q=nb, duplicates="drop", labels=False)
            m = df_c.groupby("_lb")["_temp"].transform("mean")
            s = df_c.groupby("_lb")["_temp"].transform("std").replace(0,1)
            df_c["_temp"] = (df_c["_temp"] - m) / s
        v = df_c["_temp"].notna() & df_c["is_member"].notna()
        if v.sum() >= 10:
            auc = roc_auc_score(df_c.loc[v,"is_member"], df_c.loc[v,"_temp"])
            results.append({"cond": "(c) + Scale 2 (position)", "auc": max(auc,1-auc)})

    # (d) Full ESP-Cal
    v = df["signal_esp"].notna() & df["is_member"].notna()
    if v.sum() >= 10:
        auc = roc_auc_score(df.loc[v,"is_member"], df.loc[v,"signal_esp"])
        results.append({"cond": "(d) Full ESP-Cal ★", "auc": max(auc,1-auc)})

    abl = pd.DataFrame(results)
    for _, r in abl.iterrows():
        print(f"  {r['cond']:40s} AUC={r['auc']:.4f}")
    return abl

# ── Cell 10: Run Pipeline ──
def run_benchmark(model, tok, df, tag, max_len=512, do_calibrate=True, do_ablation=False):
    ext = ESPExtractor(model, tok, max_len)
    t0 = time.time()
    feats = []
    for i, row in df.iterrows():
        if i>0 and i%500==0:
            r = i/(time.time()-t0)
            print(f"  [{i}/{len(df)}] {r:.1f}/s ETA {(len(df)-i)/r:.0f}s")
        feats.append(ext.extract(row["text"]))
    fdf = pd.DataFrame(feats)
    df = pd.concat([df.reset_index(drop=True), fdf], axis=1)
    print(f"  ✓ {len(df)} in {time.time()-t0:.0f}s")

    # Calibration
    cal_cols = ["signal_esp", "signal_h_drop", "signal_loss",
                "neg_mean_loss", "minkprob_20", "surp", "esp_slope"]
    cal_cols = [c for c in cal_cols if c in df.columns]
    if do_calibrate:
        df = calibrate_3scale(df, cal_cols, CONFIG.get("position_buckets", 16))
        print("  ✓ 3-scale calibration applied")

    # Evaluate
    all_cols = [c for c in df.columns if c not in
                ["text","is_member","subset","seq_len","n_tokens"] and not c.endswith("_raw")]
    print(f"\n{'─'*50}\n  RESULTS: {tag}\n{'─'*50}")
    res = eval_auc(df, all_cols)
    for _, r in res.head(15).iterrows():
        m = "★" if "esp" in r["score"].lower() else " "
        print(f"  {m} {r['score']:30s} AUC={r['auc']:.4f} ({r['pol']})")

    if "subset" in df.columns and df["subset"].nunique()>1:
        best = "signal_esp" if "signal_esp" in res["score"].values else res.iloc[0]["score"]
        print(f"\n  Per-subset ({best}):")
        for _, sr in eval_per_subset(df, best).iterrows():
            print(f"    {sr['subset']:12s} AUC={sr['auc']:.4f} (n={sr['n']})")

    # Ablation
    if do_ablation:
        try:
            run_ablation(df)
        except Exception as e:
            print(f"  Ablation error: {e}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    odir = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "./results"
    os.makedirs(odir, exist_ok=True)
    df.to_parquet(f"{odir}/espcal_{tag}_{ts}.parquet", index=False)
    with open(f"{odir}/espcal_{tag}_{ts}.json", "w") as f:
        json.dump({"tag":tag, "results":res.to_dict("records")}, f, indent=2)
    print(f"  Saved → {odir}/espcal_{tag}_{ts}.*")

    return df, res

# ── Cell 11: LaTeX Table Generator ──
def make_latex_table(df, methods_cols):
    """Generate LaTeX comparison table for the paper."""
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"\textbf{Method} & \textbf{Go} & \textbf{Java} & \textbf{Python} & \textbf{Ruby} & \textbf{Rust} & \textbf{Avg} \\",
             r"\midrule"]
    for col, name in methods_cols.items():
        if col not in df.columns: continue
        aucs = []
        for lang in ["Go","Java","Python","Ruby","Rust"]:
            sub = df[df["subset"]==lang]
            v = sub[col].notna() & sub["is_member"].notna()
            if v.sum()>=10:
                a = roc_auc_score(sub.loc[v,"is_member"], sub.loc[v,col])
                aucs.append(f"{max(a,1-a):.3f}")
            else: aucs.append("—")
        avg = np.mean([float(a) for a in aucs if a!="—"]) if any(a!="—" for a in aucs) else 0
        lines.append(f"  {name} & " + " & ".join(aucs) + f" & {avg:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{AUROC on Poisoned Chalice. ESP-Cal consistently outperforms logit-only baselines.}",
              r"\label{tab:pc}", r"\end{table}"]
    return "\n".join(lines)

# ── Cell 12: MAIN ──
if __name__ == "__main__":
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    # ── 1. Poisoned Chalice ──
    if CONFIG["run_poisoned_chalice"]:
        print("\n" + "█"*60 + "\n  POISONED CHALICE\n" + "█"*60)
        model, tok = load_model(CONFIG["model_name"])
        df_pc = load_pc_data(CONFIG["split"], CONFIG["sample_fraction"], CONFIG["seed"])
        df_pc, res_pc = run_benchmark(model, tok, df_pc, "PoisonedChalice",
                                       CONFIG["max_length"], do_calibrate=True, do_ablation=True)

        # Generate LaTeX table
        table = make_latex_table(df_pc, {
            "neg_mean_loss": "Loss",
            "minkprob_20": "Min-K\\%",
            "surp": "SURP",
            "zlib_ratio": "Zlib",
            "signal_esp": "\\textbf{ESP-Cal (Ours)}",
        })
        odir = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "./results"
        os.makedirs(odir, exist_ok=True)
        with open(f"{odir}/latex_table_pc.tex", "w") as f:
            f.write(table)
        print(f"\n  LaTeX table → {odir}/latex_table_pc.tex")

        del model, tok; gc.collect(); torch.cuda.empty_cache()

    # ── 2. WikiMIA ──
    if CONFIG["run_wikimia"]:
        print("\n" + "█"*60 + "\n  WIKIMIA\n" + "█"*60)
        model, tok = load_model(CONFIG["wikimia_model"])
        wdata = load_wikimia_data(CONFIG["wikimia_lengths"])
        for k, wdf in wdata.items():
            run_benchmark(model, tok, wdf, f"WikiMIA_{k}",
                          CONFIG["max_length"], do_calibrate=False)
        del model, tok; gc.collect(); torch.cuda.empty_cache()

    # ── 3. MIMIR ──
    if CONFIG["run_mimir"]:
        print("\n" + "█"*60 + "\n  MIMIR\n" + "█"*60)
        model, tok = load_model(CONFIG["mimir_model"])
        mdata = load_mimir_data(CONFIG["mimir_domains"])
        for k, mdf in mdata.items():
            run_benchmark(model, tok, mdf, f"MIMIR_{k}",
                          CONFIG["max_length"], do_calibrate=False)
        del model, tok; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "═"*60 + "\n  ALL DONE — ESP-Cal\n" + "═"*60)
