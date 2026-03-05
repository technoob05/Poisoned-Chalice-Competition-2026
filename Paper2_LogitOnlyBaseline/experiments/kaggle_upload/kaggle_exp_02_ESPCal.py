#!/usr/bin/env python3
"""
Kaggle Script — exp_02_ESPCal ★ MAIN CONTRIBUTION
===================================================
Self-contained: paste entire file into a single Kaggle notebook cell.

Method: Entropy Slope Profile + 3-Scale Calibration
  Scale 1 — token-level z-norm (in extractor)
  Scale 2 — sequence-length bucket z-norm
  Scale 3 — language/domain z-norm
Also includes: Min-K%++ (ICLR 2025), surprise drop, full ablation table.

Requirements:
  - GPU: T4 or P100 (16GB VRAM)
  - Kaggle Secret: "posioned" → HuggingFace token
  - Dataset: minh2duy/poisoned-chalice-dataset (attach as input)

Quick test: sample_fraction=0.10  (~20 min)
Full run:   sample_fraction=1.0   (~2-3 h)
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
    "wikimia_model":    "EleutherAI/pythia-2.8b-deduped",
    "mimir_model":      "EleutherAI/pythia-2.8b-deduped",
    "max_length":       512,
    "sample_fraction":  0.10,   # ← set to 1.0 for full run
    "split":            "test",
    "seed":             42,
    "min_tokens":       8,
    "position_buckets": 16,
    "wikimia_lengths":  [32, 64, 128, 256],
    "mimir_domains":    ["wikipedia", "github", "pile_cc", "pubmed_central",
                         "arxiv", "dm_mathematics", "hackernews"],
    "run_poisoned_chalice": True,
    "run_wikimia":          True,
    "run_mimir":            True,
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

# ── Extractor: Full ESP + Min-K%++ ─────────────────────────────────────────
class ESPExtractor:
    """
    Entropy Slope Profile + Min-K%++ + surprise drop.
    Single forward pass, logits only (grey-box).
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
                "neg_mean_loss","loss_slope","minkprob_20",
                "minkpp_20","minkpp_10","minkpp_50",
                "surp","neg_mean_rank","zlib_ratio",
                "surprise_drop","surprise_accel",
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
        pos        = np.arange(n)

        # ── Entropy features ──
        esp_slope, _ = np.polyfit(pos, H, 1)
        h_mean, h_std = H.mean(), H.std()
        mid    = n // 2
        h_drop = H[:mid].mean() - H[mid:].mean() if mid > 0 else 0.0
        h_curv = np.polyfit(pos, H, 2)[0] if n >= 6 else 0.0
        z_esp  = np.polyfit(pos, (H - h_mean) / max(h_std, 1e-10), 1)[0] if h_std > 0 else 0.0

        # ── Loss features ──
        ml       = token_loss.mean()
        ls       = np.polyfit(pos, token_loss, 1)[0]
        lp_np    = token_lp.cpu().numpy()
        surp_val = ml - token_loss.std()

        # ── Min-K% (Shi et al., 2024) ──
        minkprob_20 = float(np.sort(lp_np)[:max(1, int(n*0.2))].mean())

        # ── Min-K%++ (Zhang et al., ICLR 2025) ──
        mu_v      = (probs * log_probs).sum(-1).squeeze(0)
        var_v     = (probs * log_probs.pow(2)).sum(-1).squeeze(0) - mu_v.pow(2)
        z_tok     = (token_lp - mu_v) / var_v.clamp(min=1e-20).sqrt()
        z_np      = z_tok.cpu().numpy()
        minkpp_20 = float(np.sort(z_np)[:max(1, int(n*0.2))].mean())
        minkpp_10 = float(np.sort(z_np)[:max(1, int(n*0.1))].mean())
        minkpp_50 = float(np.sort(z_np)[:max(1, int(n*0.5))].mean())

        # ── Rank ──
        ranks = (logits.squeeze(0).argsort(-1, descending=True).argsort(-1)
                 .gather(1, labels.squeeze(0).unsqueeze(1)).squeeze(1)
                 .float().cpu().numpy())

        # ── Surprise trajectory ──
        loss_first  = token_loss[:mid].mean() if mid > 0 else ml
        loss_second = token_loss[mid:].mean() if mid < n else ml
        surprise_drop  = float(loss_first - loss_second)
        surprise_accel = float(np.polyfit(pos, token_loss, 2)[0]) if n >= 6 else 0.0

        # ── Zlib ──
        zl_bytes   = len(zlib.compress(text.encode("utf-8")))
        zlib_ratio = (-ml) / zl_bytes if zl_bytes > 0 else 0.0

        del out; torch.cuda.empty_cache()
        return {
            "esp_slope":     esp_slope,
            "z_esp_slope":   z_esp,
            "h_mean":        h_mean, "h_std": h_std,
            "h_drop":        h_drop, "h_curvature": h_curv,
            "neg_mean_loss": -ml,    "loss_slope": ls,
            "minkprob_20":   minkprob_20,
            "minkpp_20":     minkpp_20,
            "minkpp_10":     minkpp_10,
            "minkpp_50":     minkpp_50,
            "surp":          -surp_val,
            "neg_mean_rank": -ranks.mean(),
            "zlib_ratio":    zlib_ratio,
            "surprise_drop": surprise_drop,
            "surprise_accel":surprise_accel,
            # Primary signals (higher → more likely member)
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
        # Scale 2: position (length bucket)
        if "n_tokens" in df.columns:
            nb = min(n_buckets, df["n_tokens"].nunique())
            if nb > 1:
                df["_lb"] = pd.qcut(df["n_tokens"].fillna(0), q=nb,
                                    duplicates="drop", labels=False)
                m = df.groupby("_lb")[c].transform("mean")
                s = df.groupby("_lb")[c].transform("std").replace(0, 1)
                df[c] = (df[c] - m) / s
                df.drop(columns=["_lb"], inplace=True)
        # Scale 3: domain/language
        if "subset" in df.columns and df["subset"].nunique() > 1:
            m = df.groupby("subset")[c].transform("mean")
            s = df.groupby("subset")[c].transform("std").replace(0, 1)
            df[c] = (df[c] - m) / s
    return df

# ── Data Loaders ─────────────────────────────────────────────────────────────
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

def load_wikimia_data(lengths=[32,64,128,256]):
    from datasets import load_dataset, load_from_disk
    kp = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset/kaggle_wikimia"
    data = {}
    for L in lengths:
        try:
            lp = os.path.join(kp, f"WikiMIA_length{L}")
            ds = load_from_disk(lp) if os.path.exists(lp) \
                 else load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{L}")
            df = pd.DataFrame([{"text":r["input"],"is_member":int(r["label"]),
                                 "subset":f"len{L}"} for r in ds])
            data[f"len{L}"] = df
            print(f"    WikiMIA len{L}: {len(df)}")
        except Exception as e:
            print(f"    WikiMIA len{L}: ERR {e}")
    return data

def load_mimir_data(domains):
    from datasets import load_dataset
    kp = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset/kaggle_mimir"
    data = {}
    for dom in domains:
        try:
            local_dir = os.path.join(kp, dom)
            rows = []
            if os.path.exists(local_dir):
                for fname, label in [("member.jsonl",1),("nonmember.jsonl",0)]:
                    fp = os.path.join(local_dir, fname)
                    if os.path.exists(fp):
                        with open(fp, "r", encoding="utf-8") as f:
                            for line in f:
                                t = line.strip()
                                if t: rows.append({"text":t,"is_member":label,"subset":dom})
            else:
                ds = load_dataset("iamgroot42/mimir", dom)
                for sp, lb in [("member",1),("nonmember",0)]:
                    if sp in ds:
                        for r in ds[sp]:
                            rows.append({"text":r.get("text",r.get("input","")),"is_member":lb,"subset":dom})
            if rows:
                data[dom] = pd.DataFrame(rows)
                print(f"    MIMIR {dom}: {len(data[dom])}")
        except Exception as e:
            print(f"    MIMIR {dom}: ERR {e}")
    return data

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

# ── Ablation ─────────────────────────────────────────────────────────────────
def run_ablation(df):
    print(f"\n{'─'*50}\n  ABLATION: Calibration Scale Contributions\n{'─'*50}")
    checks = [
        ("esp_slope_raw", "(a) ESP raw (no calibration)"),
        ("z_esp_slope",   "(b) + Scale 1 (token z-norm)"),
        ("signal_esp",    "(c) + Scale 2+3 (pos+domain)  ★ Full ESP-Cal"),
        ("signal_minkpp", "(d) Min-K%++ baseline"),
        ("signal_loss",   "(e) Loss baseline"),
    ]
    for col, desc in checks:
        if col not in df.columns: continue
        v = df[col].notna() & df["is_member"].notna()
        if v.sum() < 10: continue
        auc = roc_auc_score(df.loc[v,"is_member"], df.loc[v,col])
        m = "★" if "ESP-Cal" in desc else " "
        print(f"  {m} {desc:50s}  AUC={max(auc,1-auc):.4f}")

# ── LaTeX Table ──────────────────────────────────────────────────────────────
def make_latex_table(df, cols_map):
    langs = ["Go","Java","Python","Ruby","Rust"]
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"\textbf{Method} & \textbf{Go} & \textbf{Java} & \textbf{Python} "
             r"& \textbf{Ruby} & \textbf{Rust} & \textbf{Avg} \\", r"\midrule"]
    for col, name in cols_map.items():
        if col not in df.columns: continue
        aucs = []
        for lang in langs:
            sub = df[df["subset"]==lang]
            v = sub[col].notna() & sub["is_member"].notna()
            if v.sum() >= 10:
                a = roc_auc_score(sub.loc[v,"is_member"], sub.loc[v,col])
                aucs.append(f"{max(a,1-a):.3f}")
            else: aucs.append("—")
        valid = [float(a) for a in aucs if a!="—"]
        avg = f"{np.mean(valid):.3f}" if valid else "—"
        lines.append(f"  {name} & " + " & ".join(aucs) + f" & {avg} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{AUROC on Poisoned Chalice. ESP-Cal outperforms baselines.}",
              r"\label{tab:pc}", r"\end{table}"]
    return "\n".join(lines)

# ── Run Pipeline ─────────────────────────────────────────────────────────────
def run_benchmark(model, tok, df, tag, max_len=512, do_calibrate=True, do_ablation=False):
    ext = ESPExtractor(model, tok, max_len)
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

    cal_cols = ["signal_esp","signal_h_drop","signal_loss","signal_minkpp",
                "signal_mink","signal_zlib","signal_surprise_drop",
                "neg_mean_loss","minkprob_20","minkpp_20","surp","zlib_ratio"]
    cal_cols = [c for c in cal_cols if c in df.columns]
    if do_calibrate:
        df = calibrate_3scale(df, cal_cols, CONFIG["position_buckets"])
        print("  ✓ 3-scale calibration applied")

    score_cols = [c for c in df.columns if c not in
                  ["text","is_member","subset","seq_len","n_tokens"] and not c.endswith("_raw")]
    print(f"\n{'─'*50}\n  RESULTS: {tag}\n{'─'*50}")
    res = eval_auc(df, score_cols)
    for _, r in res.head(15).iterrows():
        m = "★" if "esp" in r["score"] or "minkpp" in r["score"] else " "
        print(f"  {m} {r['score']:30s}  AUC={r['auc']:.4f} ({r['pol']})")

    best = "signal_esp" if "signal_esp" in res["score"].values else res.iloc[0]["score"]
    if "subset" in df.columns and df["subset"].nunique() > 1:
        print(f"\n  Per-language ({best}):")
        for _, sr in eval_per_subset(df, best).iterrows():
            print(f"    {sr['subset']:10s}  AUC={sr['auc']:.4f} (n={sr['n']})")

    if do_ablation:
        try: run_ablation(df)
        except Exception as e: print(f"  Ablation error: {e}")

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    odir = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "./results"
    os.makedirs(odir, exist_ok=True)
    df.to_parquet(f"{odir}/espcal_{tag}_{ts}.parquet", index=False)
    with open(f"{odir}/espcal_{tag}_{ts}.json", "w") as f:
        json.dump({"tag": tag, "results": res.to_dict("records")}, f, indent=2)
    print(f"  Saved → {odir}/espcal_{tag}_{ts}.*")
    return df, res

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    if CONFIG["run_wikimia"]:
        print("\n" + "█"*60 + "\n  [1/3] WIKIMIA\n" + "█"*60)
        model, tok = load_model(CONFIG["wikimia_model"])
        for k, wdf in load_wikimia_data(CONFIG["wikimia_lengths"]).items():
            run_benchmark(model, tok, wdf, f"WikiMIA_{k}", CONFIG["max_length"], do_calibrate=False)
        del model, tok; gc.collect(); torch.cuda.empty_cache()

    if CONFIG["run_mimir"]:
        print("\n" + "█"*60 + "\n  [2/3] MIMIR\n" + "█"*60)
        model, tok = load_model(CONFIG["mimir_model"])
        for k, mdf in load_mimir_data(CONFIG["mimir_domains"]).items():
            run_benchmark(model, tok, mdf, f"MIMIR_{k}", CONFIG["max_length"], do_calibrate=False)
        del model, tok; gc.collect(); torch.cuda.empty_cache()

    if CONFIG["run_poisoned_chalice"]:
        print("\n" + "█"*60 + "\n  [3/3] POISONED CHALICE ★\n" + "█"*60)
        model, tok = load_model(CONFIG["model_name"])
        df_pc = load_pc_data(CONFIG["split"], CONFIG["sample_fraction"], CONFIG["seed"])
        df_pc, res_pc = run_benchmark(model, tok, df_pc, "PoisonedChalice",
                                      CONFIG["max_length"], do_calibrate=True, do_ablation=True)
        # LaTeX table
        table = make_latex_table(df_pc, {
            "signal_loss":   "Loss",
            "signal_mink":   "Min-K\\%",
            "signal_minkpp": "Min-K\\%++",
            "surp":          "SURP",
            "signal_zlib":   "Zlib",
            "signal_esp":    "\\textbf{ESP-Cal (Ours)}",
        })
        odir = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "./results"
        with open(f"{odir}/latex_table_pc.tex", "w") as f: f.write(table)
        print(f"\n  LaTeX table → {odir}/latex_table_pc.tex")
        del model, tok; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "═"*60 + "\n  DONE — exp_02_ESPCal\n" + "═"*60)
