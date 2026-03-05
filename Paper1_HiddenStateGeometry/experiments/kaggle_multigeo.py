#!/usr/bin/env python3
"""
Kaggle Notebook — Paper 1: MultiGeo-MIA
========================================
Copy this entire file into a single Kaggle notebook cell.

Requirements:
  - GPU: T4 or P100 (16GB VRAM)
  - Kaggle Secrets: "posioned" → HuggingFace token
  - Dataset: minh2duy/poisoned-chalice-dataset (attach as input)

Runtime: ~4-6 hours for full Poisoned Chalice + ~2h WikiMIA + ~2h MIMIR
Tip: Set sample_fraction=0.1 for quick test (~30 min total)
"""

# ── Cell 1: Install & authenticate ──
import subprocess, sys, os

# ── Redirect HF cache to RAM-backed tmpfs (bypasses Kaggle disk quota) ──
_shm_free = 0
try:
    _st = os.statvfs("/dev/shm")
    _shm_free = (_st.f_bavail * _st.f_frsize) / 1e9
except Exception:
    pass
if _shm_free > 30:
    _hf_root = "/dev/shm/hf_cache"
else:
    _hf_root = "/tmp/hf_cache"
os.makedirs(os.path.join(_hf_root, "hub"), exist_ok=True)
os.environ["HF_HOME"] = _hf_root
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_root, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_hf_root, "hub")
print(f"✓ HF cache → {_hf_root}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.40", "accelerate", "datasets",
                "scikit-learn", "scipy", "huggingface_hub>=0.23"],
               capture_output=True)

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    user_secrets = UserSecretsClient()
    token = None
    for secret_name in ["HF_TOKEN", "posioned"]:
        try:
            token = user_secrets.get_secret(secret_name)
            if token:
                print(f"✓ Found secret: {secret_name}")
                break
        except Exception:
            continue
    if token:
        login(token=token, add_to_git_credential=True)
        print("✓ HuggingFace authenticated")
    else:
        print("○ No HF secret found")
except Exception as e:
    print(f"○ No Kaggle secrets: {e}")

# ── Cell 2: Configuration ──
CONFIG = {
    "model_name": "bigcode/starcoder2-3b",     # Poisoned Chalice target model
    "wikimia_model": "EleutherAI/pythia-2.8b-deduped",   # WikiMIA model
    "mimir_model": "EleutherAI/pythia-2.8b-deduped",     # MIMIR model
    "max_length": 512,
    "sample_fraction": 0.1,  # 10% for quick test runs
    "split": "test",         # test split only
    "seed": 42,

    # WikiMIA lengths to test
    "wikimia_lengths": [32, 64, 128, 256],

    # MIMIR domains
    "mimir_domains": ["wikipedia", "github", "pile_cc", "pubmed_central",
                      "arxiv", "dm_mathematics", "hackernews"],

    # Run flags (set False to skip a benchmark)
    "run_poisoned_chalice": True,
    "run_wikimia": True,
    "run_mimir": True,
}

# ── Cell 3: Imports ──
import gc
import json
import time
import warnings
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

# ── Cell 4: Model Loader ──
def load_model(name, max_length=512):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, output_hidden_states=True,
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"✓ {name}: {n_layers}L, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tok, n_layers

# ── Cell 5: MultiGeo Extractor ──
class MultiGeoExtractor:
    def __init__(self, model, tok, n_layers, max_len=512):
        self.model, self.tok, self.nl = model, tok, n_layers
        self.max_len = max_len
        self.dev = next(model.parameters()).device
        mid = n_layers // 2
        self.mid_idx = mid
        k = min(6, n_layers)
        self.cascade_idx = [int(i*(n_layers-1)/(k-1)) for i in range(k)]
        self.attn_idx = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]

    @torch.no_grad()
    def extract(self, text):
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_len, padding=False)
        ids = enc["input_ids"].to(self.dev)
        if ids.shape[1] < 4:
            return {k: np.nan for k in [
                "mag_mid_norm","dim_eff_rank","dyn_mean_drift","rout_mean_entropy",
                "signal_magnitude","signal_dimensionality","signal_dynamics","signal_routing",
                "loss","seq_len"]}

        out = self.model(input_ids=ids,
                         attention_mask=enc.get("attention_mask", torch.ones_like(ids)).to(self.dev),
                         output_hidden_states=True, output_attentions=True)
        hs = out.hidden_states
        attn = out.attentions

        # Axis 1: Magnitude
        mid_hs = hs[self.mid_idx+1].float()
        mid_norm = torch.norm(mid_hs[0], dim=-1).mean().item()

        # Axis 2: Dimensionality
        H = hs[self.mid_idx+1].float().squeeze(0)
        try:
            k_svd = min(50, *H.shape)
            S = torch.linalg.svdvals(H)[:k_svd]
            S_n = S / S.sum()
            eff_rank = np.exp(-(S_n * torch.log(S_n.clamp(min=1e-10))).sum().item())
        except:
            eff_rank = 0.0

        # Axis 3: Dynamics
        drifts = []
        for i in range(len(self.cascade_idx)-1):
            h1 = hs[self.cascade_idx[i]+1].float().squeeze(0).mean(0)
            h2 = hs[self.cascade_idx[i+1]+1].float().squeeze(0).mean(0)
            drifts.append(1 - F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item())
        mean_drift = np.mean(drifts) if drifts else 0.0

        # Axis 4: Routing
        entropies = []
        for li in self.attn_idx:
            if li < len(attn):
                a = attn[li].float().squeeze(0).clamp(min=1e-10)
                entropies.append(-(a * torch.log(a)).sum(-1).mean().item())
        mean_entropy = np.mean(entropies) if entropies else 0.0

        # Loss
        sl = out.logits[:, :-1, :].contiguous()
        loss = -F.cross_entropy(sl.view(-1, sl.size(-1)), ids[:, 1:].contiguous().view(-1)).item()

        del out; torch.cuda.empty_cache()
        return {
            "mag_mid_norm": mid_norm, "dim_eff_rank": eff_rank,
            "dyn_mean_drift": mean_drift, "rout_mean_entropy": mean_entropy,
            "signal_magnitude": -mid_norm, "signal_dimensionality": -eff_rank,
            "signal_dynamics": -mean_drift, "signal_routing": -mean_entropy,
            "loss": loss, "seq_len": ids.shape[1],
        }

# ── Cell 6: Data Loader ──
def load_pc_data(split="test", frac=1.0, seed=42):
    from datasets import load_dataset, load_from_disk
    kp = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset/poisoned_chalice_dataset"
    rows = []
    for lang in ["Go","Java","Python","Ruby","Rust"]:
        try:
            local_path = os.path.join(kp, lang, split)
            if os.path.exists(local_path):
                ds = load_from_disk(local_path)
            else:
                ds = load_dataset("AISE-TUDelft/Poisoned-Chalice", lang, split=split)
            count = 0
            for r in ds:
                text = r.get("content") or ""
                if not text or not text.strip():
                    continue
                mem = r["membership"]
                is_mem = 1 if (mem == "member" or mem == 1) else 0
                rows.append({"text": text, "is_member": is_mem, "subset": lang})
                count += 1
            print(f"    {lang}: {len(ds)} samples (kept {count})")
        except Exception as e:
            print(f"    {lang}: ERR {e}")
    df = pd.DataFrame(rows)
    if frac < 1:
        df = df.groupby(["subset","is_member"]).apply(
            lambda x: x.sample(frac=frac, random_state=seed)).reset_index(drop=True)
        print(f"  Sampled {frac:.0%} → {len(df)} rows")
    print(f"  Total: {len(df)} ({df.is_member.sum()} members)")
    return df

def load_wikimia_data(lengths=[32,64,128,256]):
    from datasets import load_dataset, load_from_disk
    kp = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset/kaggle_wikimia"
    data = {}
    for L in lengths:
        try:
            local_path = os.path.join(kp, f"WikiMIA_length{L}")
            if os.path.exists(local_path):
                ds = load_from_disk(local_path)
            else:
                ds = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{L}")
            df = pd.DataFrame([{"text":r["input"],"is_member":int(r["label"]),"subset":f"len{L}"} for r in ds])
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
                import json as _json
                for fname, label in [("member.jsonl", 1), ("nonmember.jsonl", 0)]:
                    fpath = os.path.join(local_dir, fname)
                    if os.path.exists(fpath):
                        with open(fpath, "r", encoding="utf-8") as f:
                            for line in f:
                                text = line.strip()
                                if text:
                                    rows.append({"text": text, "is_member": label, "subset": dom})
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

# ── Cell 7: Evaluation ──
def rank_avg(df, cols):
    R = pd.DataFrame({c: df[c].rank(pct=True) for c in cols if c in df.columns and df[c].notna().sum()>0})
    return R.mean(axis=1)

def znorm_per_lang(df, cols):
    df = df.copy()
    for c in cols:
        if c not in df.columns: continue
        m = df.groupby("subset")[c].transform("mean")
        s = df.groupby("subset")[c].transform("std").replace(0,1)
        df[c] = (df[c] - m) / s
    return df

def eval_auc(df, cols, label="is_member"):
    res = []
    for c in cols:
        v = df[c].notna() & df[label].notna()
        if v.sum() < 10: continue
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

# ── Cell 8: Run Pipeline ──
def run_benchmark(model, tok, n_layers, df, tag, max_len=512, do_znorm=True):
    ext = MultiGeoExtractor(model, tok, n_layers, max_len)
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

    sig = ["signal_magnitude","signal_dimensionality","signal_dynamics","signal_routing"]
    if do_znorm and "subset" in df.columns and df["subset"].nunique()>1:
        df = znorm_per_lang(df, [c for c in fdf.columns if c!="seq_len"])

    df["multigeo_4axis"] = rank_avg(df, sig)
    all_cols = sig + ["multigeo_4axis", "loss"]

    print(f"\n{'─'*50}\n  RESULTS: {tag}\n{'─'*50}")
    res = eval_auc(df, all_cols)
    for _, r in res.iterrows():
        m = "★" if r["score"]=="multigeo_4axis" else " "
        print(f"  {m} {r['score']:30s} AUC={r['auc']:.4f} ({r['pol']})")

    if "subset" in df.columns and df["subset"].nunique()>1:
        print(f"\n  Per-subset (multigeo_4axis):")
        for _, sr in eval_per_subset(df, "multigeo_4axis").iterrows():
            print(f"    {sr['subset']:12s} AUC={sr['auc']:.4f} (n={sr['n']})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    odir = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "./results"
    os.makedirs(odir, exist_ok=True)
    df.to_parquet(f"{odir}/multigeo_{tag}_{ts}.parquet", index=False)
    with open(f"{odir}/multigeo_{tag}_{ts}.json", "w") as f:
        json.dump({"tag":tag, "results":res.to_dict("records")}, f, indent=2)
    return df, res

# ── Cell 9: MAIN ──
if __name__ == "__main__":
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    # ── 1. WikiMIA ──
    if CONFIG["run_wikimia"]:
        print("\n" + "█"*60 + "\n  [1/3] WIKIMIA\n" + "█"*60)
        model, tok, nl = load_model(CONFIG["wikimia_model"], CONFIG["max_length"])
        wdata = load_wikimia_data(CONFIG["wikimia_lengths"])
        for k, wdf in wdata.items():
            run_benchmark(model, tok, nl, wdf, f"WikiMIA_{k}", CONFIG["max_length"], do_znorm=False)
        del model, tok; gc.collect(); torch.cuda.empty_cache()

    # ── 2. MIMIR ──
    if CONFIG["run_mimir"]:
        print("\n" + "█"*60 + "\n  [2/3] MIMIR\n" + "█"*60)
        model, tok, nl = load_model(CONFIG["mimir_model"], CONFIG["max_length"])
        mdata = load_mimir_data(CONFIG["mimir_domains"])
        for k, mdf in mdata.items():
            run_benchmark(model, tok, nl, mdf, f"MIMIR_{k}", CONFIG["max_length"], do_znorm=False)
        del model, tok; gc.collect(); torch.cuda.empty_cache()

    # ── 3. Poisoned Chalice (LAST — competition target) ──
    if CONFIG["run_poisoned_chalice"]:
        print("\n" + "█"*60 + "\n  [3/3] POISONED CHALICE (test split)\n" + "█"*60)
        model, tok, nl = load_model(CONFIG["model_name"], CONFIG["max_length"])
        df_pc = load_pc_data(CONFIG["split"], CONFIG["sample_fraction"], CONFIG["seed"])
        df_pc, res_pc = run_benchmark(model, tok, nl, df_pc, "PoisonedChalice", CONFIG["max_length"])
        del model, tok; gc.collect(); torch.cuda.empty_cache()

    print("\n" + "═"*60 + "\n  ALL DONE — MultiGeo-MIA\n" + "═"*60)
