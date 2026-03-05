"""Multi-model experiment runner for ESP-Cal."""
import os
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict

from .config import Config
from .models import load_model, free_model
from .extractors import ESPExtractor
from .calibration import MultiScaleCalibrator
from .data_loaders import load_poisoned_chalice, load_wikimia, load_mimir, load_bookmia
from .evaluation import evaluate_scores, evaluate_per_subset
from sklearn.metrics import roc_auc_score


class ESPCalExperiment:
    """Run ESP-Cal across benchmarks and models."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Poisoned Chalice (single model) ──
    def run_poisoned_chalice(self) -> Dict:
        print("\n" + "█" * 60)
        print("  BENCHMARK: Poisoned Chalice (Code MIA)")
        print("█" * 60)

        model, tokenizer = load_model(self.cfg.model_name, self.cfg.torch_dtype)
        extractor = ESPExtractor(model, tokenizer, self.cfg)
        df = load_poisoned_chalice(self.cfg)
        result = self._extract_and_evaluate(df, extractor, "PoisonedChalice", do_ablation=True)
        free_model(model, tokenizer, extractor)
        return result

    # ── WikiMIA (multi-model) ──
    def run_wikimia(self) -> Dict:
        print("\n" + "█" * 60)
        print("  BENCHMARK: WikiMIA (multi-model)")
        print("█" * 60)

        models = self.cfg.wikimia_models if self.cfg.multi_model else [self.cfg.wikimia_models[0]]
        data_by_length = load_wikimia(self.cfg)
        all_results = {}

        for model_name in models:
            short = model_name.split("/")[-1]
            print(f"\n  ╔══ Model: {model_name} ══╗")
            try:
                model, tokenizer = load_model(model_name, self.cfg.torch_dtype)
                extractor = ESPExtractor(model, tokenizer, self.cfg)
                for lk, df in data_by_length.items():
                    tag = f"WikiMIA_{lk}_{short}"
                    print(f"\n  ── {tag} ──")
                    all_results[tag] = self._extract_and_evaluate(
                        df.copy(), extractor, tag, calibrate=False
                    )
                free_model(model, tokenizer, extractor)
            except Exception as e:
                print(f"  ✗ {model_name}: {e}")
        return all_results

    # ── MIMIR (multi-model) ──
    def run_mimir(self) -> Dict:
        print("\n" + "█" * 60)
        print("  BENCHMARK: MIMIR (multi-model)")
        print("█" * 60)

        models = self.cfg.mimir_models if self.cfg.multi_model else [self.cfg.mimir_models[0]]
        data_by_domain = load_mimir(self.cfg)
        all_results = {}

        for model_name in models:
            short = model_name.split("/")[-1]
            print(f"\n  ╔══ Model: {model_name} ══╗")
            try:
                model, tokenizer = load_model(model_name, self.cfg.torch_dtype)
                extractor = ESPExtractor(model, tokenizer, self.cfg)
                for domain, df in data_by_domain.items():
                    tag = f"MIMIR_{domain}_{short}"
                    print(f"\n  ── {tag} ──")
                    all_results[tag] = self._extract_and_evaluate(
                        df.copy(), extractor, tag, calibrate=False
                    )
                free_model(model, tokenizer, extractor)
            except Exception as e:
                print(f"  ✗ {model_name}: {e}")
        return all_results

    # ── BookMIA (multi-model) ──
    def run_bookmia(self) -> Dict:
        print("\n" + "█" * 60)
        print("  BENCHMARK: BookMIA (multi-model)")
        print("█" * 60)

        models = self.cfg.bookmia_models if self.cfg.multi_model else [self.cfg.bookmia_models[0]]
        df_base = load_bookmia(self.cfg)
        if len(df_base) == 0:
            return {}
        all_results = {}

        for model_name in models:
            short = model_name.split("/")[-1]
            print(f"\n  ╔══ Model: {model_name} ══╗")
            try:
                model, tokenizer = load_model(model_name, self.cfg.torch_dtype)
                extractor = ESPExtractor(model, tokenizer, self.cfg)
                tag = f"BookMIA_{short}"
                all_results[tag] = self._extract_and_evaluate(
                    df_base.copy(), extractor, tag, calibrate=False
                )
                free_model(model, tokenizer, extractor)
            except Exception as e:
                print(f"  ✗ {model_name}: {e}")
        return all_results

    # ── Core extraction + evaluation ──
    def _extract_and_evaluate(self, df: pd.DataFrame, extractor: ESPExtractor,
                               tag: str, do_ablation: bool = False,
                               calibrate: bool = True) -> Dict:
        print(f"\n  Extracting features for {len(df)} samples...")
        t0 = time.time()

        features_list = []
        for idx, row in df.iterrows():
            if idx > 0 and idx % 500 == 0:
                elapsed = time.time() - t0
                rate = idx / elapsed
                print(f"    [{idx}/{len(df)}] {rate:.1f} s/s, ETA {(len(df)-idx)/rate:.0f}s")
            features_list.append(extractor.extract(row["text"]))

        features_df = pd.DataFrame(features_list)
        df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        elapsed = time.time() - t0
        print(f"  ✓ Done in {elapsed:.1f}s ({len(df)/elapsed:.1f} s/s)")

        calibration_cols = [c for c in ["signal_esp", "signal_h_drop", "signal_loss",
                            "neg_mean_loss", "minkprob_20", "surp"] if c in df.columns]

        if calibrate:
            calibrator = MultiScaleCalibrator(self.cfg)
            df = calibrator.calibrate(df, calibration_cols)

        all_score_cols = [c for c in df.columns if c not in
                          ["text", "is_member", "subset", "seq_len", "n_tokens",
                           "_len_bucket"] and not c.endswith("_raw")]

        results_df = evaluate_scores(df, all_score_cols)

        print("\n" + "─" * 50)
        print(f"  RESULTS: {tag}")
        print("─" * 50)
        if len(results_df) > 0:
            for _, r in results_df.head(15).iterrows():
                m = "★" if "esp" in r["score"].lower() else " "
                print(f"  {m} {r['score']:30s}  AUC={r['auc']:.4f}  ({r['polarity']})")

        best_col = "signal_esp" if "signal_esp" in results_df["score"].values else results_df.iloc[0]["score"] if len(results_df) > 0 else None
        if best_col and "subset" in df.columns and df["subset"].nunique() > 1:
            print(f"\n  Per-subset ({best_col}):")
            for _, sr in evaluate_per_subset(df, best_col).iterrows():
                print(f"    {sr['subset']:15s}  AUC={sr['auc']:.4f}  (n={sr['n']})")

        # Ablation
        ablation_df = None
        if do_ablation:
            try:
                ablation_df = self._run_ablation(df)
            except Exception as e:
                print(f"  Ablation error: {e}")

        # Save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(os.path.join(self.cfg.output_dir, f"espcal_{tag}_{ts}.parquet"), index=False)

        summary = {
            "benchmark": tag, "timestamp": ts, "n_samples": len(df),
            "results": results_df.to_dict(orient="records") if len(results_df) > 0 else [],
            "ablation": ablation_df.to_dict(orient="records") if ablation_df is not None else None,
        }
        with open(os.path.join(self.cfg.output_dir, f"espcal_{tag}_{ts}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return {"df": df, "results": results_df, "summary": summary}

    def _run_ablation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ablation: measure contribution of each calibration scale."""
        print("\n  ABLATION: Calibration Scale Contributions")
        results = []

        # (a) Raw ESP
        if "esp_slope_raw" in df.columns:
            v = df["esp_slope_raw"].notna()
            auc = roc_auc_score(df.loc[v, "is_member"], df.loc[v, "esp_slope_raw"])
            results.append({"condition": "(a) ESP raw", "auc": max(auc, 1-auc)})

        # (b) Token z-norm
        v = df["z_esp_slope"].notna()
        auc = roc_auc_score(df.loc[v, "is_member"], df.loc[v, "z_esp_slope"])
        results.append({"condition": "(b) + Scale 1 (token)", "auc": max(auc, 1-auc)})

        # (c)+(d) Full calibration
        v = df["signal_esp"].notna()
        auc = roc_auc_score(df.loc[v, "is_member"], df.loc[v, "signal_esp"])
        results.append({"condition": "(d) Full ESP-Cal", "auc": max(auc, 1-auc)})

        abl = pd.DataFrame(results)
        for _, r in abl.iterrows():
            m = "★" if "Full" in r["condition"] else " "
            print(f"  {m} {r['condition']:40s}  AUC={r['auc']:.4f}")
        return abl
