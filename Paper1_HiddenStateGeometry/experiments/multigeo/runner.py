"""Multi-model experiment runner for MultiGeo-MIA."""
import os
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, Any

from .config import Config
from .models import load_model, free_model
from .extractors import MultiGeoExtractor
from .data_loaders import (
    load_poisoned_chalice, load_wikimia, load_mimir, load_bookmia,
)
from .evaluation import (
    evaluate_scores, evaluate_per_subset, rank_average, per_language_znorm,
)


class MultiGeoExperiment:
    """Run MultiGeo-MIA across benchmarks and models."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)

    # ────────────────────────────────────────
    # Poisoned Chalice (single model: starcoder2-3b)
    # ────────────────────────────────────────
    def run_poisoned_chalice(self) -> Dict:
        print("\n" + "█" * 60)
        print("  BENCHMARK: Poisoned Chalice (Code MIA)")
        print("█" * 60)

        model, tokenizer, n_layers = load_model(self.cfg.model_name, self.cfg.torch_dtype)
        extractor = MultiGeoExtractor(model, tokenizer, n_layers, self.cfg)
        df = load_poisoned_chalice(self.cfg)
        result = self._extract_and_evaluate(df, extractor, "PoisonedChalice")
        free_model(model, tokenizer, extractor)
        return result

    # ────────────────────────────────────────
    # WikiMIA (multi-model)
    # ────────────────────────────────────────
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
                model, tokenizer, n_layers = load_model(model_name, self.cfg.torch_dtype)
                extractor = MultiGeoExtractor(model, tokenizer, n_layers, self.cfg)
                for lk, df in data_by_length.items():
                    tag = f"WikiMIA_{lk}_{short}"
                    print(f"\n  ── {tag} ──")
                    all_results[tag] = self._extract_and_evaluate(df.copy(), extractor, tag, znorm=False)
                free_model(model, tokenizer, extractor)
            except Exception as e:
                print(f"  ✗ {model_name}: {e}")
        return all_results

    # ────────────────────────────────────────
    # MIMIR (multi-model)
    # ────────────────────────────────────────
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
                model, tokenizer, n_layers = load_model(model_name, self.cfg.torch_dtype)
                extractor = MultiGeoExtractor(model, tokenizer, n_layers, self.cfg)
                for domain, df in data_by_domain.items():
                    tag = f"MIMIR_{domain}_{short}"
                    print(f"\n  ── {tag} ──")
                    all_results[tag] = self._extract_and_evaluate(df.copy(), extractor, tag, znorm=False)
                free_model(model, tokenizer, extractor)
            except Exception as e:
                print(f"  ✗ {model_name}: {e}")
        return all_results

    # ────────────────────────────────────────
    # BookMIA (multi-model)
    # ────────────────────────────────────────
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
                model, tokenizer, n_layers = load_model(model_name, self.cfg.torch_dtype)
                extractor = MultiGeoExtractor(model, tokenizer, n_layers, self.cfg)
                tag = f"BookMIA_{short}"
                all_results[tag] = self._extract_and_evaluate(df_base.copy(), extractor, tag, znorm=False)
                free_model(model, tokenizer, extractor)
            except Exception as e:
                print(f"  ✗ {model_name}: {e}")
        return all_results

    # ────────────────────────────────────────
    # Core extraction + evaluation
    # ────────────────────────────────────────
    def _extract_and_evaluate(self, df: pd.DataFrame, extractor: MultiGeoExtractor,
                               tag: str, znorm: bool = True) -> Dict:
        print(f"\n  Extracting features for {len(df)} samples...")
        t0 = time.time()

        features_list = []
        for idx, row in df.iterrows():
            if idx > 0 and idx % 500 == 0:
                elapsed = time.time() - t0
                rate = idx / elapsed
                eta = (len(df) - idx) / rate
                print(f"    [{idx}/{len(df)}] {rate:.1f} s/s, ETA {eta:.0f}s")
            text = row.get("text") or ""
            if not text or not text.strip():
                features_list.append(extractor._empty_features())
                continue
            features_list.append(extractor.extract(text))

        features_df = pd.DataFrame(features_list)
        df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        elapsed = time.time() - t0
        print(f"  ✓ Done in {elapsed:.1f}s ({len(df)/elapsed:.1f} s/s)")

        signal_cols = ["signal_magnitude", "signal_dimensionality",
                       "signal_dynamics", "signal_routing"]
        all_feature_cols = [c for c in features_df.columns if c not in ["seq_len"]]

        if znorm and self.cfg.per_language_znorm and "subset" in df.columns:
            df = per_language_znorm(df, all_feature_cols)

        df["multigeo_4axis"] = rank_average(df, signal_cols)
        df["multigeo_mag_dim"] = rank_average(df, ["signal_magnitude", "signal_dimensionality"])
        df["multigeo_dyn_rout"] = rank_average(df, ["signal_dynamics", "signal_routing"])

        combo_cols = ["multigeo_4axis", "multigeo_mag_dim", "multigeo_dyn_rout"]
        eval_cols = signal_cols + combo_cols + ["loss"]
        results_df = evaluate_scores(df, eval_cols)

        print("\n" + "─" * 50)
        print(f"  RESULTS: {tag}")
        print("─" * 50)
        if len(results_df) > 0:
            for _, r in results_df.iterrows():
                m = "★" if r["score"] == "multigeo_4axis" else " "
                print(f"  {m} {r['score']:30s}  AUC={r['auc']:.4f}  ({r['polarity']})")

        if "subset" in df.columns and df["subset"].nunique() > 1:
            print(f"\n  Per-subset (multigeo_4axis):")
            for _, sr in evaluate_per_subset(df, "multigeo_4axis").iterrows():
                print(f"    {sr['subset']:15s}  AUC={sr['auc']:.4f}  (n={sr['n']})")

        # Save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_parquet(os.path.join(self.cfg.output_dir, f"multigeo_{tag}_{ts}.parquet"), index=False)

        summary = {
            "benchmark": tag, "timestamp": ts, "n_samples": len(df),
            "results": results_df.to_dict(orient="records") if len(results_df) > 0 else [],
        }
        with open(os.path.join(self.cfg.output_dir, f"multigeo_{tag}_{ts}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return {"df": df, "results": results_df, "summary": summary}
