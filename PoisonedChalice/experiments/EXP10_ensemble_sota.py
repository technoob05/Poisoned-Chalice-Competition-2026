
"""
EXPERIMENT 10: Multi-Objective SOTA Ensemble
Method: Aggregate scores from all experiments (EXP01 - EXP09).
Goal: Maximize AUC by combining diverse signals (Likelihood, Neighborhood, SIA-TTS, Graph).
Methodology: Rank Averaging or Weighted Sum.
Usage: Run this after all other EXP files have produced .parquet results in results/ directory.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from typing import List

def setup_environment():
    # Minimal setup for aggregation
    pass

def load_results(results_dir: str) -> List[pd.DataFrame]:
    path = Path(results_dir)
    files = list(path.glob("*.parquet"))
    dfs = []
    for f in files:
        print(f"Loading {f.name}...")
        dfs.append(pd.read_parquet(f))
    return dfs

def ensemble_scoring(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs: return pd.DataFrame()
    
    # Use the first DF as base (assuming same sampling/indices or we join on 'content')
    # Better to join on 'content' to be safe across different sample fractions
    base_df = dfs[0][['content', 'membership', 'is_member']].copy()
    
    score_cols = []
    for i, df in enumerate(dfs):
        # Identify the score column (usually ends in _score)
        sc = [c for c in df.columns if c.endswith("_score")]
        if not sc: continue
        sc = sc[0]
        
        # Merge
        temp = df[['content', sc]]
        # Drop duplicates if any in content
        temp = temp.drop_duplicates(subset='content')
        
        base_df = base_df.merge(temp, on='content', how='left')
        score_cols.append(sc)
        
    # --- Weighted Ensemble ---
    # Heuristic weights: SIA-TTS and MinK++ usually perform better
    weights = {}
    for col in score_cols:
        if 'sia' in col or 'funnel' in col: weights[col] = 2.0
        elif 'mink' in col: weights[col] = 1.5
        elif 'canonical' in col: weights[col] = 1.2
        else: weights[col] = 1.0
        
    print("\nEnsembling with weights:", weights)
    
    # Normalize scores before averaging (Rank standardizing)
    for col in score_cols:
        # Fill NaNs with median/mean
        base_df[col] = base_df[col].fillna(base_df[col].mean())
        # Rank-based normalization (robust to scale differences)
        base_df[f"{col}_rank"] = base_df[col].rank(pct=True)
        
    rank_cols = [f"{col}_rank" for col in score_cols]
    
    # Weighted Rank Average
    base_df['final_ensemble_score'] = 0.0
    total_w = 0.0
    for col in score_cols:
        w = weights.get(col, 1.0)
        base_df['final_ensemble_score'] += base_df[f"{col}_rank"] * w
        total_w += w
        
    base_df['final_ensemble_score'] /= total_w
    
    return base_df

def run_evaluation(df: pd.DataFrame):
    print("\n--- Evaluation Results ---")
    score_cols = [c for c in df.columns if c.endswith("_score") or c == 'final_ensemble_score']
    
    results = []
    for col in score_cols:
        try:
            auc = roc_auc_score(df['is_member'], df[col].fillna(0))
            print(f"AUC ({col}): {auc:.4f}")
            results.append({'Method': col, 'AUC': auc})
        except:
            pass
            
    # Save tracker
    res_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
    res_df.to_csv("results/performance_summary.csv", index=False)
    print("\nSaved summary to results/performance_summary.csv")

if __name__ == "__main__":
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} not found. Run experiments first.")
    else:
        dfs = load_results(results_dir)
        if dfs:
            ensemble_df = ensemble_scoring(dfs)
            if not ensemble_df.empty:
                run_evaluation(ensemble_df)
                ensemble_df.to_parquet("results/FINAL_SUBMISSION_SCORES.parquet", index=False)
                print("Final combined scores saved to results/FINAL_SUBMISSION_SCORES.parquet")
        else:
            print("No result files found in results/")
