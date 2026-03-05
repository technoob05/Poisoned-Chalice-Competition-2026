
"""
EXPERIMENT 15: XGBoost Meta-Stacker (White-Box Master Ensemble)
Method: Feature-level stacking from all previous experiments.
        1. Load .parquet files from EXP01-EXP14.
        2. Clean and align features.
        3. Train XGBoost Classifier on a 80/20 train/val split.
        4. Predict Membership Probabilities.
Goal: Maximize AUC by combining diverse signals (Likelihood, Structure, Gradients, Entropy).
Usage: Copy-paste this entire file into a Kaggle cell.
"""
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
try:
    import xgboost as xgb
except ImportError:
    print("Installing xgboost...")
    os.system("pip install -q xgboost")
    import xgboost as xgb

# ============================================================================
# Data Loading & Feature Engineering
# ============================================================================

def load_all_results(results_dir="results"):
    print(f"Loading results from {results_dir}...")
    # Look for parquet files from EXP01 to EXP16
    files = glob.glob(os.path.join(results_dir, "*.parquet"))
    # Filtering for relevant experiments
    exp_files = [f for f in files if any(f"EXP{i:02d}" in f for i in range(1, 17))]
    if not exp_files:
        print("No relevant parquet files found! Run experiments first.")
        return None
    
    # We'll use the first file as the base (ID alignment)
    # Most files share the same index if sampled with the same seed.
    # To be safe, we should merge on unique content or original index.
    # For this starter, we assume they all contain a 'content' or 'id' column.
    
    dfs = []
    for f in all_files:
        try:
            temp_df = pd.read_parquet(f)
            # Identify score columns (usually end with _score or are specific metric names)
            score_cols = [c for c in temp_df.columns if '_score' in c or c in [
               'embedding', 'layer_0', 'layer_6', 'layer_12', 'layer_18', 'layer_23', 'head',
               'final_loss', 'avg_attn_entropy', 'mid_layer_loss', 'loss_gap'
            ]]
            
            # Keep only ID/Content and Score columns to avoid duplicate metadata
            # We'll merge on 'content'
            keep_cols = ['content', 'is_member'] + score_cols
            dfs.append(temp_df[keep_cols])
            print(f"Loaded {len(temp_df)} rows from {os.path.basename(f)}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not dfs: return None
    
    # Merge all
    main_df = dfs[0]
    for i in range(1, len(dfs)):
        # Merge on content to ensure alignment
        main_df = pd.merge(main_df, dfs[i], on=['content', 'is_member'], how='inner')
    
    print(f"Final Merged Dataframe Shape: {main_df.shape}")
    return main_df

# ============================================================================
# Meta-Classifier Training
# ============================================================================

def run_meta_stacking():
    df = load_all_results()
    if df is None: return
    
    # 1. Feature Preparation
    target = 'is_member'
    features = [c for c in df.columns if c not in ['content', target, 'membership', 'id']]
    
    X = df[features]
    y = df[target]
    
    print(f"Features: {features}")
    
    # 2. Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Train XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    print("\nTraining XGBoost Meta-Classifier...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    
    # 4. Evaluation
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"\n--- Validation Performance ---")
    print(f"XGBoost Meta-AUC: {val_auc:.4f}")
    # Evaluation
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"\n[Meta-Ensemble] Validation AUC: {auc:.4f}")
    
    # 5. Feature Importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    print("\nTop Features:")
    print(importance.head(10))
    
    # Save Final Predictions
    os.makedirs("results", exist_ok=True)
    df['meta_ensemble_score'] = model.predict_proba(X)[:, 1]
    df.to_parquet("results/EXP15_Final_Ensemble.parquet", index=False)
    print(f"\nSaved final predictions to results/EXP15_Final_Ensemble.parquet")

if __name__ == "__main__":
    run_meta_stacking()
