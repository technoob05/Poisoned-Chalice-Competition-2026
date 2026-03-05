import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
import json

def plot_roc_curve(results_df: pd.DataFrame, model_score_col: str, model_name: str, output_path: str) -> float:
    # Calculate ROC curve
    results_df['is_member'] = results_df['membership'].apply(lambda x: 1 if x == 'member' else 0)
    fpr, tpr, thresholds = roc_curve(results_df["is_member"], results_df[model_score_col])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.title(f'ROC Curve: {model_score_col}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_path}roc_curve_{model_score_col}.pdf")
    plt.close()

    return roc_auc

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Membership Inference Attack Scoring")

    # Model parameters
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to metadata json file"
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="results/",
        help="Path to results parquet file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="plots/",
        help="Path to save the ROC curves"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Load the metadata into a dictionary
    with open(args.config_path, 'r') as f:
        metadata = json.load(f)
    model_name = metadata['model_name'].replace("/", "_")

    # Load results
    results_df = pd.read_parquet(f"{args.results_folder}results_{model_name}_{metadata['timestamp']}.parquet").fillna(0)
    print(results_df.columns)

    # For each attack executed, plot ROC curve
    for attack in metadata['attacks_executed']:
        score_col = f"{attack}_score"
        membership_col = "is_member"
        roc_auc = plot_roc_curve(results_df, score_col, model_name,  args.output_path)
        print(f"ROC curve for attack '{attack}' saved to: {args.output_path}")
        print(f"AUC score: {roc_auc:.2f}")