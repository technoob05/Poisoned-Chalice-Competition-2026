# Standard library
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Type

# Third-party
import numpy as np
import pandas as pd
import torch

# Project imports
from MIAttack import MIAttack
from Loss import LossAttack
from Pac import PACAttack, load_model_from_directory
from MinKProbAttack import MinKProbAttack
from MinKProbAttack import MinKProbAttack
from datasets import load_dataset, load_from_disk
import os

# Attack Registry
ATTACK_REGISTRY: Dict[str, Type[MIAttack]] = {
    "loss": LossAttack,
    "pac": PACAttack,
    "mkp": MinKProbAttack,
}

# ============================================================================
# Main Experiment Class
# ============================================================================

class MIAExperiment:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Initialize model and tokenizer
        print(f"Loading model: {args.model_name}")
        self.model, self.tokenizer = load_model_from_directory("" + args.model_name)

    def load_datasets(self) -> pd.DataFrame:
        """Load member and non-member datasets."""
        subsets= ['Go', 'Java', 'Python', 'Ruby', 'Rust']
        dfs = []
        
        # Check if dataset is a local path
        is_local = os.path.exists(self.args.dataset)
        if is_local:
            print(f"Loading dataset from local path: {self.args.dataset}")
        else:
            print(f"Loading dataset from Hugging Face: {self.args.dataset}")

        for subset in subsets:
            if is_local:
                # Load from disk: expects structure like dataset_path/subset
                subset_path = os.path.join(self.args.dataset, subset)
                if not os.path.exists(subset_path):
                     raise FileNotFoundError(f"Subset {subset} not found at {subset_path}")
                ds = load_from_disk(subset_path)
                # If saved with save_to_disk, it might not have 'test' split if saved directly
                # Checking structure from download_data.py: ds.save_to_disk(save_path)
                # It saves the dataset/split directly.
                if hasattr(ds, "keys") and "test" in ds.keys():
                     ds = ds["test"]
                # If it's already the dataset (Arrow), just use it
                
            else:
                ds = load_dataset(self.args.dataset, subset, split="test")
                
            dfs.append(ds.to_pandas())
            
        # create is_member column value based on membership column (member vs non-member)
        ds = pd.concat(dfs, ignore_index=True)
        ds['is_member'] = ds['membership'].apply(lambda x: 1 if x == 'member' else 0)
        ds = ds.sample(frac=self.args.sample_fraction, random_state=self.args.seed)
        return ds

    def save_results(self, df: pd.DataFrame, executed_attacks: List[str]):
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a unique identifier for this experiment
        exp_id = f"{self.args.model_name.replace('/', '_')}_{timestamp}"

        # Save full results
        output_file = self.output_dir / f"results_{exp_id}.parquet"

        # Only keep essential columns to save space
        columns_to_keep = [
            'content', 'membership', 'is_member'
        ]

        # Add attack score columns
        for attack_name in executed_attacks:
            score_col = f"{attack_name}_score"
            if score_col in df.columns:
                columns_to_keep.append(score_col)
        df = df[columns_to_keep]

        df.to_parquet(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_name': self.args.model_name,
            'sample_fraction': self.args.sample_fraction,
            'seed': self.args.seed,
            'max_length': self.args.max_length,
            'attacks_executed': executed_attacks,
            'total_samples': len(df),
        }

        metadata_file = self.output_dir / f"metadata_{exp_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_file}")

    def run(self):
        """Run the complete experiment."""
        print(f"\n{'=' * 60}")
        print("Starting MIA Experiment")
        print(f"{'=' * 60}")
        print(f"Model: {self.args.model_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Attacks: {', '.join(self.args.attacks)}")
        print(f"{'=' * 60}\n")

        # Load data
        df = self.load_datasets()

        # Execute attacks
        executed_attacks = []
        for attack_key in self.args.attacks:
            if attack_key not in ATTACK_REGISTRY:
                print(f"Warning: Attack '{attack_key}' not found in registry. Skipping.")
                continue

            attack_class = ATTACK_REGISTRY[attack_key]
            attacker = attack_class(self.args, self.model, self.tokenizer)

            print(f"\nRunning attack: {attacker.name}")
            scores = attacker.compute_scores(df['content'].tolist())

            df[f"{attacker.name}_score"] = scores
            executed_attacks.append(attacker.name)

        # Save results
        self.save_results(df, executed_attacks)

        print(f"\n{'=' * 60}")
        print("Experiment completed successfully!")
        print(f"{'=' * 60}\n")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Membership Inference Attack experiments on code datasets"
    )

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., 'JetBrains/Mellum-4b-base')"
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=False,
        help="Use FP16 precision for faster inference"
    )

    # MKP-specific/utility flags
    parser.add_argument(
        "--use_sliding_window",
        action="store_true",
        default=False,
        help="Use sliding-window probability calculation in Min-K Prob attack"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="AISE-TUDelft/Poisoned-Chalice",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1,
        help="Fraction of data to sample (0.0-1.0)"
    )

    # Attack parameters
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        choices=["loss", "pac", "mkp"],
        default=["loss"],
        help="Which attacks to run"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )

    # Other parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # PAC-specific parameters
    parser.add_argument(
        "--pac_near_count",
        type=int,
        default=30,
        help="PAC near count parameter"
    )
    parser.add_argument(
        "--pac_far_count",
        type=int,
        default=5,
        help="PAC far count parameter"
    )
    parser.add_argument(
        "--pac_m_ratio",
        type=float,
        default=0.3,
        help="PAC mutation ratio"
    )
    parser.add_argument(
        "--pac_n_samples",
        type=int,
        default=5,
        help="Number of PAC adjacent samples"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    experiment = MIAExperiment(args)
    experiment.run()