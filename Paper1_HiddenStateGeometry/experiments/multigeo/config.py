"""Configuration for MultiGeo-MIA experiments."""
import os
from dataclasses import dataclass, field
from typing import List

# ═══════════════════════════════════════════════════════════════
# Model Registries — H100 80 GB can handle up to ~40B (bf16)
# ═══════════════════════════════════════════════════════════════

WIKIMIA_MODELS = [
    # ── Pythia family (EleutherAI, trained on Pile — Wikipedia included) ──
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    # ── GPT-Neo / NeoX family (EleutherAI, trained on Pile) ──
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neox-20b",
    # ── OPT family (Meta, broad web data) ──
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # ── Mamba SSM family (state-space, different architecture) ──
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-2.8b-hf",
]

MIMIR_MODELS = [
    # Pythia suite (trained on Pile — MIMIR's source)
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    # GPT-Neo (also Pile-trained)
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neox-20b",
]

BOOKMIA_MODELS = [
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neox-20b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
]


@dataclass
class Config:
    """Experiment configuration."""
    # ── Primary model (Poisoned Chalice target) ──
    model_name: str = "bigcode/starcoder2-3b"
    max_length: int = 512
    torch_dtype: str = "bfloat16"

    # ── Poisoned Chalice dataset ──
    dataset_name: str = "AISE-TUDelft/Poisoned-Chalice"
    languages: List[str] = field(default_factory=lambda: ["Go", "Java", "Python", "Ruby", "Rust"])
    sample_fraction: float = 0.1          # 10% for quick test runs
    split: str = "test"

    # ── MultiGeo axes ──
    magnitude_layers: str = "mid"
    svd_top_k: int = 50
    cascade_pairs: int = 5

    # ── Evaluation ──
    seed: int = 42
    output_dir: str = "./results"
    per_language_znorm: bool = True

    # ── Benchmark-specific settings ──
    wikimia_lengths: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    mimir_domains: List[str] = field(default_factory=lambda: [
        "wikipedia", "github", "pile_cc", "pubmed_central",
        "arxiv", "dm_mathematics", "hackernews"
    ])

    # ── Multi-model control ──
    multi_model: bool = True
    wikimia_models: List[str] = field(default_factory=lambda: WIKIMIA_MODELS.copy())
    mimir_models: List[str] = field(default_factory=lambda: MIMIR_MODELS.copy())
    bookmia_models: List[str] = field(default_factory=lambda: BOOKMIA_MODELS.copy())

    # ── Kaggle data paths ──
    kaggle_dataset_root: str = "/kaggle/input/datasets/minh2duy/poisoned-chalice-dataset"

    def get_data_path(self, benchmark: str) -> str:
        """Return the resolved data path for a benchmark on Kaggle."""
        paths = {
            "poisoned_chalice": os.path.join(self.kaggle_dataset_root, "poisoned_chalice_dataset"),
            "wikimia": os.path.join(self.kaggle_dataset_root, "kaggle_wikimia"),
            "mimir": os.path.join(self.kaggle_dataset_root, "kaggle_mimir"),
            "bookmia": os.path.join(self.kaggle_dataset_root, "kaggle_bookmia"),
            "xsum_mia": os.path.join(self.kaggle_dataset_root, "kaggle_xsum_mia"),
            "agnews_mia": os.path.join(self.kaggle_dataset_root, "kaggle_agnews_mia"),
        }
        return paths.get(benchmark, "")
