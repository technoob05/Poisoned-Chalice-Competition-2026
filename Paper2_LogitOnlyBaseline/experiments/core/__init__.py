"""
core — shared infrastructure for Paper 2 (ESP-Cal) experiments.

Modules
-------
config       : Config dataclass + model registries
extractors   : ESPExtractor — all logit-level features in one forward pass
models       : load_model / free_model with disk-safety checks
calibration  : MultiScaleCalibrator (3-scale z-norm)
baselines    : BaselineComparison (Min-K%, Min-K%++, SURP, Zlib helpers)
data_loaders : load_poisoned_chalice / load_wikimia / load_mimir / load_bookmia
evaluation   : evaluate_scores / evaluate_per_subset (AUROC)
runner       : ESPCalExperiment high-level runner
"""

from .config import Config, WIKIMIA_MODELS, MIMIR_MODELS, BOOKMIA_MODELS
from .models import load_model, free_model
from .extractors import ESPExtractor
from .calibration import MultiScaleCalibrator
from .data_loaders import load_poisoned_chalice, load_wikimia, load_mimir, load_bookmia
from .evaluation import evaluate_scores, evaluate_per_subset
from .baselines import BaselineComparison
from .runner import ESPCalExperiment

__all__ = [
    "Config", "WIKIMIA_MODELS", "MIMIR_MODELS", "BOOKMIA_MODELS",
    "load_model", "free_model",
    "ESPExtractor", "MultiScaleCalibrator",
    "load_poisoned_chalice", "load_wikimia", "load_mimir", "load_bookmia",
    "evaluate_scores", "evaluate_per_subset",
    "BaselineComparison",
    "ESPCalExperiment",
]
