"""Model loading and GPU memory management."""
import gc
import os
import shutil
import torch


def _disk_free_gb() -> float:
    """Return free disk space in GB for the filesystem containing HF cache."""
    try:
        cache_dir = os.path.expanduser("~")
        stat = os.statvfs(cache_dir)
        return (stat.f_bavail * stat.f_frsize) / 1e9
    except Exception:
        try:
            import shutil as _sh
            total, used, free = _sh.disk_usage(os.path.expanduser("~"))
            return free / 1e9
        except Exception:
            return -1.0


# Estimated on-disk sizes (GB, bf16 safetensors) for large models
_MODEL_DISK_SIZES = {
    "facebook/opt-30b": 56.0,
    "EleutherAI/gpt-neox-20b": 39.0,
    "facebook/opt-13b": 24.0,
    "EleutherAI/pythia-12b-deduped": 23.0,
    "facebook/opt-6.7b": 12.5,
    "EleutherAI/pythia-6.9b-deduped": 13.0,
}
_DISK_SAFETY_MARGIN_GB = 5.0  # keep at least 5 GB free


def load_model(model_name: str, dtype_str: str = "bfloat16"):
    """Load a causal LM for logit extraction (no hidden states needed)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    free_gb = _disk_free_gb()
    print(f"\n  Loading model: {model_name}  [disk free: {free_gb:.1f} GB]")

    # Safety: skip if model is known-large and disk is insufficient
    need = _MODEL_DISK_SIZES.get(model_name, 0)
    if need > 0 and free_gb >= 0 and free_gb < need + _DISK_SAFETY_MARGIN_GB:
        raise RuntimeError(
            f"Skipping {model_name}: needs ~{need:.0f} GB but only {free_gb:.1f} GB free"
        )

    dtype = getattr(torch, dtype_str, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ✓ {model_name}: {model.config.num_hidden_layers}L, {n_params:.1f}B params, dtype={dtype}")
    return model, tokenizer


def free_model(*objects, model_name: str = None):
    """Free GPU memory and optionally clean HF cache for a model."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"  ♻ GPU memory freed. Current: {allocated:.1f} GB")

    # Clean HF cache to prevent disk overflow on Kaggle
    if model_name:
        _clean_hf_cache(model_name)
    free_gb = _disk_free_gb()
    if free_gb >= 0:
        print(f"  📀 Disk free after cleanup: {free_gb:.1f} GB")


def _clean_hf_cache(model_name: str):
    """Remove a specific model from HuggingFace cache to save disk."""
    try:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        if not os.path.exists(cache_dir):
            return
        # HF cache uses models--org--name format
        safe_name = "models--" + model_name.replace("/", "--")
        model_cache = os.path.join(cache_dir, safe_name)
        if os.path.exists(model_cache):
            size_gb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, fns in os.walk(model_cache)
                for f in fns
            ) / 1e9
            shutil.rmtree(model_cache, ignore_errors=True)
            print(f"  🗑 Cleaned cache: {model_name} ({size_gb:.1f} GB)")
    except Exception:
        pass  # Non-critical
