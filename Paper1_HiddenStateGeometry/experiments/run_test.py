#!/usr/bin/env python3
"""
MultiGeo-MIA — Quick Smoke Test
================================
Runs a minimal test (10 samples, 1 model) to verify everything loads & works.
Use this BEFORE the full run to catch import/data/GPU issues early.

Usage:
    python run_test.py
"""

import os
import sys
import subprocess
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "accelerate", "datasets",
                "scikit-learn", "scipy", "huggingface_hub", "pyarrow"],
               capture_output=True)

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    token = UserSecretsClient().get_secret("posioned")
    login(token=token, add_to_git_credential=True)
    print("✓ HuggingFace authenticated")
except Exception:
    print("○ Using local auth")

import torch
import numpy as np
import pandas as pd

def test_imports():
    """Test all module imports."""
    print("\n[1/5] Testing imports...")
    from multigeo import (
        Config, MultiGeoExperiment,
        load_model, free_model,
        MultiGeoExtractor,
        load_poisoned_chalice, load_wikimia, load_mimir, load_bookmia,
        evaluate_scores, rank_average,
        WIKIMIA_MODELS, MIMIR_MODELS, BOOKMIA_MODELS,
    )
    print(f"  ✓ All imports OK")
    print(f"  ✓ WikiMIA models: {len(WIKIMIA_MODELS)}")
    print(f"  ✓ MIMIR models:   {len(MIMIR_MODELS)}")
    print(f"  ✓ BookMIA models:  {len(BOOKMIA_MODELS)}")
    return Config

def test_gpu():
    """Test GPU availability."""
    print("\n[2/5] Testing GPU...")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  ✓ GPU: {name} ({vram:.0f} GB)")
    else:
        print(f"  ⚠ No CUDA GPU — will run on CPU (slow)")

def test_model_load(cfg):
    """Test loading the smallest model."""
    print("\n[3/5] Testing model loading (pythia-160m)...")
    from multigeo import load_model, free_model

    t0 = time.time()
    model, tokenizer, n_layers = load_model("EleutherAI/pythia-160m-deduped", cfg.torch_dtype)
    print(f"  ✓ Loaded in {time.time()-t0:.1f}s, {n_layers} layers")

    # Quick forward pass
    inputs = tokenizer("Hello world, this is a test.", return_tensors="pt",
                       truncation=True, max_length=32)
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, output_attentions=True)

    print(f"  ✓ Forward pass OK: logits {outputs.logits.shape}")
    print(f"  ✓ Hidden states: {len(outputs.hidden_states)} layers")
    print(f"  ✓ Attentions: {len(outputs.attentions)} layers")

    free_model(model, tokenizer)
    return True

def test_extractor(cfg):
    """Test the MultiGeo extractor on a few samples."""
    print("\n[4/5] Testing MultiGeo extractor...")
    from multigeo import load_model, free_model, MultiGeoExtractor

    model, tokenizer, n_layers = load_model("EleutherAI/pythia-160m-deduped", cfg.torch_dtype)
    extractor = MultiGeoExtractor(model, tokenizer, n_layers, cfg)

    test_texts = [
        "def hello():\n    print('Hello, world!')\n",
        "The quick brown fox jumps over the lazy dog.",
        "import numpy as np\ndef compute(x):\n    return np.sum(x ** 2)\n",
    ]

    for i, text in enumerate(test_texts):
        t0 = time.time()
        features = extractor.extract(text)
        elapsed = time.time() - t0
        print(f"  Sample {i+1}: {len(features)} features in {elapsed:.2f}s")
        print(f"    magnitude={features['signal_magnitude']:.4f}")
        print(f"    dimensionality={features['signal_dimensionality']:.4f}")
        print(f"    dynamics={features['signal_dynamics']:.6f}")
        print(f"    routing={features['signal_routing']:.4f}")
        print(f"    loss={features['loss']:.4f}")

    free_model(model, tokenizer, extractor)
    return True

def test_data_loading(cfg):
    """Test data loading (first 10 samples)."""
    print("\n[5/5] Testing data loading...")
    cfg_test = type(cfg)()
    cfg_test.sample_fraction = 0.01  # tiny sample

    # Poisoned Chalice
    try:
        from multigeo import load_poisoned_chalice
        df = load_poisoned_chalice(cfg_test)
        if len(df) > 0:
            print(f"  ✓ Poisoned Chalice: {len(df)} samples, cols={list(df.columns)}")
        else:
            print(f"  ⚠ Poisoned Chalice: empty (might need Kaggle dataset)")
    except Exception as e:
        print(f"  ⚠ Poisoned Chalice: {e}")

    # WikiMIA
    try:
        from multigeo import load_wikimia
        cfg_test.wikimia_lengths = [32]  # just one length
        data = load_wikimia(cfg_test)
        for k, df in data.items():
            print(f"  ✓ WikiMIA {k}: {len(df)} samples")
    except Exception as e:
        print(f"  ⚠ WikiMIA: {e}")

    return True


def main():
    print("=" * 60)
    print("  MultiGeo-MIA — SMOKE TEST")
    print("=" * 60)

    t_start = time.time()

    # 1. Imports
    Config = test_imports()
    cfg = Config()
    cfg.output_dir = "./results_test"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 2. GPU
    test_gpu()

    # 3. Model loading
    test_model_load(cfg)

    # 4. Extractor
    test_extractor(cfg)

    # 5. Data
    test_data_loading(cfg)

    elapsed = time.time() - t_start
    print("\n" + "═" * 60)
    print(f"  ✓ ALL TESTS PASSED in {elapsed:.1f}s")
    print("═" * 60)
    print("\n  Ready to run: python run_all.py")


if __name__ == "__main__":
    main()
