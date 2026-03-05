#!/usr/bin/env python3
"""
ESP-Cal — Quick Smoke Test
===========================
Runs a minimal test (3 samples, smallest model) to verify everything works.

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


def test_imports():
    print("\n[1/5] Testing imports...")
    from espcal import (
        Config, ESPCalExperiment,
        load_model, free_model,
        ESPExtractor, MultiScaleCalibrator,
        load_poisoned_chalice, load_wikimia, load_mimir, load_bookmia,
        evaluate_scores,
        BaselineComparison,
        WIKIMIA_MODELS, MIMIR_MODELS, BOOKMIA_MODELS,
    )
    print(f"  ✓ All imports OK")
    print(f"  ✓ WikiMIA models: {len(WIKIMIA_MODELS)}")
    print(f"  ✓ MIMIR models:   {len(MIMIR_MODELS)}")
    print(f"  ✓ BookMIA models:  {len(BOOKMIA_MODELS)}")
    return Config


def test_gpu():
    print("\n[2/5] Testing GPU...")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  ✓ GPU: {name} ({vram:.0f} GB)")
    else:
        print(f"  ⚠ No CUDA GPU — will run on CPU (slow)")


def test_model_load(cfg):
    print("\n[3/5] Testing model loading (pythia-160m)...")
    from espcal import load_model, free_model

    t0 = time.time()
    model, tokenizer = load_model("EleutherAI/pythia-160m-deduped", cfg.torch_dtype)
    print(f"  ✓ Loaded in {time.time()-t0:.1f}s")

    inputs = tokenizer("Hello world, this is a test.", return_tensors="pt",
                       truncation=True, max_length=32)
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    print(f"  ✓ Forward pass OK: logits {outputs.logits.shape}")

    free_model(model, tokenizer)
    return True


def test_extractor(cfg):
    print("\n[4/5] Testing ESP extractor...")
    from espcal import load_model, free_model, ESPExtractor

    model, tokenizer = load_model("EleutherAI/pythia-160m-deduped", cfg.torch_dtype)
    extractor = ESPExtractor(model, tokenizer, cfg)

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
        print(f"    esp_slope={features['esp_slope']:.6f}")
        print(f"    signal_esp={features['signal_esp']:.6f}")
        print(f"    signal_loss={features['signal_loss']:.4f}")
        print(f"    h_drop={features['h_drop']:.4f}")
        print(f"    minkprob_20={features['minkprob_20']:.4f}")

    free_model(model, tokenizer, extractor)
    return True


def test_data_loading(cfg):
    print("\n[5/5] Testing data loading...")

    try:
        from espcal import load_poisoned_chalice
        cfg_test = type(cfg)()
        cfg_test.sample_fraction = 0.01
        df = load_poisoned_chalice(cfg_test)
        if len(df) > 0:
            print(f"  ✓ Poisoned Chalice: {len(df)} samples")
        else:
            print(f"  ⚠ Poisoned Chalice: empty (might need Kaggle dataset)")
    except Exception as e:
        print(f"  ⚠ Poisoned Chalice: {e}")

    try:
        from espcal import load_wikimia
        cfg_test = type(cfg)()
        cfg_test.wikimia_lengths = [32]
        data = load_wikimia(cfg_test)
        for k, df in data.items():
            print(f"  ✓ WikiMIA {k}: {len(df)} samples")
    except Exception as e:
        print(f"  ⚠ WikiMIA: {e}")

    return True


def main():
    print("=" * 60)
    print("  ESP-Cal — SMOKE TEST")
    print("=" * 60)

    t_start = time.time()

    Config = test_imports()
    cfg = Config()
    cfg.output_dir = "./results_test"
    os.makedirs(cfg.output_dir, exist_ok=True)

    test_gpu()
    test_model_load(cfg)
    test_extractor(cfg)
    test_data_loading(cfg)

    print("\n" + "═" * 60)
    print(f"  ✓ ALL TESTS PASSED in {time.time()-t_start:.1f}s")
    print("═" * 60)
    print("\n  Ready to run: python run_all.py")


if __name__ == "__main__":
    main()
