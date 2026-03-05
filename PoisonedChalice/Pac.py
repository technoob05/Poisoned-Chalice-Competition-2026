import argparse
import json
import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Type, Set, Tuple
from MIAttack import MIAttack

import numpy as np
import pandas as pd
import torch
from torch import log_softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_token_probabilities(text: str, model, tokenizer, max_length: int = 4096) -> np.ndarray:
    """Calculate per-token log probabilities for given text."""
    inputs = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        log_probs = log_softmax(logits, dim=-1)

        token_log_probs = []
        for i in range(inputs["input_ids"].shape[1] - 1):
            token_id = inputs["input_ids"][0, i + 1]
            token_log_prob = log_probs[0, i, token_id].item()
            token_log_probs.append(token_log_prob)

    return np.array(token_log_probs)

def calculate_token_probabilities_batch(texts: List[str], model, tokenizer, max_length: int = 4096) -> List[np.ndarray]:
    """Calculate per-token log probabilities for multiple texts at once."""
    inputs = tokenizer(texts, max_length=max_length, truncation=True,
                       return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        log_probs = log_softmax(logits, dim=-1)

        all_token_log_probs = []
        for batch_idx in range(inputs["input_ids"].shape[0]):
            token_log_probs = []
            seq_len = (inputs["attention_mask"][batch_idx] == 1).sum().item()

            for i in range(seq_len - 1):
                token_id = inputs["input_ids"][batch_idx, i + 1]
                token_log_prob = log_probs[batch_idx, i, token_id].detach().cpu().item()
                token_log_probs.append(token_log_prob)

            all_token_log_probs.append(np.array(token_log_probs))

    # Explicitly delete large tensors to free GPU memory
    del inputs, outputs, logits, log_probs
    torch.cuda.empty_cache()

    return all_token_log_probs

def calculate_token_probabilities_sequential(texts: List[str], model, tokenizer, max_length: int = 4096) -> List[np.ndarray]:
    """Calculate per-token log probabilities for multiple texts sequentially (one at a time)."""
    all_token_log_probs = []

    for text in texts:
        try:
            inputs = tokenizer(text, max_length=max_length, truncation=True,
                               return_tensors="pt", padding=False).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits
                log_probs = log_softmax(logits, dim=-1)

                token_log_probs = []
                seq_len = inputs["input_ids"].shape[1]

                for i in range(seq_len - 1):
                    token_id = inputs["input_ids"][0, i + 1]
                    token_log_prob = log_probs[0, i, token_id].detach().cpu().item()
                    token_log_probs.append(token_log_prob)

                all_token_log_probs.append(np.array(token_log_probs))

            # Clean up after each text
            del inputs, outputs, logits, log_probs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing text: {e}")
            all_token_log_probs.append(np.array([]))

    return all_token_log_probs

def compute_polarized_distance(list_of_probs: List[float], near_count: int, far_count: int) -> float:
    """Compute the polarized distance based on a sorted list of probabilities."""
    if len(list_of_probs) == 0:
        return 0.0

    list_length = len(list_of_probs)
    far_count = max(1, min(far_count, list_length))
    near_count = max(1, min(near_count, list_length))

    sorted_list_of_probs = np.sort(list_of_probs)

    if near_count + far_count > list_length:
        scale = list_length / (near_count + far_count)
        near_count = max(1, int(near_count * scale))
        far_count = max(1, int(far_count * scale))

    return (
            np.mean(sorted_list_of_probs[::-1][:far_count]) -
            np.mean(sorted_list_of_probs[:near_count])
    )

def generate_adjacent_samples_by_token_split(
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        m_ratio: float = 0.3,
        n_samples: int = 5
) -> List[str]:
    """Generate adjacent samples by randomly swapping tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    adjacent_samples = []

    for _ in range(n_samples):
        swapped_tokens = tokens.copy()
        for _ in range(int(m_ratio * len(swapped_tokens))):
            if len(swapped_tokens) >= 2:
                idx1, idx2 = random.sample(list(range(len(swapped_tokens))), 2)
                swapped_tokens[idx1], swapped_tokens[idx2] = swapped_tokens[idx2], swapped_tokens[idx1]

        adjacent_text = tokenizer.decode(swapped_tokens, skip_special_tokens=True)
        adjacent_samples.append(adjacent_text)

    return adjacent_samples

def polarized_distance(
        original_probs: List[float],
        adjacent_probs_list: List[List[float]],
        near_count: int = 30,
        far_count: int = 5
) -> float:
    """Calculate polarized distance between original and adjacent samples."""
    adjacent_polarized_distances = [
        compute_polarized_distance(adj_probs, near_count, far_count)
        for adj_probs in adjacent_probs_list
    ]

    original_polarized_distance = compute_polarized_distance(original_probs, near_count, far_count)

    if len(adjacent_polarized_distances) > 0:
        return float(original_polarized_distance) - float(np.mean(adjacent_polarized_distances))
    else:
        return 0.0

def get_pac_score(
        text: str,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        near_count: int = 30,
        far_count: int = 5,
        max_length: int = 4096,
        m_ratio: float = 0.3,
        n_samples: int = 5
) -> float:
    """Calculate PAC score for a given text."""
    mutated_samples = generate_adjacent_samples_by_token_split(text, tokenizer, m_ratio, n_samples)

    all_texts = [text] + mutated_samples
    all_probs = calculate_token_probabilities_sequential(all_texts, model, tokenizer, max_length)

    original_probs = all_probs[0]
    mutated_probs = all_probs[1:]

    return polarized_distance(original_probs.tolist(), mutated_probs, near_count, far_count)

def load_model_from_directory(model_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            legacy=False,
            use_fast=True,
            trust_remote_code=True,
        )
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            legacy=False,
            use_fast=False,
            trust_remote_code=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        return_dict=True,
        trust_remote_code=True,
        device_map="cuda:0" # <- Uncomment this if you have a cuda-supporting gpu
    )
    model.eval()
    return model, tokenizer

# ============================================================================
# Attack Logic
# ============================================================================
class PACAttack(MIAttack):
    """PAC-based MIA: Polaired-Augment Calibration"""

    @property
    def name(self) -> str:
        return "pac"

    def compute_scores(self, texts: List[str]) -> List[float]:
        """Calculate negative loss scores (higher = member)."""
        print("\nCalculating PAC scores...")
        pac_scores = []

        for text in tqdm(texts, desc="PAC calculation"):
            try:
                pac_score = get_pac_score(
                    text=text,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    near_count=self.args.pac_near_count,
                    far_count=self.args.pac_far_count,
                    max_length=self.args.max_length,
                    m_ratio=self.args.pac_m_ratio,
                    n_samples=self.args.pac_n_samples
                )
                pac_scores.append(pac_score)
            except Exception as e:
                print(f"Error calculating PAC: {e}")
                pac_scores.append(np.nan)

        return pac_scores