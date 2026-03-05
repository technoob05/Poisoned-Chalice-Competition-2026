import numpy as np
from typing import List
from pathlib import Path
from tqdm import tqdm
import torch
from torch import log_softmax

from MIAttack import MIAttack


class MinKProbAttack(MIAttack):
    """
    Min-K% Prob Attack:
    Optimized to calculate scores for multiple k-ratios (0.05, 0.1, 0.2, etc.)
    in a single pass.
    """
    def __init__(self, args, model, tokenizer):
        super().__init__(args, model, tokenizer)
        self.sw = args.use_sliding_window

        self.k = 0.2
        if args.max_length != -1:
            self.window_size = args.max_length
            print("Using provided window size of {}".format(self.window_size))
        else:
            self.window_size = 256
            print(f"[Min-K] Using defaults: k={self.k}, win={self.window_size}")

    @property
    def name(self) -> str:
        return "mkp"

    def calculate_score_offline(self, sorted_probs: np.ndarray, k_ratio: float):
        """
        Calculates score for a single file, from ALREADY SORTED probabilities.
        """
        if len(sorted_probs) == 0:
            return np.nan

        k_length = max(1, int(len(sorted_probs) * k_ratio))
        top_k_min_log_probs = sorted_probs[:k_length]
        return np.mean(top_k_min_log_probs)

    @staticmethod
    def calculate_token_probabilities_truncation(text: str, model, tokenizer, max_length) -> tuple[np.ndarray,np.ndarray]:
        """Calculate per-token log probabilities for given text.
        Returns:
            1. values in (-inf,0) corresponding to the probabilities of each of the tokens.
            2. token ids
        """
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

        return np.array(token_log_probs), inputs["input_ids"][0, 1:].cpu().numpy()

    @staticmethod
    def calculate_token_probabilities_sliding(text: str, model, tokenizer, max_length) -> np.ndarray:
        """
        Calculates probabilities for the entire text using a Non-Overlapping Sliding Window.

        Why Non-Overlapping?
        1. It limits context, preventing the model from 'learning' the file style (Better AUC).
        2. It ensures every token is scored exactly once (No double counting).
        """
        # 1. Tokenize the ENTIRE text without truncation first
        # We use return_tensors='pt' but keep on CPU initially to slice easily
        encodings = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = encodings.input_ids[0]  # Flatten to 1D

        # Use the max_length argument as the Window Size
        window_size = max_length

        all_token_log_probs = []

        # 2. Loop through the tokens in chunks
        for i in range(0, len(input_ids), window_size):
            # Slice the chunk (e.g., 0-256, 256-512)
            chunk_ids = input_ids[i: i + window_size]

            # Skip tiny chunks at the end that are too small to predict anything
            if len(chunk_ids) < 2:
                continue

            # Move chunk to GPU
            chunk_tensor = chunk_ids.unsqueeze(0).to(model.device)  # Shape: [1, seq_len]

            with torch.no_grad():
                outputs = model(chunk_tensor, labels=chunk_tensor)
                logits = outputs.logits
                log_probs = log_softmax(logits, dim=-1)

                # Extract probabilities for the next tokens
                # We skip the first token (it has no history in this chunk)
                for j in range(chunk_ids.shape[0] - 1):
                    token_id = chunk_ids[j + 1]
                    # log_probs[batch, seq_index, vocab_index]
                    token_log_prob = log_probs[0, j, token_id].item()
                    all_token_log_probs.append(token_log_prob)
        return np.array(all_token_log_probs)

    def get_token_probs(self, texts: List[str]) -> List[np.ndarray]:
        """
        Public method to just get the sorted token probabilities (GPU intensive)
        for all code files at once.

        e.g. "for i in range(1,2)" -> "for":0,1 ; "i in" : 0,2 ...
        ...

        """
        all_probs = []
        for text in tqdm(texts, desc="Calculating token probabilities for MKP"):
            if self.sw:
                probs = self.calculate_token_probabilities_sliding(
                    text, self.model, self.tokenizer, self.window_size
                )
            else:
                probs, _ = self.calculate_token_probabilities_truncation(
                    text, self.model, self.tokenizer, self.window_size
                )
            all_probs.append(np.sort(probs))
        return all_probs

    def compute_scores(self, texts: List[str]) -> List[float]:
        """
        Main execution flow with a pre-defined k.
        """
        all_sorted_probs = self.get_token_probs(texts)
        scores = []
        for probs in all_sorted_probs:
            scores.append(self.calculate_score_offline(probs, self.k))
        return scores