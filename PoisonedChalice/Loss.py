from MIAttack import MIAttack
from typing import List
import numpy as np
import torch
from tqdm import tqdm

class LossAttack(MIAttack):
    """Loss-based MIA: Lower loss indicates membership."""

    @property
    def name(self) -> str:
        return "loss"

    def compute_scores(self, texts: List[str]) -> List[float]:
        """Calculate negative loss scores (higher = member)."""
        scores = []

        for text in tqdm(texts, desc=f"Computing {self.name} scores"):
            try:
                inputs = self.tokenizer(
                    text,
                    max_length=self.args.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])

                # Negative loss: higher score indicates membership
                scores.append(-outputs.loss.item())
            except Exception as e:
                print(f"Error calculating loss: {e}")
                scores.append(np.nan)

        return scores