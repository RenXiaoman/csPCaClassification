"""Class-balanced sampling for the classification training sets."""
from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


class ClassBalancedSampler(WeightedRandomSampler):
    """Inverse-frequency sampler matching PMCNet's training strategy."""

    def __init__(self, labels, dataset_name: str | None = None):
        labels = [int(label) for label in labels]
        counts = Counter(labels)
        sample_weights = torch.tensor(
            [1.0 / max(counts[label], 1) for label in labels],
            dtype=torch.double,
        )
        # Keep one epoch at the original dataset size, with replacement.
        super().__init__(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        self.class_counts = dict(sorted(counts.items()))


# Backward-compatible name used by the training scripts.
ClassRepeatSampler = ClassBalancedSampler
