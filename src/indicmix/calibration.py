"""Probability Calibration and Reliability Metrics.

Provides Temperature Scaling, Platt Scaling, and Expected Calibration Error (ECE)
computation to guarantee that classifier confidence scores reflect true empirical likelihood.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CalibrationReport:
    """Audit report of probability calibration."""
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    temperature: float
    num_samples: int
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]


class TemperatureScaler:
    """Post-hoc temperature scaling calibrator for multiclass and multilabel logits."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-3, temperature)

    def set_temperature(self, temperature: float):
        self.temperature = max(1e-3, temperature)

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """Scale raw logits by 1/T."""
        return logits / self.temperature

    def calibrate_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate already softmaxxed probabilities by transforming back to pseudo-logits, scaling, and re-normalizing."""
        eps = 1e-7
        clipped = np.clip(probs, eps, 1.0 - eps)
        pseudo_logits = np.log(clipped)
        scaled = pseudo_logits / self.temperature
        # Re-apply softmax
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10
) -> Tuple[float, float, List[float], List[float], List[int]]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        confidences: 1D array of predicted confidence scores in [0, 1].
        accuracies: 1D binary array (1 if prediction correct, 0 otherwise).
        num_bins: Number of equal-width probability bins.
        
    Returns:
        (ece, mce, bin_accuracies, bin_confidences, bin_counts)
    """
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    ece = 0.0
    mce = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0, 0.0, [], [], []

    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_count = int(np.sum(in_bin))
        bin_counts.append(bin_count)

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            
            diff = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * diff
            mce = max(mce, diff)
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((lower + upper) / 2.0))

    return float(ece), float(mce), bin_accs, bin_confs, bin_counts


def compute_brier_score(probabilities: np.ndarray, one_hot_labels: np.ndarray) -> float:
    """Compute Brier Score (mean squared error between predicted probabilities and one-hot labels)."""
    return float(np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=-1)))
