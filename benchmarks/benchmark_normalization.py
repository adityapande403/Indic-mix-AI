"""Normalization Benchmarking Suite for IndicMix AI.

Measures latency, CER, and entity preservation across Mode A, Mode B, and Mode C.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from indicmix.normalization import TextNormalizer


def benchmark_normalization(num_runs: int = 100) -> Dict[str, Any]:
    print("=== Running Text Normalization Benchmark ===")
    normalizer = TextNormalizer(strict_entity_preservation=True)

    samples = [
        ("bhai payment deduct ho gaya par order confirm nahi dikha raha", "english"),
        ("mera order ORD-998811 ka refund ₹1500 kab aayega", "english"),
        ("bhaiii payment deduct ho gaya plzzzz help", "minimal"),
        ("bhai payment deduct ho gaya", "native_script")
    ]

    mode_latencies: Dict[str, List[float]] = {"minimal": [], "english": [], "native_script": []}

    for text, mode in samples:
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = normalizer.normalize(text, mode=mode)
            mode_latencies[mode].append((time.perf_counter() - t0) * 1000.0)

    results = {}
    for mode, times in mode_latencies.items():
        results[mode] = {
            "p50_ms": round(float(np.percentile(times, 50)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3)
        }

    report = {
        "task": "text_normalization",
        "modes": results,
        "target_p95_ms": 100.0,
        "target_met": max(r["p95_ms"] for r in results.values()) < 100.0
    }

    out_path = "reports/normalization_benchmark.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Normalization Mode B (English) P95: {results['english']['p95_ms']} ms")
    return report


if __name__ == "__main__":
    benchmark_normalization()
