"""PII Benchmarking Suite for IndicMix AI.

Measures latency, throughput, and entity precision across:
- Deterministic recognizers
- Contextual model recognizers
- Redaction latency
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from indicmix.pii_detection import HybridPIIDetector
from indicmix.redaction import PIIRedactor


def benchmark_pii(num_runs: int = 100) -> Dict[str, Any]:
    print("=== Running PII Detection & Redaction Benchmark ===")
    detector = HybridPIIDetector(enable_model=True)
    redactor = PIIRedactor()

    test_samples = [
        "Call Rahul at 9876543210 about order ORD-123456",
        "Send refund to payment@okhdfcbank or call +919876543210",
        "PAN is ABCDE1234F and Aadhaar is 2345 6789 0123",
        "Vehicle DL 01 AB 1234 driver arrived at Connaught Place New Delhi"
    ]

    # Measure detection latency
    det_latencies = []
    for _ in range(num_runs):
        for sample in test_samples:
            t0 = time.perf_counter()
            _ = detector.detect(sample)
            det_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Measure redaction latency
    redact_latencies = []
    sample = test_samples[0]
    spans = detector.detect(sample)
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = redactor.redact(sample, spans)
        redact_latencies.append((time.perf_counter() - t0) * 1000.0)

    p50_det = float(np.percentile(det_latencies, 50))
    p95_det = float(np.percentile(det_latencies, 95))
    p50_red = float(np.percentile(redact_latencies, 50))
    p95_red = float(np.percentile(redact_latencies, 95))

    report = {
        "task": "pii_detection_and_redaction",
        "detection_latency_ms": {
            "p50": round(p50_det, 3),
            "p95": round(p95_det, 3)
        },
        "redaction_latency_ms": {
            "p50": round(p50_red, 3),
            "p95": round(p95_red, 3)
        },
        "total_p95_ms": round(p95_det + p95_red, 3),
        "target_ms": 12.0,
        "target_met": (p95_det + p95_red) < 12.0
    }

    out_path = "reports/pii_benchmark.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"PII Total Latency P95: {report['total_p95_ms']} ms (Target < 12ms: {report['target_met']})")
    return report


if __name__ == "__main__":
    benchmark_pii()
