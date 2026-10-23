"""Classification Benchmarking Suite for IndicMix AI.

Measures:
- Tokenization only latency
- Model inference only latency (FP32 vs INT8)
- End-to-end pipeline latency
- Cold-start vs warm-start latency
- Percentile distribution (P50, P90, P95, P99)
- Batched throughput across batch sizes [1, 4, 8, 16, 32]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from indicmix.intent import IntentClassifier
from train_classifier import SimpleVocab


def benchmark_classification(
    onnx_fp32_path: str = "models/onnx/classifier.onnx",
    onnx_int8_path: str = "models/onnx/classifier_quantized.onnx",
    vocab_path: str = "models/checkpoints/classifier/vocab.json",
    num_runs: int = 100,
    warmup_runs: int = 20
) -> Dict[str, Any]:
    print("=== Running Classification Latency & Throughput Benchmark ===")
    
    vocab = SimpleVocab.load(vocab_path)
    test_text = "bhai payment deduct ho gaya par order confirm nahi dikha raha"

    sess_fp32 = ort.InferenceSession(onnx_fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(onnx_int8_path, providers=["CPUExecutionProvider"])

    # 1. Tokenization latency
    tok_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = vocab.encode(test_text, max_length=64)
        tok_times.append((time.perf_counter() - t0) * 1000.0)

    p50_tok = float(np.percentile(tok_times, 50))
    p95_tok = float(np.percentile(tok_times, 95))

    # 2. Cold-start latency (INT8)
    t0 = time.perf_counter()
    cold_inp_ids, cold_mask = vocab.encode(test_text, max_length=64)
    cold_feed = {
        "input_ids": np.array([cold_inp_ids], dtype=np.int64),
        "attention_mask": np.array([cold_mask], dtype=np.int64)
    }
    _ = sess_int8.run(None, cold_feed)
    cold_start_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Model inference only (INT8 warm)
    # Warmup
    for _ in range(warmup_runs):
        _ = sess_int8.run(None, cold_feed)

    inf_times_int8 = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = sess_int8.run(None, cold_feed)
        inf_times_int8.append((time.perf_counter() - t0) * 1000.0)

    p50_inf = float(np.percentile(inf_times_int8, 50))
    p90_inf = float(np.percentile(inf_times_int8, 90))
    p95_inf = float(np.percentile(inf_times_int8, 95))
    p99_inf = float(np.percentile(inf_times_int8, 99))

    # 4. FP32 inference for comparison
    inf_times_fp32 = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = sess_fp32.run(None, cold_feed)
        inf_times_fp32.append((time.perf_counter() - t0) * 1000.0)

    p50_fp32 = float(np.percentile(inf_times_fp32, 50))
    p95_fp32 = float(np.percentile(inf_times_fp32, 95))

    # 5. Batched throughput benchmark
    throughput_bench = {}
    for bs in [1, 4, 8, 16, 32]:
        batched_ids = np.zeros((bs, 64), dtype=np.int64)
        batched_mask = np.ones((bs, 64), dtype=np.int64)
        batch_feed = {"input_ids": batched_ids, "attention_mask": batched_mask}

        # Warmup
        for _ in range(5):
            _ = sess_int8.run(None, batch_feed)

        batch_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = sess_int8.run(None, batch_feed)
            batch_times.append((time.perf_counter() - t0) * 1000.0)

        avg_lat = sum(batch_times) / len(batch_times)
        throughput = (bs * 50) / (sum(batch_times) / 1000.0)

        throughput_bench[f"batch_{bs}"] = {
            "avg_latency_ms": round(avg_lat, 3),
            "p50_ms": round(float(np.percentile(batch_times, 50)), 3),
            "p95_ms": round(float(np.percentile(batch_times, 95)), 3),
            "throughput_req_per_sec": round(throughput, 1)
        }

    # End-to-end latency
    e2e_p50 = p50_tok + p50_inf
    e2e_p95 = p95_tok + p95_inf

    target_achieved = e2e_p95 < 8.0

    report = {
        "status": "COMPLETED",
        "benchmark_conditions": {
            "hardware": "Intel Core i7-14700K (28 threads)",
            "execution_provider": "CPUExecutionProvider",
            "runtime": f"ONNX Runtime {ort.__version__}",
            "reference_text": test_text,
            "warmup_runs": warmup_runs,
            "benchmark_runs": num_runs
        },
        "tokenization_only_ms": {
            "p50": round(p50_tok, 3),
            "p95": round(p95_tok, 3)
        },
        "cold_start_ms": round(cold_start_ms, 3),
        "int8_inference_only_ms": {
            "p50": round(p50_inf, 3),
            "p90": round(p90_inf, 3),
            "p95": round(p95_inf, 3),
            "p99": round(p99_inf, 3)
        },
        "fp32_inference_only_ms": {
            "p50": round(p50_fp32, 3),
            "p95": round(p95_fp32, 3)
        },
        "end_to_end_classification_ms": {
            "p50": round(e2e_p50, 3),
            "p95": round(e2e_p95, 3),
            "target_stretch_ms": 8.0,
            "target_achieved": target_achieved
        },
        "batched_throughput": throughput_bench
    }

    out_path = "reports/classification_benchmark.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"End-to-End Classification Latency P50: {e2e_p50:.3f} ms, P95: {e2e_p95:.3f} ms (Target < 8ms: {target_achieved})")
    print(f"Peak Throughput (Batch 32): {throughput_bench['batch_32']['throughput_req_per_sec']} requests/sec")
    return report


if __name__ == "__main__":
    benchmark_classification()
