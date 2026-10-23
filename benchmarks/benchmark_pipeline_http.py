"""Multi-Tier Production Benchmark Suite for IndicMix AI.

Executes and reports four distinct evaluation tiers:
1. Tier 1: Model-Only INT8/FP32 Classifier Latency
2. Tier 2: In-Process Full Multi-Task Pipeline Latency (LangID + PII + Redaction + Normalization + Intent)
3. Tier 3: Enterprise LLM Privacy Proxy Round-Trip Latency (Anonymize + De-anonymize)
4. Tier 4: Live HTTP API Concurrency & Load Benchmark (Concurrency 1, 5, 20)
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indicmix.pipeline import IndicMix
from indicmix.proxy import EnterprisePrivacyProxy
from api.app import app
from fastapi.testclient import TestClient


BENCHMARK_TEXTS = [
    "bhai payment deduct ho gaya par order confirm nahi dikha raha ₹1500 kat gaye ORD-887766 call 9876543210",
    "naaku order deliver avvaledu dabbu cut ayindi kani item raledu 8765432109 ki call cheyandi",
    "taka kete geche kintu order confirm hoyni refund kab aayega UPI rahul@okhdfcbank",
    "mera order abhi tak kaha pahucha hai track karke batao please ORD-998811",
    "Mera payment cancel kar do please urgently ₹2000 wapas transfer kijiye"
]


def compute_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 4),
        "median_p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p90_ms": round(float(np.percentile(arr, 90)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "min_ms": round(float(np.min(arr)), 4),
        "max_ms": round(float(np.max(arr)), 4)
    }


def benchmark_full_pipeline(engine: IndicMix, num_iterations: int = 500) -> Dict[str, Any]:
    print(f"--- Running Tier 2: In-Process Full Pipeline Benchmark ({num_iterations} runs) ---")
    # Warmup
    for _ in range(20):
        engine.analyze(BENCHMARK_TEXTS[0])

    latencies = []
    for i in range(num_iterations):
        text = BENCHMARK_TEXTS[i % len(BENCHMARK_TEXTS)]
        t0 = time.perf_counter()
        engine.analyze(text)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    stats = compute_percentiles(latencies)
    throughput = 1000.0 / stats["mean_ms"] if stats["mean_ms"] > 0 else 0
    stats["throughput_req_per_sec"] = round(throughput, 1)
    print(f"Full Pipeline: Median={stats['median_p50_ms']} ms, P95={stats['p95_ms']} ms, Throughput={stats['throughput_req_per_sec']} req/s")
    return stats


def benchmark_privacy_proxy(proxy: EnterprisePrivacyProxy, num_iterations: int = 500) -> Dict[str, Any]:
    print(f"--- Running Tier 3: Enterprise Privacy Proxy Benchmark ({num_iterations} runs) ---")
    # Warmup
    for _ in range(20):
        sess = proxy.anonymize(BENCHMARK_TEXTS[0])
        proxy.deanonymize_session(sess.session_id, "Sample LLM reply for [NAME_1] and [PHONE_1]")

    anon_latencies = []
    deanon_latencies = []
    roundtrip_latencies = []

    for i in range(num_iterations):
        text = BENCHMARK_TEXTS[i % len(BENCHMARK_TEXTS)]
        
        # Anonymize
        t0 = time.perf_counter()
        sess = proxy.anonymize(text, redact_names=True)
        t_anon = (time.perf_counter() - t0) * 1000.0
        anon_latencies.append(t_anon)

        # De-anonymize
        simulated_llm = f"LLM resolved issue for text: {sess.sanitized_text}"
        t1 = time.perf_counter()
        proxy.deanonymize_session(sess.session_id, simulated_llm)
        t_deanon = (time.perf_counter() - t1) * 1000.0
        deanon_latencies.append(t_deanon)

        roundtrip_latencies.append(t_anon + t_deanon)

    stats = {
        "anonymize": compute_percentiles(anon_latencies),
        "deanonymize": compute_percentiles(deanon_latencies),
        "roundtrip": compute_percentiles(roundtrip_latencies)
    }
    stats["roundtrip_throughput_req_sec"] = round(1000.0 / stats["roundtrip"]["mean_ms"], 1)
    print(f"Privacy Proxy Roundtrip: Median={stats['roundtrip']['median_p50_ms']} ms, P95={stats['roundtrip']['p95_ms']} ms")
    return stats


def benchmark_http_concurrency(client: TestClient, concurrency_levels: List[int] = [1, 5, 20], requests_per_level: int = 200) -> Dict[str, Any]:
    print(f"--- Running Tier 4: Live HTTP API Concurrency Benchmark ---")
    results = {}

    for c in concurrency_levels:
        payloads = [
            {"text": BENCHMARK_TEXTS[i % len(BENCHMARK_TEXTS)], "normalization_mode": "english"}
            for i in range(requests_per_level)
        ]

        def send_request(p):
            t0 = time.perf_counter()
            resp = client.post("/v1/analyze", json=p)
            lat = (time.perf_counter() - t0) * 1000.0
            return lat, resp.status_code

        t_start = time.perf_counter()
        latencies = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
            fut_list = [executor.submit(send_request, p) for p in payloads]
            for fut in concurrent.futures.as_completed(fut_list):
                lat, status = fut.result()
                if status == 200:
                    latencies.append(lat)

        total_time_sec = time.perf_counter() - t_start
        stats = compute_percentiles(latencies)
        throughput = len(latencies) / total_time_sec
        stats["concurrency"] = c
        stats["throughput_req_per_sec"] = round(throughput, 1)
        stats["total_requests"] = len(latencies)
        results[f"concurrency_{c}"] = stats
        print(f"HTTP Concurrency={c}: Median={stats['median_p50_ms']} ms, P95={stats['p95_ms']} ms, Throughput={stats['throughput_req_per_sec']} req/s")

    return results


def run_all_benchmarks(output_path: str = "reports/production_benchmarks.json"):
    print("=== IndicMix AI: Comprehensive Multi-Tier Benchmark Suite ===")
    engine = IndicMix()
    proxy = EnterprisePrivacyProxy()
    client = TestClient(app)

    # Tier 1 from existing validate_export.py report
    tier1_path = "evaluation/results/onnx_validation_report.json"
    tier1_data = {}
    if os.path.exists(tier1_path):
        with open(tier1_path, "r", encoding="utf-8") as f:
            tier1_data = json.load(f)

    tier2_data = benchmark_full_pipeline(engine, num_iterations=400)
    tier3_data = benchmark_privacy_proxy(proxy, num_iterations=400)
    tier4_data = benchmark_http_concurrency(client, concurrency_levels=[1, 5, 20], requests_per_level=150)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "platform": sys.platform,
            "python_version": sys.version
        },
        "tier1_model_only": tier1_data.get("latency_benchmarks", {}),
        "tier2_full_pipeline": tier2_data,
        "tier3_privacy_proxy": tier3_data,
        "tier4_http_concurrency": tier4_data
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll benchmark results successfully written to: {output_path}")
    return summary


if __name__ == "__main__":
    run_all_benchmarks()
