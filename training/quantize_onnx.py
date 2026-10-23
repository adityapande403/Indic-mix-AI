"""Quantization Pipeline using ONNX Runtime.

Quantizes exported ONNX graphs to INT8/UINT8 representations with dynamic quantization,
evaluates operator-level conversion, measures size reduction, and checks prediction agreement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_onnx_model(
    input_model_path: str,
    output_model_path: str,
    weight_type: str = "QInt8",
    test_data_path: str = "data/processed/intent/test.jsonl",
    vocab_path: str = "models/checkpoints/classifier/vocab.json"
) -> Dict[str, Any]:
    """Quantize ONNX model and benchmark size, latency, and prediction agreement."""
    print(f"Quantizing {input_model_path} -> {output_model_path} (WeightType: {weight_type})...")
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    q_weight_type = QuantType.QInt8 if weight_type.lower() == "qint8" else QuantType.QUInt8

    # Apply dynamic quantization
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=q_weight_type
    )

    # Validate quantized graph with onnx.checker
    print("Validating quantized ONNX graph...")
    q_onnx = onnx.load(output_model_path)
    onnx.checker.check_model(q_onnx)
    print("Quantized ONNX graph is valid!")

    # Compare file sizes
    orig_size_bytes = os.path.getsize(input_model_path)
    quant_size_bytes = os.path.getsize(output_model_path)
    size_reduction_pct = (1.0 - (quant_size_bytes / orig_size_bytes)) * 100.0

    print(f"Original size: {orig_size_bytes / 1024:.1f} KB")
    print(f"Quantized size: {quant_size_bytes / 1024:.1f} KB ({size_reduction_pct:.1f}% reduction)")

    # Analyze node types in quantized model (to accurately report which ops were quantized)
    op_counts: Dict[str, int] = {}
    for node in q_onnx.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    # Load both models in ONNX Runtime to benchmark agreement and latency
    sess_fp32 = ort.InferenceSession(input_model_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(output_model_path, providers=["CPUExecutionProvider"])

    # Load test inputs
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
    from train_classifier import SimpleVocab
    vocab = SimpleVocab.load(vocab_path)

    test_samples = []
    if os.path.exists(test_data_path):
        with open(test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_samples.append(json.loads(line)["text"])
    else:
        test_samples = ["bhai payment deduct ho gaya", "order kab aayega", "cancel order please"]

    # Benchmark prediction agreement & latency
    agreements = 0
    total = len(test_samples)
    latencies_fp32 = []
    latencies_int8 = []

    for text in test_samples:
        inp_ids, mask = vocab.encode(text, max_length=64)
        ort_in = {
            "input_ids": np.array([inp_ids], dtype=np.int64),
            "attention_mask": np.array([mask], dtype=np.int64)
        }

        # FP32 timing
        t0 = time.perf_counter()
        out_fp32 = sess_fp32.run(None, ort_in)[0]
        latencies_fp32.append((time.perf_counter() - t0) * 1000.0)

        # INT8 timing
        t0 = time.perf_counter()
        out_int8 = sess_int8.run(None, ort_in)[0]
        latencies_int8.append((time.perf_counter() - t0) * 1000.0)

        pred_fp32 = int(np.argmax(out_fp32, axis=-1)[0])
        pred_int8 = int(np.argmax(out_int8, axis=-1)[0])

        if pred_fp32 == pred_int8:
            agreements += 1

    agreement_pct = (agreements / total) * 100.0 if total > 0 else 100.0
    p50_fp32 = float(np.percentile(latencies_fp32, 50))
    p95_fp32 = float(np.percentile(latencies_fp32, 95))
    p50_int8 = float(np.percentile(latencies_int8, 50))
    p95_int8 = float(np.percentile(latencies_int8, 95))

    report = {
        "original_model_path": input_model_path,
        "quantized_model_path": output_model_path,
        "weight_type": weight_type,
        "original_size_kb": round(orig_size_bytes / 1024, 2),
        "quantized_size_kb": round(quant_size_bytes / 1024, 2),
        "size_reduction_pct": round(size_reduction_pct, 2),
        "prediction_agreement_pct": round(agreement_pct, 2),
        "latency_p50_fp32_ms": round(p50_fp32, 3),
        "latency_p95_fp32_ms": round(p95_fp32, 3),
        "latency_p50_int8_ms": round(p50_int8, 3),
        "latency_p95_int8_ms": round(p95_int8, 3),
        "graph_operator_counts": op_counts,
        "note": "Quantized weights to INT8. Note that certain operations (LayerNorm, Softmax) remain in FP32 precision per ONNX spec."
    }

    report_path = output_model_path.replace(".onnx", "_quantization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Quantization complete. Agreement: {agreement_pct:.1f}%, INT8 Latency P95: {p95_int8:.3f} ms")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", default="models/onnx/classifier.onnx")
    parser.add_argument("--output_model", default="models/onnx/classifier_quantized.onnx")
    parser.add_argument("--weight_type", default="QInt8")
    args = parser.parse_args()

    quantize_onnx_model(
        input_model_path=args.input_model,
        output_model_path=args.output_model,
        weight_type=args.weight_type
    )
