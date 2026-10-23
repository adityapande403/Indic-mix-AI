"""ONNX Export and Graph Validation Pipeline for IndicMix AI.

Exports PyTorch checkpoints to ONNX format with dynamic batch and sequence lengths,
runs graph validation with onnx.checker, and validates numerical parity against ONNX Runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from train_classifier import IndicNeuralClassifier, SimpleVocab


def export_classifier_to_onnx(
    checkpoint_path: str,
    output_onnx_path: str,
    vocab_path: str,
    opset_version: int = 17
) -> Dict[str, Any]:
    """Export trained IndicNeuralClassifier to ONNX and verify parity."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    num_classes = len(checkpoint["all_labels"])
    vocab_size = checkpoint["vocab_size"]
    embed_dim = checkpoint["embed_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    max_length = checkpoint.get("max_length", 64)

    model = IndicNeuralClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    # Prepare dummy input for trace / script export
    dummy_input_ids = torch.randint(0, vocab_size, (2, 32), dtype=torch.long)
    dummy_attention_mask = torch.ones((2, 32), dtype=torch.long)

    print(f"Exporting PyTorch model to ONNX: {output_onnx_path} (Opset {opset_version})...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        }
    )

    # 1. Graph validation
    print("Validating ONNX graph with onnx.checker...")
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX graph validation passed successfully!")

    # 2. Parity check with ONNX Runtime
    print("Verifying numerical parity against ONNX Runtime...")
    ort_session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    
    test_batch_sizes = [1, 4, 8]
    max_diff = 0.0

    for bs in test_batch_sizes:
        test_ids = torch.randint(0, vocab_size, (bs, 48), dtype=torch.long)
        test_mask = torch.ones((bs, 48), dtype=torch.long)

        # PyTorch forward
        with torch.no_grad():
            torch_logits = model(test_ids, test_mask).numpy()

        # ONNX Runtime forward
        ort_inputs = {
            "input_ids": test_ids.numpy().astype(np.int64),
            "attention_mask": test_mask.numpy().astype(np.int64)
        }
        ort_logits = ort_session.run(None, ort_inputs)[0]

        diff = float(np.max(np.abs(torch_logits - ort_logits)))
        max_diff = max(max_diff, diff)

    print(f"Numerical parity check completed. Max absolute difference: {max_diff:.6e}")
    is_parity_ok = max_diff < 1e-4

    model_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)

    export_report = {
        "output_onnx_path": output_onnx_path,
        "model_size_mb": round(model_size_mb, 3),
        "opset_version": opset_version,
        "max_absolute_diff": max_diff,
        "parity_verified": is_parity_ok,
        "labels": checkpoint["all_labels"],
        "num_classes": num_classes
    }

    report_path = output_onnx_path.replace(".onnx", "_export_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(export_report, f, indent=2)

    return export_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/checkpoints/classifier/pytorch_model.bin")
    parser.add_argument("--output_onnx", default="models/onnx/classifier.onnx")
    parser.add_argument("--vocab", default="models/checkpoints/classifier/vocab.json")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_classifier_to_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output_onnx,
        vocab_path=args.vocab,
        opset_version=args.opset
    )
