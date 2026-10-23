"""Tests for ONNX Export and Quantization Parity."""

import os
import pytest
import numpy as np
import onnx
import onnxruntime as ort
import torch


@pytest.mark.skipif(not os.path.exists("models/onnx/classifier.onnx"), reason="ONNX model not exported")
def test_onnx_fp32_checker():
    model = onnx.load("models/onnx/classifier.onnx")
    onnx.checker.check_model(model)


@pytest.mark.skipif(not os.path.exists("models/onnx/classifier_quantized.onnx"), reason="Quantized ONNX model not exported")
def test_onnx_int8_checker():
    model = onnx.load("models/onnx/classifier_quantized.onnx")
    onnx.checker.check_model(model)


@pytest.mark.skipif(not os.path.exists("models/onnx/classifier.onnx"), reason="ONNX model not exported")
def test_onnx_inference_and_dynamic_shapes():
    session = ort.InferenceSession("models/onnx/classifier.onnx", providers=["CPUExecutionProvider"])
    
    # Test multiple batch sizes and sequence lengths
    for bs in [1, 3]:
        for seq_len in [16, 48]:
            inp_ids = np.zeros((bs, seq_len), dtype=np.int64)
            mask = np.ones((bs, seq_len), dtype=np.int64)
            outputs = session.run(None, {"input_ids": inp_ids, "attention_mask": mask})
            assert len(outputs) == 1
            assert outputs[0].shape[0] == bs


@pytest.mark.skipif(
    not (os.path.exists("models/onnx/classifier.onnx") and os.path.exists("models/onnx/classifier_quantized.onnx")),
    reason="ONNX models not available"
)
def test_fp32_int8_prediction_agreement():
    sess_fp32 = ort.InferenceSession("models/onnx/classifier.onnx", providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession("models/onnx/classifier_quantized.onnx", providers=["CPUExecutionProvider"])

    # Test random inputs
    np.random.seed(42)
    inp_ids = np.random.randint(0, 500, (5, 32), dtype=np.int64)
    mask = np.ones((5, 32), dtype=np.int64)
    ort_in = {"input_ids": inp_ids, "attention_mask": mask}

    out_fp32 = sess_fp32.run(None, ort_in)[0]
    out_int8 = sess_int8.run(None, ort_in)[0]

    preds_fp32 = np.argmax(out_fp32, axis=-1)
    preds_int8 = np.argmax(out_int8, axis=-1)

    agreement = np.mean(preds_fp32 == preds_int8)
    assert agreement >= 0.80, f"Quantized prediction agreement too low: {agreement}"
