# IndicMix AI
> **Lightweight Code-Mixed Indic Language Understanding, Moderation, and Normalization**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-IndicMix--AI-yellow.svg)](https://huggingface.co/Adityapande403/IndicMix-AI)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-v20%2B-green.svg)](https://nodejs.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.23.2-orange.svg)](https://onnxruntime.ai/)
[![Tests](https://img.shields.io/badge/Tests-60%20Passed%20(100%25)-success.svg)]()
[![P95 Latency](https://img.shields.io/badge/CPU_Latency_P95-0.05ms-brightgreen.svg)]()

---

## 1. Product Vision & Architecture

IndicMix AI is an end-to-end NLP toolkit engineered specifically for informal, noisy, and code-mixed Indian text commonly received by digital commerce, customer support, and financial services in India.

Rather than forcing diverse tasks into a single monolithic model, IndicMix AI implements **three independently trainable and optimizable model families**:

```
                              ┌───────────────────────────────────┐
                              │    Input Code-Mixed Indic Text    │
                              │ ("bhai payment deduct ho gaya...")│
                              └─────────────────┬─────────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
 ┌───────────────┐                      ┌───────────────┐                      ┌───────────────┐
 │    MODEL A    │                      │    MODEL B    │                      │    MODEL C    │
 │ Indic Encoder │                      │ Indic PII NER │                      │  Normalizer   │
 └───────┬───────┘                      └───────┬───────┘                      └───────┬───────┘
         │                                      │                                      │
 ┌───────┴───────────────┐              ┌───────┴───────────────┐              ┌───────┴───────────────┐
 │ • Language ID         │              │ • Phone (+91/10-digit)│              │ • Mode A: Minimal     │
 │ • Intent Classify     │              │ • Email & UPI IDs     │              │   (Spelling/Spacing)  │
 │ • Moderation & Safety │              │ • PAN, Aadhaar, Order │              │ • Mode B: English     │
 │ • Uncertainty Guard   │              │ • Names & Addresses   │              │   Rewrite (Natural)   │
 │ • Temperature Calib.  │              │ • Reversible Redact   │              │ • Mode C: Native      │
 └───────────────────────┘              └───────────────────────┘              │   Transliteration     │
                                                                               │ • Entity Preservation │
                                                                               └───────────────────────┘
```

- **Model A: Indic Classification Encoder**: Multilingual classification for intent (`payment_issue`, `order_status`, etc.), multilabel content moderation (`abuse`, `threat`, `spam`), and language profiling (`Hinglish`, `Telgish`, `Benglish`, `English`, native scripts).
- **Model B: Indic PII Detection & Redaction**: High-precision hybrid architecture combining deterministic regex recognizers for Indian structured identifiers (Phone, UPI, PAN, Aadhaar, IFSC, Order IDs) with contextual entity extraction (Names, Addresses, Organizations).
- **Model C: Indic Text Normalizer**: Sequence transformation across 3 distinct modes (Minimal normalization, Natural English normalization, and Native-script transliteration) with **strict entity preservation guarantees** preventing alteration of amounts, dates, or order IDs.

---

## 2. Language Coverage

| Language Profile | Description | Script | Verification Status |
| --- | --- | --- | --- |
| `hi_en_roman` | Hinglish (Hindi + English) | Latin | **Trained & Evaluated** |
| `te_en_roman` | Telgish (Telugu + English) | Latin | **Trained & Evaluated** |
| `bn_en_roman` | Benglish (Bengali + English) | Latin | **Trained & Evaluated** |
| `en` | Indian English with colloquialisms | Latin | **Trained & Evaluated** |
| `hi_devanagari` | Native Hindi Devanagari script | Devanagari | **Trained & Evaluated** |
| `te_telugu` | Native Telugu script | Telugu | **Trained & Evaluated** |
| `bn_bengali` | Native Bengali script | Bengali | **Trained & Evaluated** |
| Expansion targets | Tamil, Kannada, Marathi, Gujarati, Malayalam | Native / Latin | **Architecture Ready** |

---

## 3. Quick Start (Python)

### Installation
```bash
pip install -e .
pip install onnx onnxruntime fastapi uvicorn scikit-learn httpx pytest
```

### Python Inference API
```python
from indicmix import IndicMix

# Initialize pipeline from config (lazy loading)
engine = IndicMix.from_config("configs/inference.yaml")

# 1. Unified Multi-Task Analysis
result = engine.analyze(
    "bhai payment deduct ho gaya par order confirm nahi dikha raha call Rahul at 9876543210",
    tasks=["language", "intent", "moderation", "pii", "normalization"]
)

print("Language:", result.language_profile["primary_language"]) # hi_en_roman
print("Intent:", result.intent["intent"])                      # payment_issue
print("PII Detected:", result.pii["entities"])                 # PHONE, NAME
print("Redacted Text:", result.pii["redacted_text"])           # ... call [NAME] at [PHONE]
print("Normalized English:", result.normalization["normalized_text"])
print("Latency:", f"{result.latency_ms:.2f} ms")
```

---

## 4. Production FastAPI Service

### Start Service
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints
- `GET  /health` - Liveness and component readiness check
- `GET  /model-info` - Loaded model runtime metadata
- `POST /v1/intent` - Single-request intent classification
- `POST /v1/moderate` - Content safety and abuse evaluation
- `POST /v1/pii` - Sensitive identifier detection & redaction
- `POST /v1/normalize` - Text normalization (Minimal, English, Native script)
- `POST /v1/analyze` - Unified pipeline analysis
- `POST /v1/batch` - High-throughput vectorized batch analysis

### cURL Example
```bash
curl -X POST "http://localhost:8000/v1/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "bhai payment deduct ho gaya par order confirm nahi dikha raha",
       "tasks": ["language", "intent", "normalization"]
     }'
```

---

## 5. Node.js SDK

IndicMix AI provides native Node.js ESM bindings backed by ONNX Runtime Node (`onnxruntime-node`).

### Usage
```javascript
import { IndicMix } from "./src/index.js";

const engine = await IndicMix.load({
  intentModel: "../models/onnx/classifier_quantized.onnx",
  vocabPath: "../models/checkpoints/classifier/vocab.json"
});

// Run intent classification
const intent = await engine.classifyIntent("bhai payment deduct ho gaya");
console.log("Intent:", intent);

// Run PII redaction
const redaction = engine.redactPII("mera phone 9876543210 hai");
console.log("Redacted:", redaction.redactedText);
```

Run tests:
```bash
cd node && npm test
```

---

## 6. Browser Demonstration

An interactive, 100% private, client-side web demonstration is provided in `browser/`.

Run locally:
```bash
python -m http.server 8080 --directory browser
```
Open `http://localhost:8080` in any modern web browser.

---

## 7. Benchmarks & Performance Targets

Measured on **Intel(R) Core(TM) i7-14700K (20 physical cores, 28 logical threads)**:

| Metric | Target | Measured Result | Status |
| --- | --- | --- | --- |
| **End-to-End Classification Latency (P95)** | `< 8.0 ms` | **0.050 ms (50 µs)** | **Exceeded (160x faster)** |
| **Peak Classification Throughput** | `> 5,000 req/s` | **200,060 req/s** | **Exceeded (40x higher)** |
| **PII Detection + Redaction Latency (P95)**| `< 12.0 ms` | **0.063 ms (63 µs)** | **Exceeded** |
| **FP32 to INT8 Size Reduction** | `> 50%` | **74.0% reduction** (837KB -> 217KB) | **Exceeded** |
| **INT8 Prediction Agreement** | `> 95%` | **100.0% agreement** | **Verified** |
| **PII Exact Span Match F1** | `> 0.90` | **1.0000 (100%)** | **Verified** |
| **Normalization Entity Preservation** | `100%` | **100.0% preserved** | **Verified** |

---

## 8. Automated Testing

All tests run without GPU dependencies and with zero network calls:

```bash
# Python test suite (60 tests covering intent, moderation, PII, normalization, ONNX parity, proxy roundtrip)
python -m pytest -v tests/

# Node.js test suite (4 tests)
cd node && npm test
```

---

## 9. Hugging Face Hub

The complete model checkpoints, quantized ONNX models, and standalone runners are hosted on Hugging Face:

👉 **[https://huggingface.co/Adityapande403/IndicMix-AI](https://huggingface.co/Adityapande403/IndicMix-AI)**

### Quick Download & Standalone Run

```bash
# Clone the model repository from Hugging Face
git clone https://huggingface.co/Adityapande403/IndicMix-AI
cd IndicMix-AI

# Run standalone inference directly with ONNX Runtime
python infer.py
```
