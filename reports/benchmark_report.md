# IndicMix AI: Comprehensive Benchmark Report

Measured on Reference Hardware: Intel Core i7-14700K (20 Cores / 28 Threads), Windows 11 Pro.

## 1. Classification Latency & Throughput

- **Cold Start Latency**: `0.40 ms`
- **Tokenization Only (P95)**: `0.017 ms`
- **INT8 Inference Only (P95)**: `0.034 ms`
- **End-to-End Latency (P95)**: `0.050 ms` *(Stretch target < 8 ms: **MET**)*

### Batched Inference Throughput

| Batch Size | Avg Latency (ms) | P50 (ms) | P95 (ms) | Throughput (req/s) |
| --- | --- | --- | --- | --- |
| batch_1 | 0.029 | 0.028 | 0.033 | **34537.5** |
| batch_4 | 0.054 | 0.053 | 0.059 | **74482.3** |
| batch_8 | 0.081 | 0.073 | 0.116 | **98699.6** |
| batch_16 | 0.11 | 0.094 | 0.193 | **146054.7** |
| batch_32 | 0.16 | 0.139 | 0.228 | **200060.0** |

## 2. PII Detection & Redaction Latency

- **Detection Latency (P95)**: `0.058 ms`
- **Redaction Latency (P95)**: `0.004 ms`
- **Total PII Latency (P95)**: `0.063 ms` *(Target < 12 ms: **MET**)*

## 3. Text Normalization Latency

- **Mode A (Minimal) P95**: `0.016 ms`
- **Mode B (English) P95**: `0.036 ms`
- **Mode C (Native Script) P95**: `0.010 ms`
