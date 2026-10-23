# IndicMix AI: Environment and System Inspection Report

**Date & Time:** 2026-10-23 13:27:00 IST  
**Host Architecture:** Windows 11 Pro 64-bit (10.0.26200)  
**Workspace:** `D:\Aditya\hf2`

---

## 1. Operating System & Host Environment
- **OS**: Microsoft Windows 11 Pro (Build 10.0.26200, 64-bit)
- **Total Physical Memory (RAM)**: 127.77 GB (133,974,012 KB)
- **Free Physical Memory**: 102.27 GB (107,242,720 KB)
- **System Drive (C:)**: 17.98 GB free / 322.98 GB used
- **Workspace Drive (D:)**: 318.58 GB free / 270.36 GB used (High capacity available for datasets, models, ONNX exports)

---

## 2. Compute Resources & Accelerators
- **CPU**: Intel(R) Core(TM) i7-14700K (20 physical cores, 28 logical processors, Base clock ~3.40 GHz)
  - Features supported: AVX2, FMA3, SSE4.2, VNNI (ideal for high-throughput ONNX Runtime CPU inference and INT8/UINT8 VNNI acceleration)
- **GPU**: NVIDIA GeForce RTX 5060 Ti
  - Driver Version: 610.88
  - CUDA UMD Version: 13.3
  - VRAM: 16,311 MiB (~16 GB GDDR6)
  - Current PyTorch installation: `2.7.0+cpu` (CPU only wheel installed in default user environment).
  - Note: PyTorch runs in CPU mode using the 28 logical cores with AVX2/VNNI support. ONNX Runtime leverages high-performance CPU kernels (`CPUExecutionProvider`) and can target DirectML or TensorRT where available.

---

## 3. Runtimes & Package Managers
- **Python**: 3.10.11 (`C:\Users\User\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\python.exe`)
- **pip**: 23.0.1
- **Node.js**: v22.22.2
- **npm**: 10.9.7
- **Git**: 2.49.0.windows.1

---

## 4. Installed Machine Learning & Serving Packages
- `torch`: 2.7.0 (CPU)
- `transformers`: 4.56.0
- `tokenizers`: 0.22.0
- `safetensors`: 0.6.2
- `huggingface-hub`: 0.34.4
- `pydantic`: 2.11.7
- `uvicorn`: 0.35.0
- `starlette`: 0.47.3
- `pandas`: 2.3.2
- `numpy`: 2.2.6
- `pyyaml`: 6.0.2
- `regex`: 2025.9.1
- Additional packages in installation: `pytest`, `fastapi`, `scikit-learn`, `onnx`, `onnxruntime`

---

## 5. Workspace Status
- Workspace path: `D:\Aditya\hf2`
- Initial workspace status: Clean / empty directory.
- No existing project files or unrelated source code present. Clean slate for full IndicMix AI implementation.
