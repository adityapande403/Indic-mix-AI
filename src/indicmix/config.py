"""Configuration loader and schema validation for IndicMix AI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ModelConfig:
    enabled: bool = True
    backend: str = "onnx"  # "onnx" or "pytorch"
    model_path: Optional[str] = None
    fallback_path: Optional[str] = None
    tokenizer_path: Optional[str] = None


@dataclass
class RuntimeConfig:
    num_threads: int = 4
    inter_op_num_threads: int = 2
    execution_provider: str = "CPUExecutionProvider"
    enable_profiling: bool = False


@dataclass
class IndicMixConfig:
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> IndicMixConfig:
        """Load configuration from a YAML file."""
        if not os.path.exists(path):
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        model_dict = {}
        for name, m_data in data.get("models", {}).items():
            model_dict[name] = ModelConfig(
                enabled=m_data.get("enabled", True),
                backend=m_data.get("backend", "onnx"),
                model_path=m_data.get("model_path"),
                fallback_path=m_data.get("fallback_path"),
                tokenizer_path=m_data.get("tokenizer_path")
            )

        rt_data = data.get("runtime", {})
        runtime_cfg = RuntimeConfig(
            num_threads=rt_data.get("num_threads", 4),
            inter_op_num_threads=rt_data.get("inter_op_num_threads", 2),
            execution_provider=rt_data.get("execution_provider", "CPUExecutionProvider"),
            enable_profiling=rt_data.get("enable_profiling", False)
        )

        return cls(models=model_dict, runtime=runtime_cfg, raw_config=data)
