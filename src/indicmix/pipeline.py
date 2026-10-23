"""Unified IndicMix Pipeline Engine.

Orchestrates Language Identification, Intent Classification, Content Moderation,
Hybrid PII Detection & Redaction, and Multi-Mode Text Normalization.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .calibration import TemperatureScaler
from .config import IndicMixConfig, ModelConfig
from .intent import IntentClassifier, IntentResult
from .language_detection import LanguageIdentifier, LanguageProfileResult
from .moderation import ModerationEngine, ModerationResult
from .normalization import NormalizationResult, TextNormalizer
from .pii_detection import EntitySpan, HybridPIIDetector
from .redaction import PIIRedactor, RedactionResult


@dataclass
class IndicMixAnalysisResult:
    """Unified response containing multi-task analysis outputs."""
    input_id: str
    source_text: str
    language_profile: Optional[Dict[str, Any]] = None
    intent: Optional[Dict[str, Any]] = None
    moderation: Optional[Dict[str, Any]] = None
    pii: Optional[Dict[str, Any]] = None
    normalization: Optional[Dict[str, Any]] = None
    model_version: str = "indicmix-0.1.0"
    latency_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_id": self.input_id,
            "source_text": self.source_text,
            "language_profile": self.language_profile,
            "intent": self.intent,
            "moderation": self.moderation,
            "pii": self.pii,
            "normalization": self.normalization,
            "model_version": self.model_version,
            "latency_ms": round(self.latency_ms, 2),
            "warnings": self.warnings
        }


class IndicMix:
    """Main IndicMix AI Pipeline Engine."""

    def __init__(self, config: Optional[IndicMixConfig] = None):
        self.config = config or IndicMixConfig()
        
        # Lazy-loaded component caches
        self._language_identifier: Optional[LanguageIdentifier] = None
        self._intent_classifier: Optional[IntentClassifier] = None
        self._moderation_engine: Optional[ModerationEngine] = None
        self._pii_detector: Optional[HybridPIIDetector] = None
        self._pii_redactor: Optional[PIIRedactor] = None
        self._normalizer: Optional[TextNormalizer] = None

        self._onnx_sessions: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, config_path_or_dict: Union[str, Dict[str, Any]]) -> IndicMix:
        """Create IndicMix instance from a YAML configuration file path or dictionary."""
        if isinstance(config_path_or_dict, str):
            cfg = IndicMixConfig.from_yaml(config_path_or_dict)
        elif isinstance(config_path_or_dict, dict):
            # Parse dict
            cfg = IndicMixConfig(raw_config=config_path_or_dict)
        else:
            cfg = IndicMixConfig()
        return cls(config=cfg)

    # --- Component Getters with Lazy Initialization ---

    @property
    def language_identifier(self) -> LanguageIdentifier:
        if self._language_identifier is None:
            self._language_identifier = LanguageIdentifier()
        return self._language_identifier

    @property
    def intent_classifier(self) -> IntentClassifier:
        if self._intent_classifier is None:
            # Check if ONNX model is configured and available
            session = self._get_onnx_session("intent")
            self._intent_classifier = IntentClassifier(onnx_session=session)
        return self._intent_classifier

    @property
    def moderation_engine(self) -> ModerationEngine:
        if self._moderation_engine is None:
            session = self._get_onnx_session("moderation")
            self._moderation_engine = ModerationEngine(onnx_session=session)
        return self._moderation_engine

    @property
    def pii_detector(self) -> HybridPIIDetector:
        if self._pii_detector is None:
            self._pii_detector = HybridPIIDetector(enable_model=True)
        return self._pii_detector

    @property
    def pii_redactor(self) -> PIIRedactor:
        if self._pii_redactor is None:
            self._pii_redactor = PIIRedactor()
        return self._pii_redactor

    @property
    def normalizer(self) -> TextNormalizer:
        if self._normalizer is None:
            self._normalizer = TextNormalizer(strict_entity_preservation=True)
        return self._normalizer

    def _get_onnx_session(self, task_name: str) -> Optional[Any]:
        """Lazy load and cache ONNX Runtime inference sessions."""
        if task_name in self._onnx_sessions:
            return self._onnx_sessions[task_name]

        model_cfg = self.config.models.get(task_name)
        if not model_cfg or not model_cfg.enabled or model_cfg.backend != "onnx":
            return None

        # Try quantized model first, then fallback
        candidate_paths = [model_cfg.model_path, model_cfg.fallback_path]
        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    import onnxruntime as ort
                    sess_options = ort.SessionOptions()
                    sess_options.intra_op_num_threads = self.config.runtime.num_threads
                    sess_options.inter_op_num_threads = self.config.runtime.inter_op_num_threads
                    providers = [self.config.runtime.execution_provider]
                    session = ort.InferenceSession(path, sess_options, providers=providers)
                    self._onnx_sessions[task_name] = session
                    return session
                except Exception:
                    pass
        return None

    # --- Independent Core APIs ---

    def detect_language(self, text: str) -> LanguageProfileResult:
        """Identify language profile and code-mixing status."""
        return self.language_identifier.identify(text)

    def classify_intent(self, text: str) -> IntentResult:
        """Classify user intent with uncertainty abstention."""
        return self.intent_classifier.classify(text)

    def moderate(self, text: str) -> ModerationResult:
        """Evaluate content safety across harassment, abuse, threat, spam, profanity."""
        return self.moderation_engine.moderate(text)

    def detect_pii(self, text: str) -> List[EntitySpan]:
        """Detect sensitive PII entities (phones, emails, UPI, PAN, names, addresses)."""
        return self.pii_detector.detect(text)

    def redact_pii(
        self,
        text: str,
        redact_names: bool = True,
        spans: Optional[List[EntitySpan]] = None
    ) -> RedactionResult:
        """Redact sensitive PII entities from text."""
        detected_spans = spans if spans is not None else self.detect_pii(text)
        return self.pii_redactor.redact(text, detected_spans, redact_names=redact_names)

    def normalize(
        self,
        text: str,
        target_language: str = "en",
        mode: str = "english"
    ) -> NormalizationResult:
        """Normalize code-mixed text with strict entity preservation."""
        lang_res = self.detect_language(text)
        return self.normalizer.normalize(
            text,
            mode=mode,
            target_language=target_language,
            source_language=lang_res.primary_language
        )

    # --- Multi-Task Unified Analysis API ---

    def analyze(
        self,
        text: str,
        tasks: Optional[List[str]] = None,
        normalization_mode: str = "english",
        target_language: str = "en",
        redact_names: bool = True,
        input_id: Optional[str] = None
    ) -> IndicMixAnalysisResult:
        """Perform unified multi-task analysis on text."""
        t0 = time.perf_counter()
        active_tasks = set(tasks or ["language", "intent", "moderation", "pii", "normalization"])
        req_id = input_id or f"req_{uuid.uuid4().hex[:8]}"

        warnings: List[str] = []
        lang_dict = None
        intent_dict = None
        mod_dict = None
        pii_dict = None
        norm_dict = None

        # 1. Language Identification
        primary_lang = "unknown"
        if "language" in active_tasks or "normalization" in active_tasks:
            lang_res = self.detect_language(text)
            primary_lang = lang_res.primary_language
            if "language" in active_tasks:
                lang_dict = {
                    "primary_language": lang_res.primary_language,
                    "confidence": round(lang_res.confidence, 4),
                    "is_code_mixed": lang_res.is_code_mixed,
                    "is_romanized": lang_res.is_romanized,
                    "is_uncertain": lang_res.is_uncertain,
                    "scores": lang_res.scores,
                    "abstention_reason": lang_res.abstention_reason
                }
                if lang_res.is_uncertain and lang_res.abstention_reason:
                    warnings.append(lang_res.abstention_reason)

        # 2. Intent Classification
        if "intent" in active_tasks:
            intent_res = self.classify_intent(text)
            intent_dict = intent_res.to_dict()
            if intent_res.is_abstained and intent_res.abstention_reason:
                warnings.append(intent_res.abstention_reason)

        # 3. Moderation
        if "moderation" in active_tasks:
            mod_res = self.moderate(text)
            mod_dict = mod_res.to_dict()

        # 4. PII Detection & Redaction
        if "pii" in active_tasks:
            spans = self.detect_pii(text)
            redaction_res = self.redact_pii(text, redact_names=redact_names, spans=spans)
            pii_dict = {
                "entities": [s.to_dict() for s in spans],
                "has_pii": len(spans) > 0,
                "redacted_text": redaction_res.redacted_text,
                "entity_counts": redaction_res.entity_counts
            }

        # 5. Normalization
        if "normalization" in active_tasks:
            norm_res = self.normalizer.normalize(
                text,
                mode=normalization_mode,
                target_language=target_language,
                source_language=primary_lang
            )
            norm_dict = norm_res.to_dict()
            warnings.extend(norm_res.warnings)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return IndicMixAnalysisResult(
            input_id=req_id,
            source_text=text,
            language_profile=lang_dict,
            intent=intent_dict,
            moderation=mod_dict,
            pii=pii_dict,
            normalization=norm_dict,
            latency_ms=latency_ms,
            warnings=warnings
        )

    def analyze_batch(
        self,
        texts: List[str],
        tasks: Optional[List[str]] = None,
        normalization_mode: str = "english",
        target_language: str = "en",
        redact_names: bool = True
    ) -> List[IndicMixAnalysisResult]:
        """Perform batched multi-task analysis."""
        return [
            self.analyze(
                t,
                tasks=tasks,
                normalization_mode=normalization_mode,
                target_language=target_language,
                redact_names=redact_names
            )
            for t in texts
        ]
