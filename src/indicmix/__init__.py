"""IndicMix AI: Lightweight Code-Mixed Indic Language Understanding, Moderation, and Normalization."""

from .calibration import TemperatureScaler, compute_expected_calibration_error, compute_brier_score
from .config import IndicMixConfig, ModelConfig, RuntimeConfig
from .intent import IntentClassifier, IntentResult
from .language_detection import LanguageIdentifier, LanguageProfileResult
from .metrics import (
    compute_classification_metrics,
    compute_multilabel_moderation_metrics,
    compute_pii_span_metrics,
    compute_normalization_metrics
)
from .moderation import ModerationEngine, ModerationResult
from .normalization import (
    NormalizationResult,
    TextNormalizer,
    check_negation_polarity,
    verify_polarity_preservation,
    verify_entity_preservation
)
from .pii_detection import (
    ContextualPIIRecognizer,
    DeterministicPIIRecognizer,
    EntitySpan,
    HybridPIIDetector,
    validate_verhoeff_checksum,
    generate_verhoeff_check_digit
)
from .pipeline import IndicMix, IndicMixAnalysisResult
from .proxy import AnonymizationSession, EnterprisePrivacyProxy
from .redaction import PIIRedactor, RedactionResult
from .text_profiles import (
    TextProfile,
    analyze_text_profile,
    codepoint_to_utf16_offset,
    utf16_to_codepoint_offset
)

__version__ = "0.1.0"

__all__ = [
    "IndicMix",
    "IndicMixAnalysisResult",
    "IndicMixConfig",
    "ModelConfig",
    "RuntimeConfig",
    "LanguageIdentifier",
    "LanguageProfileResult",
    "IntentClassifier",
    "IntentResult",
    "ModerationEngine",
    "ModerationResult",
    "HybridPIIDetector",
    "DeterministicPIIRecognizer",
    "ContextualPIIRecognizer",
    "EntitySpan",
    "validate_verhoeff_checksum",
    "generate_verhoeff_check_digit",
    "PIIRedactor",
    "RedactionResult",
    "TextNormalizer",
    "NormalizationResult",
    "check_negation_polarity",
    "verify_polarity_preservation",
    "verify_entity_preservation",
    "EnterprisePrivacyProxy",
    "AnonymizationSession",
    "TemperatureScaler",
    "compute_expected_calibration_error",
    "compute_brier_score",
    "compute_classification_metrics",
    "compute_multilabel_moderation_metrics",
    "compute_pii_span_metrics",
    "compute_normalization_metrics",
    "analyze_text_profile",
    "TextProfile",
    "codepoint_to_utf16_offset",
    "utf16_to_codepoint_offset",
]
