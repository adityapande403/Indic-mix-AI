"""Pydantic schemas for IndicMix AI REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    service: str = "IndicMix AI Production API"
    components: Dict[str, str] = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    version: str = "0.1.0"
    runtime: str
    execution_provider: str
    models_configured: Dict[str, Any]


class IntentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Input text to classify")
    taxonomy: Optional[List[str]] = None


class IntentResponse(BaseModel):
    intent: str
    confidence: float
    all_scores: Dict[str, float]
    is_abstained: bool
    abstention_reason: Optional[str] = None


class ModerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    thresholds: Optional[Dict[str, float]] = None


class ModerationResponse(BaseModel):
    is_flagged: bool
    flagged_categories: List[str]
    category_scores: Dict[str, float]
    highest_severity_category: Optional[str] = None
    explanation: Optional[str] = None


class PIIEntityItem(BaseModel):
    type: str
    start: int
    end: int
    text: str
    confidence: float
    source: str
    utf16_start: Optional[int] = None
    utf16_end: Optional[int] = None


class PIIRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    redact_names: bool = True
    custom_masks: Optional[Dict[str, str]] = None


class PIIResponse(BaseModel):
    entities: List[PIIEntityItem]
    has_pii: bool
    redacted_text: str
    entity_counts: Dict[str, int]


class NormalizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    mode: str = Field("english", description="Options: 'minimal', 'english', 'native_script'")
    target_language: str = Field("en", description="Target language code (e.g. 'en', 'hi_devanagari', 'te_telugu', 'bn_bengali')")


class NormalizeResponse(BaseModel):
    source_text: str
    normalized_text: str
    mode: str
    target_language: str
    preserved_entities: List[str]
    preservation_intact: bool
    warnings: List[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    tasks: Optional[List[str]] = Field(default=None, description="Tasks: 'language', 'intent', 'moderation', 'pii', 'normalization'")
    normalization_mode: str = "english"
    target_language: str = "en"
    redact_names: bool = True
    input_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    input_id: str
    source_text: str
    language_profile: Optional[Dict[str, Any]] = None
    intent: Optional[Dict[str, Any]] = None
    moderation: Optional[Dict[str, Any]] = None
    pii: Optional[Dict[str, Any]] = None
    normalization: Optional[Dict[str, Any]] = None
    model_version: str = "indicmix-0.1.0"
    latency_ms: float
    warnings: List[str] = Field(default_factory=list)


class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=128)
    tasks: Optional[List[str]] = None
    normalization_mode: str = "english"
    target_language: str = "en"
    redact_names: bool = True


class BatchAnalyzeResponse(BaseModel):
    results: List[AnalyzeResponse]
    total_processed: int
    batch_latency_ms: float


class ProxyAnonymizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Raw customer message to anonymize before sending to LLM")
    redact_names: bool = Field(True, description="Whether to mask person names")
    session_id: Optional[str] = Field(None, description="Optional custom session identifier")


class ProxyAnonymizeResponse(BaseModel):
    session_id: str
    original_text: str
    sanitized_text: str
    token_mapping: Dict[str, str] = Field(..., description="Mapping from placeholder tokens to original entities")
    detected_entities: List[PIIEntityItem]
    entity_counts: Dict[str, int]


class ProxyDeanonymizeRequest(BaseModel):
    text_from_llm: str = Field(..., min_length=1, max_length=50000, description="Sanitized text returned by LLM")
    session_id: Optional[str] = Field(None, description="Session ID to retrieve token mapping from")
    token_mapping: Optional[Dict[str, str]] = Field(None, description="Explicit token mapping if stateless")


class ProxyDeanonymizeResponse(BaseModel):
    rehydrated_text: str
    tokens_replaced: int
