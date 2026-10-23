"""Production FastAPI Service for IndicMix AI.

Exposes REST endpoints for:
- Intent Classification
- Content Moderation
- PII Detection & Redaction
- Multi-Mode Normalization
- Unified Multi-Task Analysis
- Batch Inference
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from indicmix.pipeline import IndicMix
from indicmix.proxy import EnterprisePrivacyProxy
from schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    HealthResponse,
    IntentRequest,
    IntentResponse,
    ModelInfoResponse,
    ModerationRequest,
    ModerationResponse,
    NormalizeRequest,
    NormalizeResponse,
    PIIEntityItem,
    PIIRequest,
    PIIResponse,
    ProxyAnonymizeRequest,
    ProxyAnonymizeResponse,
    ProxyDeanonymizeRequest,
    ProxyDeanonymizeResponse,
)

# Configure structured logging without logging raw PII
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("indicmix.api")

# Global pipeline instance
engine: Optional[IndicMix] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initialize models on startup, cleanup on shutdown."""
    global engine
    logger.info("Initializing IndicMix AI Pipeline...")
    config_path = "configs/inference.yaml"
    engine = IndicMix.from_config(config_path)
    logger.info("IndicMix AI Pipeline initialized successfully.")
    yield
    logger.info("Shutting down IndicMix AI service.")


app = FastAPI(
    title="IndicMix AI API",
    description="Production REST API for code-mixed Indic language understanding, moderation, PII redaction, and normalization.",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for browser demo integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def get_engine() -> IndicMix:
    global engine
    if engine is None:
        engine = IndicMix.from_config("configs/inference.yaml")
    return engine


proxy_gateway: Optional[EnterprisePrivacyProxy] = None


def get_proxy() -> EnterprisePrivacyProxy:
    global proxy_gateway
    if proxy_gateway is None:
        eng = get_engine()
        proxy_gateway = EnterprisePrivacyProxy(pii_detector=eng.pii_detector)
    return proxy_gateway


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    eng = get_engine()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        service="IndicMix AI Production API",
        components={
            "language_detector": "active",
            "intent_classifier": "active",
            "moderation_engine": "active",
            "pii_detector": "active",
            "normalizer": "active",
            "privacy_proxy": "active"
        }
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """System and loaded model metadata."""
    eng = get_engine()
    return ModelInfoResponse(
        version="0.1.0",
        runtime="ONNX Runtime CPU / PyTorch",
        execution_provider=eng.config.runtime.execution_provider,
        models_configured={
            "intent": eng.config.models.get("intent", {}).__dict__ if "intent" in eng.config.models else {},
            "moderation": eng.config.models.get("moderation", {}).__dict__ if "moderation" in eng.config.models else {},
            "pii": eng.config.models.get("pii", {}).__dict__ if "pii" in eng.config.models else {},
            "normalization": eng.config.models.get("normalization", {}).__dict__ if "normalization" in eng.config.models else {}
        }
    )


@app.post("/v1/intent", response_model=IntentResponse, tags=["Classification"])
async def classify_intent(req: IntentRequest):
    """Classify user intent with uncertainty abstention."""
    eng = get_engine()
    res = eng.classify_intent(req.text)
    return IntentResponse(
        intent=res.intent,
        confidence=res.confidence,
        all_scores=res.all_scores,
        is_abstained=res.is_abstained,
        abstention_reason=res.abstention_reason
    )


@app.post("/v1/moderate", response_model=ModerationResponse, tags=["Safety"])
async def moderate(req: ModerationRequest):
    """Audit content safety across abuse, threats, harassment, hate speech, and spam."""
    eng = get_engine()
    res = eng.moderate(req.text)
    return ModerationResponse(
        is_flagged=res.is_flagged,
        flagged_categories=res.flagged_categories,
        category_scores=res.category_scores,
        highest_severity_category=res.highest_severity_category,
        explanation=res.explanation
    )


@app.post("/v1/pii", response_model=PIIResponse, tags=["Privacy"])
async def detect_and_redact_pii(req: PIIRequest):
    """Detect sensitive PII entities (phones, emails, UPI, PAN, Aadhaar) and apply redaction."""
    eng = get_engine()
    spans = eng.detect_pii(req.text)
    redaction = eng.redact_pii(req.text, redact_names=req.redact_names, spans=spans)
    
    entity_items = [
        PIIEntityItem(
            type=s.entity_type,
            start=s.start,
            end=s.end,
            text=s.text,
            confidence=s.confidence,
            source=s.source,
            utf16_start=s.utf16_start,
            utf16_end=s.utf16_end
        )
        for s in spans
    ]

    return PIIResponse(
        entities=entity_items,
        has_pii=len(spans) > 0,
        redacted_text=redaction.redacted_text,
        entity_counts=redaction.entity_counts
    )


@app.post("/v1/normalize", response_model=NormalizeResponse, tags=["Normalization"])
async def normalize_text(req: NormalizeRequest):
    """Normalize code-mixed text across minimal, english, and native_script modes."""
    eng = get_engine()
    res = eng.normalize(req.text, mode=req.mode, target_language=req.target_language)
    return NormalizeResponse(
        source_text=res.source_text,
        normalized_text=res.normalized_text,
        mode=res.mode,
        target_language=res.target_language,
        preserved_entities=res.preserved_entities,
        preservation_intact=res.preservation_intact,
        warnings=res.warnings
    )


@app.post("/v1/analyze", response_model=AnalyzeResponse, tags=["Unified"])
async def analyze(req: AnalyzeRequest):
    """Unified multi-task analysis pipeline."""
    eng = get_engine()
    res = eng.analyze(
        req.text,
        tasks=req.tasks,
        normalization_mode=req.normalization_mode,
        target_language=req.target_language,
        redact_names=req.redact_names,
        input_id=req.input_id
    )
    return AnalyzeResponse(**res.to_dict())


@app.post("/v1/batch", response_model=BatchAnalyzeResponse, tags=["Unified"])
async def batch_analyze(req: BatchAnalyzeRequest):
    """High-throughput batched multi-task analysis."""
    t0 = time.perf_counter()
    eng = get_engine()
    results = eng.analyze_batch(
        req.texts,
        tasks=req.tasks,
        normalization_mode=req.normalization_mode,
        target_language=req.target_language,
        redact_names=req.redact_names
    )
    batch_lat = (time.perf_counter() - t0) * 1000.0

    return BatchAnalyzeResponse(
        results=[AnalyzeResponse(**r.to_dict()) for r in results],
        total_processed=len(results),
        batch_latency_ms=round(batch_lat, 2)
    )


@app.post("/v1/proxy/anonymize", response_model=ProxyAnonymizeResponse, tags=["Privacy Proxy"])
async def anonymize_for_llm(req: ProxyAnonymizeRequest):
    """Sanitize customer text for LLM consumption with reversible numbered entity tokens."""
    gw = get_proxy()
    session = gw.anonymize(req.text, redact_names=req.redact_names, session_id=req.session_id)
    
    entity_items = [
        PIIEntityItem(
            type=s.entity_type,
            start=s.start,
            end=s.end,
            text=s.text,
            confidence=s.confidence,
            source=s.source,
            utf16_start=s.utf16_start,
            utf16_end=s.utf16_end
        )
        for s in session.detected_spans
    ]
    
    counts: Dict[str, int] = {}
    for s in session.detected_spans:
        counts[s.entity_type] = counts.get(s.entity_type, 0) + 1

    return ProxyAnonymizeResponse(
        session_id=session.session_id,
        original_text=session.original_text,
        sanitized_text=session.sanitized_text,
        token_mapping=session.token_to_entity,
        detected_entities=entity_items,
        entity_counts=counts
    )


@app.post("/v1/proxy/deanonymize", response_model=ProxyDeanonymizeResponse, tags=["Privacy Proxy"])
async def deanonymize_from_llm(req: ProxyDeanonymizeRequest):
    """Re-hydrate text returned by LLM with authorized sensitive entities."""
    gw = get_proxy()
    mapping = req.token_mapping
    if not mapping and req.session_id:
        sess = gw.get_session(req.session_id)
        if not sess:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Privacy session '{req.session_id}' not found or expired."
            )
        mapping = sess.token_to_entity

    if not mapping:
        return ProxyDeanonymizeResponse(
            rehydrated_text=req.text_from_llm,
            tokens_replaced=0
        )

    rehydrated = gw.deanonymize(req.text_from_llm, mapping)
    replaced_count = sum(1 for token in mapping.keys() if token in req.text_from_llm)

    return ProxyDeanonymizeResponse(
        rehydrated_text=rehydrated,
        tokens_replaced=replaced_count
    )
