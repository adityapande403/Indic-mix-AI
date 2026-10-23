"""API Integration and Unit Tests using FastAPI TestClient."""

import os
import sys
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "components" in data


def test_model_info_endpoint(client):
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "version" in data
    assert "execution_provider" in data


def test_intent_endpoint(client):
    payload = {"text": "bhai payment deduct ho gaya par order confirm nahi dikha raha"}
    res = client.post("/v1/intent", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "payment_issue"
    assert "confidence" in data
    assert "all_scores" in data


def test_moderate_endpoint(client):
    payload = {"text": "kya chutiya service hai tumhari saale"}
    res = client.post("/v1/moderate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["is_flagged"] is True
    assert "direct_abuse" in data["flagged_categories"] or "profanity" in data["flagged_categories"]


def test_pii_endpoint(client):
    payload = {"text": "Call Rahul at 9876543210 about order ORD-123456", "redact_names": True}
    res = client.post("/v1/pii", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["has_pii"] is True
    assert "[PHONE]" in data["redacted_text"]
    assert "[ORDER_ID]" in data["redacted_text"]


def test_normalize_endpoint(client):
    payload = {
        "text": "mera order abhi tak deliver nhi hua",
        "mode": "english",
        "target_language": "en"
    }
    res = client.post("/v1/normalize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["preservation_intact"] is True
    assert "delivered" in data["normalized_text"].lower()


def test_analyze_endpoint(client):
    payload = {
        "text": "bhai payment deduct ho gaya par order confirm nahi dikha raha",
        "tasks": ["language", "intent", "moderation", "pii", "normalization"]
    }
    res = client.post("/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["language_profile"]["primary_language"] == "hi_en_roman"
    assert data["intent"]["intent"] == "payment_issue"
    assert "latency_ms" in data


def test_batch_analyze_endpoint(client):
    payload = {
        "texts": [
            "payment deduct ho gaya",
            "mera order kab aayega",
            "thank you very much"
        ]
    }
    res = client.post("/v1/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 3
    assert len(data["results"]) == 3


def test_malformed_request_validation(client):
    # Empty text violates min_length=1
    res = client.post("/v1/intent", json={"text": ""})
    assert res.status_code == 422
