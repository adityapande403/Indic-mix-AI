"""Tests for Enterprise LLM Privacy Proxy and Re-hydration Gateway."""

import pytest
from fastapi.testclient import TestClient

from indicmix.proxy import EnterprisePrivacyProxy, AnonymizationSession
from api.app import app


@pytest.fixture
def proxy():
    return EnterprisePrivacyProxy()


@pytest.fixture
def client():
    return TestClient(app)


def test_proxy_anonymize_indexed_tokens(proxy):
    text = "Please refund ₹1500 to Rahul. Contact me on 9876543210 or email rahul@example.com."
    session = proxy.anonymize(text, redact_names=True)

    # 1. PII should be replaced by indexed tokens
    assert "[PHONE_1]" in session.sanitized_text or "[PHONE" in session.sanitized_text
    assert "[NAME_1]" in session.sanitized_text or "[NAME" in session.sanitized_text
    assert "[EMAIL_1]" in session.sanitized_text or "[EMAIL" in session.sanitized_text

    # 2. Raw sensitive values should NOT be in sanitized text
    assert "9876543210" not in session.sanitized_text
    assert "rahul@example.com" not in session.sanitized_text
    assert "Rahul" not in session.sanitized_text

    # 3. Transaction amount must remain intact!
    assert "₹1500" in session.sanitized_text


def test_proxy_deanonymize_llm_response(proxy):
    text = "Please refund ₹1500 to Rahul. Contact me on 9876543210."
    session = proxy.anonymize(text, redact_names=True)

    # Simulate an LLM completing a task using the sanitized prompt
    # LLM output: "I have initiated the ₹1500 refund for [NAME_1]. Confirmation sent to [PHONE_1]."
    name_token = session.entity_to_token.get("Rahul", "[NAME_1]")
    phone_token = session.entity_to_token.get("9876543210", "[PHONE_1]")

    llm_output = f"Refund of ₹1500 for {name_token} has been processed. SMS sent to {phone_token}."
    
    # De-anonymize
    rehydrated = session.deanonymize(llm_output)
    assert "Rahul" in rehydrated
    assert "9876543210" in rehydrated
    assert name_token not in rehydrated
    assert phone_token not in rehydrated


def test_proxy_name_redaction_toggle(proxy):
    text = "Rahul wants update on phone 9876543210."
    # Disabling name redaction allows person name to pass through for personalized responses
    session = proxy.anonymize(text, redact_names=False)
    assert "Rahul" in session.sanitized_text
    assert "9876543210" not in session.sanitized_text


def test_proxy_api_anonymize_and_deanonymize_endpoints(client):
    payload = {
        "text": "Bhai mera ₹2000 ka payment kat gaya, call Suresh at 9876543210 or UPI suresh@okhdfcbank",
        "redact_names": True
    }
    # 1. Anonymize
    anon_resp = client.post("/v1/proxy/anonymize", json=payload)
    assert anon_resp.status_code == 200
    data = anon_resp.json()
    assert "session_id" in data
    assert "sanitized_text" in data
    assert "token_mapping" in data
    assert "₹2000" in data["sanitized_text"]
    assert "9876543210" not in data["sanitized_text"]
    assert "suresh@okhdfcbank" not in data["sanitized_text"]

    sid = data["session_id"]
    sanitized = data["sanitized_text"]

    # 2. De-anonymize using session_id
    llm_simulated_reply = f"Support response: Checked your account. We will contact customer regarding {sanitized}."
    deanon_resp = client.post("/v1/proxy/deanonymize", json={
        "text_from_llm": llm_simulated_reply,
        "session_id": sid
    })
    assert deanon_resp.status_code == 200
    deanon_data = deanon_resp.json()
    assert "rehydrated_text" in deanon_data
    assert "9876543210" in deanon_data["rehydrated_text"]
    assert "suresh@okhdfcbank" in deanon_data["rehydrated_text"]
