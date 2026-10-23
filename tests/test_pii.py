"""Tests for PII Detection and Redaction."""

import pytest
from indicmix.pii_detection import HybridPIIDetector
from indicmix.redaction import PIIRedactor


@pytest.fixture
def detector():
    return HybridPIIDetector(enable_model=True)


@pytest.fixture
def redactor():
    return PIIRedactor()


def test_phone_detection(detector, redactor):
    text = "mera mobile number 9876543210 hai call karo"
    spans = detector.detect(text)
    phone_spans = [s for s in spans if s.entity_type == "PHONE"]
    assert len(phone_spans) == 1
    assert phone_spans[0].text == "9876543210"

    res = redactor.redact(text, spans)
    assert "[PHONE]" in res.redacted_text
    assert "9876543210" not in res.redacted_text


def test_email_and_upi_detection(detector, redactor):
    text = "send invoice to test.user@example.com and refund to rahul@okhdfcbank"
    spans = detector.detect(text)
    types = [s.entity_type for s in spans]
    assert "EMAIL" in types
    assert "UPI_ID" in types

    res = redactor.redact(text, spans)
    assert "[EMAIL]" in res.redacted_text
    assert "[UPI_ID]" in res.redacted_text


def test_pan_and_aadhaar_detection(detector, redactor):
    text = "My PAN is ABCDE1234F and Aadhaar is 2345 6789 0123"
    spans = detector.detect(text)
    types = [s.entity_type for s in spans]
    assert "PAN" in types
    assert "AADHAAR" in types


def test_order_and_name_redaction_options(detector, redactor):
    text = "Call Rahul at 9876543210 about order ORD-123456"
    spans = detector.detect(text)
    
    # Redact everything including name
    res_all = redactor.redact(text, spans, redact_names=True)
    assert "[PHONE]" in res_all.redacted_text
    assert "[ORDER_ID]" in res_all.redacted_text
    assert "[NAME]" in res_all.redacted_text

    # Redact everything EXCEPT name
    res_keep_name = redactor.redact(text, spans, redact_names=False)
    assert "[PHONE]" in res_keep_name.redacted_text
    assert "[ORDER_ID]" in res_keep_name.redacted_text
    assert "Rahul" in res_keep_name.redacted_text


def test_non_pii_numbers_not_falsely_flagged_as_phone(detector):
    # Prices and amounts must not be labeled as phone numbers
    text = "total amount is Rs 500 for 2 items"
    spans = detector.detect(text)
    phone_spans = [s for s in spans if s.entity_type == "PHONE"]
    assert len(phone_spans) == 0
