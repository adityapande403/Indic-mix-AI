"""Tests for Text Normalization and Entity Preservation."""

import pytest
from indicmix.normalization import TextNormalizer, extract_critical_entities, verify_entity_preservation


@pytest.fixture
def normalizer():
    return TextNormalizer(strict_entity_preservation=True)


def test_mode_a_minimal_normalization(normalizer):
    text = "bhaiii mera payment deduct ho gaya plzzzz help"
    res = normalizer.normalize(text, mode="minimal")
    # Repeated characters reduced
    assert "plzzzz" not in res.normalized_text
    assert "bhaiii" not in res.normalized_text
    assert "payment" in res.normalized_text
    assert "deduct" in res.normalized_text


def test_mode_b_english_normalization(normalizer):
    text = "bhai payment deduct ho gaya par order confirm nahi dikha raha"
    res = normalizer.normalize(text, mode="english", target_language="en")
    assert "payment" in res.normalized_text.lower()
    assert "order" in res.normalized_text.lower()
    assert "deducted" in res.normalized_text.lower() or "deduct" in res.normalized_text.lower()


def test_mode_c_native_devanagari_transliteration(normalizer):
    text = "bhai payment deduct ho gaya"
    res = normalizer.normalize(text, mode="native_script", source_language="hi_en_roman")
    assert "भाई" in res.normalized_text
    assert "पेमेंट" in res.normalized_text


def test_mode_c_native_telugu_transliteration(normalizer):
    text = "naaku order deliver avvaledu"
    res = normalizer.normalize(text, mode="native_script", source_language="te_en_roman")
    assert "నాకు" in res.normalized_text
    assert "ఆర్డర్" in res.normalized_text


def test_mode_c_native_bengali_transliteration(normalizer):
    text = "taka kete geche kintu order confirm hoyni"
    res = normalizer.normalize(text, mode="native_script", source_language="bn_en_roman")
    assert "টাকা" in res.normalized_text
    assert "অর্ডার" in res.normalized_text


def test_strict_entity_preservation(normalizer):
    text = "mera order ORD-998811 ka refund ₹1500 kab aayega"
    res = normalizer.normalize(text, mode="english", target_language="en")
    assert "ORD-998811" in res.normalized_text or "998811" in res.normalized_text
    assert "₹1500" in res.normalized_text or "1500" in res.normalized_text
    assert res.preservation_intact is True


def test_polarity_preservation_negation_intact(normalizer):
    # Negated input: "payment receive nahi hua" must remain negated
    text = "mera payment receive nahi hua"
    res = normalizer.normalize(text, mode="english", target_language="en")
    # Must preserve negation (e.g. "not", "nahi", etc.)
    from indicmix.normalization import check_negation_polarity
    assert check_negation_polarity(res.normalized_text) is True


def test_polarity_inversion_detection():
    from indicmix.normalization import verify_polarity_preservation
    # Source is negative, target is affirmative -> inversion detected
    is_intact, warnings = verify_polarity_preservation(
        "mera payment nahi aaya",
        "My payment has arrived successfully."
    )
    assert is_intact is False
    assert any("Polarity inversion" in w for w in warnings)
