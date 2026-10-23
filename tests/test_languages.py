"""Tests for Language Identification and Text Profiling."""

import pytest
from indicmix.language_detection import LanguageIdentifier
from indicmix.text_profiles import (
    analyze_text_profile,
    codepoint_to_utf16_offset,
    utf16_to_codepoint_offset
)


@pytest.fixture
def identifier():
    return LanguageIdentifier(abstention_threshold=0.35, min_length_for_confidence=4)


def test_hinglish_detection(identifier):
    text = "bhai payment deduct ho gaya par order confirm nahi dikha raha"
    res = identifier.identify(text)
    assert res.primary_language == "hi_en_roman"
    assert res.is_romanized is True
    assert res.confidence > 0.40
    assert not res.is_uncertain


def test_telgish_detection(identifier):
    text = "naaku order deliver avvaledu anna dabbu cut ayindi"
    res = identifier.identify(text)
    assert res.primary_language == "te_en_roman"
    assert res.is_romanized is True
    assert res.confidence > 0.40


def test_benglish_detection(identifier):
    text = "dada amar order ekhono elo na taka kete geche"
    res = identifier.identify(text)
    assert res.primary_language == "bn_en_roman"
    assert res.is_romanized is True
    assert res.confidence > 0.40


def test_native_devanagari(identifier):
    text = "नमस्ते भाई, मेरा पेमेंट कट गया है लेकिन आर्डर नहीं मिला"
    res = identifier.identify(text)
    assert res.primary_language == "hi_devanagari"
    assert res.confidence > 0.80
    assert not res.is_romanized


def test_native_telugu(identifier):
    text = "నమస్కారం అండి, నా ఆర్డర్ రాలేదు"
    res = identifier.identify(text)
    assert res.primary_language == "te_telugu"
    assert res.confidence > 0.80


def test_native_bengali(identifier):
    text = "নমস্কার দাদা, আমার অর্ডার এখনো আসেনি"
    res = identifier.identify(text)
    assert res.primary_language == "bn_bengali"
    assert res.confidence > 0.80


def test_short_and_empty_text_abstention(identifier):
    res_empty = identifier.identify("")
    assert res_empty.primary_language == "unknown"
    assert res_empty.is_uncertain is True
    assert res_empty.abstention_reason is not None

    res_short = identifier.identify("hi")
    assert res_short.is_uncertain is True


def test_utf16_codepoint_conversion():
    # String with astral symbol (emoji requiring surrogate pair in UTF-16)
    text = "Hello 😀 India"
    # '😀' is at code-point 6, occupies 1 code point, but 2 UTF-16 code units
    cp_idx = 8 # Index of 'I' in "India"
    u16_idx = codepoint_to_utf16_offset(text, cp_idx)
    # "Hello " is 6 units, 😀 is 2 units, " " is 1 unit -> total 9 units
    assert u16_idx == 9
    
    # Reverse conversion
    converted_cp = utf16_to_codepoint_offset(text, u16_idx)
    assert converted_cp == cp_idx
