"""Tests for Toxicity and Content Moderation."""

import pytest
from indicmix.moderation import ModerationEngine


@pytest.fixture
def moderation():
    return ModerationEngine()


def test_clean_informal_text_not_flagged(moderation):
    clean_texts = [
        "bhai payment deduct ho gaya par order nahi dikha raha",
        "arey yaar bohot deri ho gayi please help",
        "anna fast delivery cheyandi",
        "dada order ta ekhono elo na ektu dekhun"
    ]
    for text in clean_texts:
        res = moderation.moderate(text)
        assert res.is_flagged is False, f"False positive flag on harmless text: {text}"
        assert len(res.flagged_categories) == 0


def test_direct_abuse_flagged(moderation):
    abusive_text = "kya chutiya service hai tumhari saale chor"
    res = moderation.moderate(abusive_text)
    assert res.is_flagged is True
    assert "direct_abuse" in res.flagged_categories or "profanity" in res.flagged_categories


def test_threat_flagged(moderation):
    threat_text = "tujhe office aakar jaan se mar dunga samjha"
    res = moderation.moderate(threat_text)
    assert res.is_flagged is True
    assert "threat" in res.flagged_categories


def test_spam_flagged(moderation):
    spam_text = "work from home earn 50000 daily click here to win lottery"
    res = moderation.moderate(spam_text)
    assert res.is_flagged is True
    assert "spam" in res.flagged_categories


def test_sexual_harassment_flagged(moderation):
    harassment_text = "send nudes on whatsapp baby"
    res = moderation.moderate(harassment_text)
    assert res.is_flagged is True
    assert "sexual_harassment" in res.flagged_categories or "harassment" in res.flagged_categories


def test_empty_input(moderation):
    res = moderation.moderate("")
    assert res.is_flagged is False
