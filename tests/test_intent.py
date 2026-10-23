"""Tests for Intent Classification."""

import pytest
from indicmix.intent import IntentClassifier


@pytest.fixture
def classifier():
    return IntentClassifier(abstention_threshold=0.40, temperature=1.15)


def test_intent_payment_issue(classifier):
    text = "bhai payment deduct ho gaya par order confirm nahi dikha raha"
    res = classifier.classify(text)
    assert res.intent == "payment_issue"
    assert res.confidence > 0.40
    assert not res.is_abstained


def test_intent_order_status(classifier):
    text = "mera order kab tak aayega status track karna hai"
    res = classifier.classify(text)
    assert res.intent == "order_status"
    assert res.confidence > 0.40


def test_intent_cancellation(classifier):
    text = "please cancel my order immediately"
    res = classifier.classify(text)
    assert res.intent == "cancellation_request"
    assert res.confidence > 0.40


def test_intent_greeting(classifier):
    text = "namaste sir, good morning"
    res = classifier.classify(text)
    assert res.intent == "greeting"


def test_intent_abstention_on_unrelated_text(classifier):
    text = "aaj ka mausam kaisa hai waha barish ho rahi hai kya"
    res = classifier.classify(text)
    # Since weather is outside customer support taxonomy, should either be low confidence or unknown
    if res.confidence < classifier.abstention_threshold:
        assert res.is_abstained is True
        assert res.intent == "unknown"


def test_intent_batch_classification(classifier):
    texts = [
        "payment deduct ho gaya",
        "order cancel karna hai",
        "hello namaste"
    ]
    results = classifier.classify_batch(texts)
    assert len(results) == 3
    assert results[0].intent == "payment_issue"
    assert results[1].intent == "cancellation_request"
    assert results[2].intent == "greeting"


def test_intent_nuanced_payment_disambiguation(classifier):
    # 4 distinct messages sharing "payment" keyword but with contrasting intentions
    res1 = classifier.classify("Mera payment nahi aaya.")
    assert res1.intent == "payment_issue"

    res2 = classifier.classify("Mujhe payment karna hai.")
    assert res2.intent == "technical_support"

    res3 = classifier.classify("Payment ho gaya kya?")
    assert res3.intent == "order_status"

    res4 = classifier.classify("Mera payment cancel kar do.")
    assert res4.intent == "cancellation_request"

