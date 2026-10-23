"""Intent Classification Engine for Indic and Code-Mixed text.

Supports configurable intent taxonomies, probability calibration,
unknown-intent abstention thresholds, and high-throughput batch inference.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .calibration import TemperatureScaler


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: str
    confidence: float
    all_scores: Dict[str, float]
    is_abstained: bool
    abstention_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
            "is_abstained": self.is_abstained,
            "abstention_reason": self.abstention_reason
        }


DEFAULT_INTENT_TAXONOMY = [
    "greeting",
    "order_status",
    "payment_issue",
    "refund_request",
    "cancellation_request",
    "account_access",
    "delivery_issue",
    "technical_support",
    "complaint",
    "product_question",
    "subscription_issue",
    "unknown",
    "other"
]

# Robust semantic indicator keywords across Romanized Indic (Hindi, Telugu, Bengali) and English
INTENT_LEXICON_PRIORS = {
    "greeting": [
        "hi", "hello", "hey", "namaste", "namaskar", "vanakkam", "pranam", "good morning",
        "good afternoon", "good evening", "kaise ho", "ela unnaru", "kemon achen", "kemon acho",
        "नमस्ते", "नमस्कार", "प्रणाम", "నమస్కారం", "హలో", "নমস্কার", "হ্যালো"
    ],
    "order_status": [
        "order status", "status", "track", "tracking", "kaha hai order", "order kab aayega",
        "track order", "kab tak aayega", "kaha pahucha", "order deliver nahi hua", "ekkada undi",
        "eppudu vastundi", "kobe ashbe", "kothay ache", "payment ho gaya kya", "payment hua kya",
        "status kya hai", "check status", "order confirm hua kya", "order status kya hai",
        "कहाँ पहुँचा", "कहाँ है", "आर्डर कहाँ", "ऑर्डर कहाँ", "कब आएगा", "कहाँ",
        "ఎక్కడ ఉంది", "ఆర్డర్ ఎక్కడ", "ఎప్పుడు వస్తుంది",
        "কোথায় আছে", "কবে আসবে", "ট্র্যাকিং", "কোথায়"
    ],
    "payment_issue": [
        "payment deduct", "payment cut", "paise kat gaye", "debited but", "money cut",
        "dabbu cut", "taka kete", "payment fail", "failed transaction", "upi failed",
        "card issue", "deducted", "debited", "payment nahi aaya", "payment nahi mila",
        "paise nahi aaye", "kat gya", "pending dikha raha", "payment atka",
        "पेमेंट कट", "कट गया है", "पैसे कट", "कट गए", "पेमेंट",
        "డబ్బులు కట్", "కట్ అయ్యాయి",
        "টাকা কেটে", "পেমেন্ট"
    ],
    "refund_request": [
        "refund", "paise wapas", "refund kab aayega", "money back", "return money",
        "dabbu wapas", "taka pabo", "refund status", "refund processing", "amount refund",
        "refund chahiye", "wapas chahiye", "refund initiate", "credit nahi hua",
        "रिफंड", "पैसे वापस", "वापस करें",
        "రీఫండ్", "రీఫండ్ డబ్బు",
        "রিফান্ড", "টাকা ফেরত", "ফেরত"
    ],
    "cancellation_request": [
        "cancel", "cancelling", "cancel order", "cancel my order", "cancel karna",
        "cancellation", "order cancel", "cancel cheyandi", "cancel korte chai",
        "dont want order", "order radd", "payment cancel", "cancel kar do", "cancel payment",
        "cancel karo", "radd kare",
        "कैंसल", "रद्द", "रद्द करें",
        "రద్దు", "ఆర్డర్ రద్దు",
        "ক্যানসেল", "বাতিল"
    ],
    "account_access": [
        "login", "password", "otp", "cannot login", "account blocked", "reset password",
        "login avvatledu", "login hocche na", "account access", "sign in",
        "लॉगिन", "ओटीपी", "पासवर्ड",
        "లాగిన్", "ఓటీపీ",
        "লগইন", "ওটিপি"
    ],
    "delivery_issue": [
        "wrong item", "late delivery", "delivery boy", "not delivered", "damaged item",
        "package torn", "deliver avvaledu", "deliver hoyni", "delay", "damaged product",
        "bhul jinish", "deliver hoyeche", "damaged", "broken", "wrong product", "packet",
        "डिलीवर", "डिलीवरी", "गलत सामान", "खराब सामान",
        "డెలివరీ", "డెలివర్ కాలేదు", "పాడైపోయింది",
        "ডেলিভারি", "ডেলিভার হয়নি", "ভুল জিনিস", "ভাঙা"
    ],
    "technical_support": [
        "app crash", "not working", "bug", "error 500", "screen blank", "kaam nahi kar raha",
        "app open nahi ho raha", "server issue", "technical fault", "payment karna hai",
        "payment kaise kare", "how to pay", "pay nahi ho raha", "checkout error", "payment option",
        "ऐप", "क्रैश", "काम नहीं कर रहा", "ఓపెన్ అవ్వట్లేదు", "অ্যাপ ওপেন"
    ],
    "complaint": [
        "worst service", "very bad", "fraud", "complaint", "cheating", "gussa", "disappointed",
        "bekar", "kharab", "chor", "consumer court",
        "शिकायत", "बेकार सर्विस", "खराब सर्विस",
        "ఫిర్యాదు", "చెత్త సర్వీస్",
        "অভিযোগ", "বাজে সার্ভিস"
    ],
    "product_question": [
        "price kya hai", "size available", "warranty", "expiry date", "in stock", "specifications",
        "features", "discount kab", "guarantee", "return policy", "size chart", "size",
        "product", "dimensions", "kothay pabo", "replacement warranty",
        "वारंटी", "कीमत", "साइज",
        "ధర ఎంత", "వారంటీ",
        "দাম কত", "ওয়ারেন্টি", "সাইজ"
    ],
    "subscription_issue": [
        "subscription", "auto debit", "membership", "renew", "prime", "pro membership",
        "monthly plan", "cancel subscription", "auto renew", "membership auto renew",
        "सब्सक्रिप्शन", "सदस्यता", "ऑटो रिन्यू",
        "సబ్‌స్క్రిప్షన్", "సభ్యత్వం",
        "সাবস্ক্রিপশন"
    ],
    "unknown": [
        "mausam", "weather", "cinema", "brishti", "random", "barish", "repu"
    ]
}


class IntentClassifier:
    """Configurable Intent Classifier with uncertainty abstention and probability calibration."""

    def __init__(
        self,
        taxonomy: Optional[List[str]] = None,
        abstention_threshold: float = 0.45,
        temperature: float = 1.15,
        onnx_session: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.taxonomy = taxonomy or list(DEFAULT_INTENT_TAXONOMY)
        self.abstention_threshold = abstention_threshold
        self.calibrator = TemperatureScaler(temperature=temperature)
        self.onnx_session = onnx_session
        self.tokenizer = tokenizer

    def classify(self, text: str) -> IntentResult:
        """Classify intent for a single input text."""
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: List[str]) -> List[IntentResult]:
        """Perform batch intent classification."""
        if not texts:
            return []

        # 1. If ONNX model session is loaded and available, run fast vectorized ONNX inference
        if self.onnx_session is not None and self.tokenizer is not None:
            return self._run_onnx_inference(texts)

        # 2. Rule & lexical baseline with calibrated softmax
        results = []
        for text in texts:
            results.append(self._classify_rule_baseline(text))
        return results

    def _classify_rule_baseline(self, text: str) -> IntentResult:
        """Robust lexical & prior distribution classifier for baseline."""
        stripped = text.strip() if text else ""
        if not stripped:
            uniform_scores = {intent: 1.0 / len(self.taxonomy) for intent in self.taxonomy}
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                all_scores=uniform_scores,
                is_abstained=True,
                abstention_reason="Empty text input"
            )

        lower = stripped.lower()
        raw_scores: Dict[str, float] = {intent: 0.1 for intent in self.taxonomy}

        # Lexical evidence matching using word boundaries
        has_substantive_intent = False
        for intent, patterns in INTENT_LEXICON_PRIORS.items():
            if intent in raw_scores:
                for pat in patterns:
                    is_non_ascii = any(ord(c) > 127 for c in pat)
                    matched = (pat in lower) if is_non_ascii else bool(re.search(rf"\b{re.escape(pat)}\b", lower))
                    if matched:
                        words_in_pat = len(pat.split())
                        weight = 7.0 if words_in_pat >= 3 else (5.0 if words_in_pat == 2 else 3.5)
                        raw_scores[intent] += weight
                        if intent != "greeting":
                            has_substantive_intent = True

        # Actionable intent disambiguation rules:
        # 1. If subscription / membership renewal is mentioned:
        if re.search(r"\b(membership|auto renew|subscription|auto debit|monthly plan)\b", lower):
            raw_scores["subscription_issue"] += 8.0
            has_substantive_intent = True
        # 2. If delivery/wrong item is mentioned:
        elif re.search(r"\b(wrong item|bhul jinish|deliver hoyeche|damaged product|broken|damaged item)\b", lower):
            raw_scores["delivery_issue"] += 7.0
            has_substantive_intent = True
        # 3. If product specs / size chart is asked:
        elif re.search(r"\b(size chart|size|specification|warranty|kothay pabo)\b", lower):
            raw_scores["product_question"] += 7.0
            has_substantive_intent = True
        # 4. If cancellation action is requested:
        elif re.search(r"\b(cancel kar do|cancel karna|cancel my order|cancel payment|payment cancel|cancel order|radd karo|radd kar do)\b", lower):
            raw_scores["cancellation_request"] += 6.0
            has_substantive_intent = True
        # 5. If status inquiry ("ho gaya kya", "kaha pahucha", "status kya hai")
        elif re.search(r"\b(ho gaya kya|hua kya|kaha pahucha|status kya hai|check status|kaha tak pahucha)\b", lower):
            raw_scores["order_status"] += 6.0
            has_substantive_intent = True
        # 6. If intent to make a payment or support with paying ("payment karna hai", "how to pay")
        elif re.search(r"\b(payment karna hai|karna chahta hu|kaise kare|how to pay|payment option|pay nahi ho raha)\b", lower):
            raw_scores["technical_support"] += 6.0
            has_substantive_intent = True
        # 7. If payment deducted/missing ("nahi aaya", "kat gaye par", "deduct ho gaya")
        elif re.search(r"\b(nahi aaya|nhi aaya|kat gaye|kat gaya|deduct ho gaya|debited)\b", lower):
            raw_scores["payment_issue"] += 6.0
            has_substantive_intent = True

        # If a substantive business intent (payment, order, refund, cancel) is detected,
        # discount casual greeting so "bhai payment deduct..." classifies as payment_issue
        if has_substantive_intent and "greeting" in raw_scores:
            raw_scores["greeting"] = min(raw_scores["greeting"], 0.5)

        # Convert to numpy array in order of taxonomy
        score_arr = np.array([raw_scores[intent] for intent in self.taxonomy], dtype=np.float32)
        
        # Softmax
        exp_arr = np.exp(score_arr - np.max(score_arr))
        probs = exp_arr / np.sum(exp_arr)

        # Apply temperature calibration
        calibrated_probs = self.calibrator.calibrate_probabilities(probs)

        # Map back to dict
        all_scores = {intent: float(calibrated_probs[i]) for i, intent in enumerate(self.taxonomy)}
        
        top_idx = int(np.argmax(calibrated_probs))
        top_intent = self.taxonomy[top_idx]
        top_confidence = float(calibrated_probs[top_idx])

        # Abstention check
        is_abstained = False
        abstention_reason = None

        if top_confidence < self.abstention_threshold:
            is_abstained = True
            abstention_reason = (
                f"Confidence ({top_confidence:.3f}) is below abstention threshold ({self.abstention_threshold:.3f})"
            )
            top_intent = "unknown"

        return IntentResult(
            intent=top_intent,
            confidence=top_confidence,
            all_scores=all_scores,
            is_abstained=is_abstained,
            abstention_reason=abstention_reason
        )

    def _run_onnx_inference(self, texts: List[str]) -> List[IntentResult]:
        """Execute ONNX Runtime batch inference."""
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        ort_outputs = self.onnx_session.run(None, ort_inputs)
        logits = ort_outputs[0]

        # Calibrate logits with temperature
        calibrated_logits = self.calibrator.calibrate_logits(logits)
        exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        results = []
        for i, text in enumerate(texts):
            p = probs[i]
            top_idx = int(np.argmax(p))
            top_intent = self.taxonomy[top_idx] if top_idx < len(self.taxonomy) else "unknown"
            top_conf = float(p[top_idx])

            all_scores = {
                self.taxonomy[j]: float(p[j])
                for j in range(min(len(p), len(self.taxonomy)))
            }

            is_abstained = False
            abstention_reason = None
            if top_conf < self.abstention_threshold:
                is_abstained = True
                abstention_reason = f"Confidence ({top_conf:.3f}) below threshold ({self.abstention_threshold:.3f})"
                top_intent = "unknown"

            results.append(IntentResult(
                intent=top_intent,
                confidence=top_conf,
                all_scores=all_scores,
                is_abstained=is_abstained,
                abstention_reason=abstention_reason
            ))

        return results
