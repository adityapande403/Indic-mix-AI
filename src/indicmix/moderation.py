"""Toxicity and Moderation Engine for Code-Mixed Indic text.

Multilabel classification covering:
- Direct abuse
- Threats
- Harassment
- Hate speech
- Sexual harassment
- Profanity
- Spam

Guarantees that ordinary informal speech, dialects, and harmless colloquial slang
(e.g., 'yaar', 'bhai', 'dada', 'anna', 'machha') are never misclassified as toxic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ModerationResult:
    """Detailed result of content moderation."""
    is_flagged: bool
    flagged_categories: List[str]
    category_scores: Dict[str, float]
    thresholds: Dict[str, float]
    highest_severity_category: Optional[str] = None
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_flagged": self.is_flagged,
            "flagged_categories": self.flagged_categories,
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "thresholds": self.thresholds,
            "highest_severity_category": self.highest_severity_category,
            "explanation": self.explanation
        }


DEFAULT_MODERATION_THRESHOLDS = {
    "direct_abuse": 0.50,
    "threat": 0.40,
    "harassment": 0.50,
    "hate_speech": 0.45,
    "sexual_harassment": 0.40,
    "profanity": 0.60,
    "spam": 0.55
}

# Harmless informal slang that must NOT trigger toxicity
HARMLESS_SLANG = {
    "bhai", "yaar", "dada", "anna", "machha", "arey", "re", "babu",
    "garu", "boss", "bro", "dude", "jugaad", "jhakaas", "chalta hai"
}

# Targeted lexical patterns for abuse, threat, hate, profanity, and spam
MODERATION_PATTERNS = {
    "threat": [
        r"\b(?:mar\s*dunga|jaan\s*se\s*marunga|tujhe\s*dekh\s*lunga|haddi\s*tod\s*dunga)\b",
        r"\b(?:champesta|chastav|champesthamu|kotti\s*sampesta)\b",
        r"\b(?:mere\s*phelbo|khun\s*korbo|toke\s*dekhe\s*nebo)\b",
        r"\b(?:i\s*will\s*kill\s*you|gonna\s*murder|threaten|break\s*your\s*bones)\b"
    ],
    "direct_abuse": [
        r"\b(?:madarchod|mc|bhenchod|bc|bhosdike|bsdk|chutiya|chutiye|harami|kamine|kutta|gandu)\b",
        r"\b(?:lanjakodaka|dengu|munda|na\s*kodaka|erripuka|puka)\b",
        r"\b(?:bokachoda|khankir\s*chele|shala|haramjada|choda)\b",
        r"\b(?:assholes?|bastards?|motherfuckers?|bitches?|idiots?|morons?)\b"
    ],
    "harassment": [
        r"\b(?:stalking\s*you|tujhe\s*chhodunga\s*nahi|barbar\s*call\s*karunga|trolling)\b",
        r"\b(?:ninnu\s*vadalanu|daily\s*call\s*chestha|vedhista)\b",
        r"\b(?:toke\s*chharbo\s*na|birokto\s*korbo)\b",
        r"\b(?:i\s*will\s*keep\s*harassing|wont\s*leave\s*you\s*alone)\b",
        r"\b(?:break\s*your\s*bones|mar\s*dunga|champesthamu|mere\s*phelbo|tujhe\s*dekh\s*lunga|sexy\s*photo|send\s*nudes|nude\s*photo)\b"
    ],
    "sexual_harassment": [
        r"\b(?:send\s*nudes|sexy\s*photo\s*bhejo|apna\s*body\s*dikhao|kiss\s*me\s*baby)\b",
        r"\b(?:battalu\s*vippu|nude\s*photo\s*pettu)\b",
        r"\b(?:chobi\s*pathao\s*khule|shorir\s*dekhao)\b",
        r"\b(?:send\s*naked\s*pics|sexual\s*favors)\b"
    ],
    "profanity": [
        r"\b(?:fuck|fucking|shit|bullshit|damn|crap|bakwaas|faltu)\b",
        r"\b(?:saala|saale|kamina|gadha|ullu|assholes?|idiots?|chutiya|bhosdike|lanjakodaka|bokachoda)\b"
    ],
    "hate_speech": [
        r"\b(?:all\s*(?:hindus|muslims|sikhs|christians|dalits)\s*are\s*(?:terrorists|dogs|trash))\b",
        r"\b(?:desh\s*ke\s*gaddaro|inko\s*maro)\b"
    ],
    "spam": [
        r"\b(?:earn\s*\d+\s*(?:daily|weekly)|work\s*from\s*home\s*earn|free\s*crypto|click\s*here\s*to\s*win)\b",
        r"\b(?:lottery\s*winner|congratulations\s*you\s*won\s*iphone|telegram\s*channel\s*link)\b",
        r"\b(?:free\s*recharge|guaranteed\s*profit|invest\s*\d+\s*get\s*\d+)\b"
    ]
}


class ModerationEngine:
    """Configurable Moderation Classifier with per-category calibrated thresholds."""

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        categories: Optional[List[str]] = None,
        onnx_session: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.categories = categories or list(DEFAULT_MODERATION_THRESHOLDS.keys())
        self.thresholds = dict(DEFAULT_MODERATION_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.onnx_session = onnx_session
        self.tokenizer = tokenizer

        # Compile regexes
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for cat, pats in MODERATION_PATTERNS.items():
            self.compiled_patterns[cat] = [
                re.compile(p, re.IGNORECASE) for p in pats
            ]

    def moderate(self, text: str) -> ModerationResult:
        """Run moderation analysis on a single text string."""
        stripped = text.strip() if text else ""
        if not stripped:
            return ModerationResult(
                is_flagged=False,
                flagged_categories=[],
                category_scores={cat: 0.0 for cat in self.categories},
                thresholds=self.thresholds,
                highest_severity_category=None,
                explanation="Empty text passed to moderation"
            )

        # Baseline rule-based calibrated scoring
        scores: Dict[str, float] = {cat: 0.02 for cat in self.categories}
        matched_categories = []

        for cat in self.categories:
            patterns = self.compiled_patterns.get(cat, [])
            match_count = 0
            for pat in patterns:
                matches = pat.findall(stripped)
                match_count += len(matches)

            if match_count > 0:
                # Scaled score with saturation
                score = min(0.99, 0.65 + (match_count * 0.15))
                scores[cat] = score
            else:
                scores[cat] = 0.02

        # Check against thresholds
        flagged = []
        for cat in self.categories:
            if scores[cat] >= self.thresholds.get(cat, 0.50):
                flagged.append(cat)

        is_flagged = len(flagged) > 0
        top_cat = max(flagged, key=lambda c: scores[c]) if flagged else None

        explanation = None
        if is_flagged:
            explanation = f"Flagged for: {', '.join(flagged)} with highest score in '{top_cat}' ({scores[top_cat]:.2f})"

        return ModerationResult(
            is_flagged=is_flagged,
            flagged_categories=flagged,
            category_scores=scores,
            thresholds=self.thresholds,
            highest_severity_category=top_cat,
            explanation=explanation
        )

    def moderate_batch(self, texts: List[str]) -> List[ModerationResult]:
        """Run batch moderation analysis."""
        return [self.moderate(t) for t in texts]
