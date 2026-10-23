"""Code-mixed Indic Language Identification Engine.

Combines Unicode script analysis, lexical signatures, and character n-gram statistical
models to accurately identify Romanized Indic languages (Hinglish, Telgish, Benglish),
native script Indic text, and English without falsely defaulting Latin text to English.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .text_profiles import analyze_text_profile, TextProfile


@dataclass
class LanguageProfileResult:
    """Detailed result of language identification."""
    primary_language: str
    confidence: float
    is_code_mixed: bool
    is_romanized: bool
    is_uncertain: bool
    scores: Dict[str, float]
    abstention_reason: Optional[str] = None
    detected_scripts: List[str] = field(default_factory=list)
    script_profile: Optional[TextProfile] = None


# Core lexical markers for Romanized Indic languages (lowercased)
ROMAN_HINDI_LEXICON = {
    # Pronouns & interrogatives
    "mera", "meri", "mere", "mujhe", "mujhko", "hum", "humara", "humari", "aap", "aapka",
    "aapko", "tum", "tumhara", "tumhe", "yeh", "ye", "woh", "wo", "kya", "kyun", "kyu",
    "kab", "kaha", "kahan", "kaise", "kisko", "kiska", "kaun", "kon",
    # Verbs and auxiliaries
    "hai", "hain", "ho", "hu", "hoon", "tha", "thi", "the", "gaya", "gayi", "gaye",
    "raha", "rahi", "rahe", "karo", "karna", "karne", "kar", "kiya", "diya", "liya",
    "dekh", "dekho", "batao", "aaya", "aayi", "aate", "sakta", "sakti", "sakte", "chahiye",
    "hoga", "hogi", "honge", "hua", "hui", "hue", "mil", "mila", "milega",
    # Particles, adverbs, conjunctions, slang
    "nahi", "nhi", "na", "mat", "bhi", "aur", "lekin", "par", "magar", "itna", "utna",
    "bohot", "bahut", "bhot", "accha", "achha", "theek", "thik", "sirf", "bhai", "yaar",
    "paise", "rupaye", "baat", "kuch", "sab", "ab", "abhi", "pehle", "baad", "jaldi"
}

ROMAN_TELUGU_LEXICON = {
    # Pronouns & interrogatives
    "nenu", "naaku", "naku", "meeru", "meeku", "atanu", "aame", "manamu", "idi", "adi",
    "evvaru", "evaru", "enti", "enduku", "endhuku", "ela", "eppudu", "ekkada", "em", "emi",
    # Verbs and auxiliaries
    "undi", "undhi", "undha", "ledu", "ledhu", "ledha", "chesanu", "chesaru", "cheyandi",
    "cheyyi", "raaledu", "raledu", "vacchesindi", "vachindi", "poyindi", "pettanu", "ivvandi",
    "kavali", "chudandi", "avutundi", "avvatledu", "unnaru", "unnadu", "unnam", "chesi",
    "chusa", "telusu", "raaledha",
    # Particles, adverbs, conjunctions, slang
    "chala", "chaala", "inka", "kuda", "kooda", "kadha", "ippudu", "appudu", "mari",
    "babu", "anna", "garu", "dabbu", "dabbulu", "roje", "telusa", "kani", "kaani", "aithe",
    "anta", "andaru", "konchem", "twaraga"
}

ROMAN_BENGALI_LEXICON = {
    # Pronouns & interrogatives
    "ami", "amake", "amar", "tumi", "tomake", "tomar", "apni", "apnake", "apnar",
    "she", "tara", "eita", "oita", "ke", "ki", "keno", "kobe", "kothay", "kivabe",
    # Verbs and auxiliaries
    "hoyeche", "hobe", "hoyni", "korechi", "korbo", "kora", "ache", "nei", "dekhun",
    "bolchi", "parlam", "gelo", "ashche", "jani", "janina", "dicche", "pelam", "hoy",
    # Particles, adverbs, conjunctions, slang
    "ektu", "shob", "khub", "aar", "kintu", "dada", "bhai", "taka", "bhalo", "thik",
    "shathe", "shobai", "kokhon", "ekhane", "othoba", "karon"
}

COMMON_ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
    "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
    "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give", "day", "most",
    "us", "order", "payment", "account", "support", "please", "help", "deliver",
    "delivery", "refund", "card", "failed", "deducted", "cancelled", "confirmed",
    "show", "showing", "issue", "money", "bank", "received", "status"
}


# Precomputed character trigram weights for Latin-script classification
# Characteristic character trigrams for Hindi, Telugu, Bengali transliteration vs English
TRIGRAM_WEIGHTS = {
    "hi_en_roman": {
        "hai": 2.5, "nah": 2.2, "ahi": 2.2, "gay": 2.0, "aya": 1.8, "mer": 1.9,
        "era": 1.8, "kar": 2.0, "kya": 2.2, "rah": 2.0, "aha": 1.7, "bha": 1.8,
        "hai": 2.5, "par": 1.6, "aay": 1.6, "tho": 1.5, "hot": 1.5, "ota": 1.5
    },
    "te_en_roman": {
        "und": 2.5, "ndi": 2.3, "led": 2.4, "edu": 2.2, "che": 2.1, "hes": 2.0,
        "cha": 1.8, "ala": 1.6, "aan": 1.8, "and": 1.5, "epp": 2.0, "ekk": 2.0,
        "end": 1.9, "dha": 1.8, "tva": 1.7, "poy": 2.1, "oyi": 2.0, "yin": 1.9
    },
    "bn_en_roman": {
        "hoy": 2.5, "yec": 2.2, "ech": 2.1, "che": 1.9, "kor": 2.2, "ore": 2.0,
        "rec": 1.9, "ama": 1.8, "mar": 1.7, "ekt": 2.3, "ktu": 2.2, "sho": 1.8,
        "hob": 2.1, "obe": 2.0, "bol": 1.9, "olc": 1.8, "lch": 1.8, "tak": 1.8
    },
    "en": {
        "the": 2.5, "ing": 2.2, "ion": 2.0, "and": 2.0, "tha": 1.8, "hat": 1.8,
        "ent": 1.8, "ter": 1.6, "ver": 1.6, "ord": 2.0, "rde": 2.0, "der": 2.0,
        "ple": 1.8, "eas": 1.8, "ase": 1.8, "pay": 2.0, "aym": 2.0, "men": 1.9
    }
}


def _extract_trigrams(text: str) -> List[str]:
    """Extract character trigrams from text."""
    clean = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = clean.split()
    trigrams = []
    for w in words:
        padded = f" {w} "
        for i in range(len(padded) - 2):
            trigrams.append(padded[i:i+3].strip())
    return [t for t in trigrams if len(t) >= 2]


class LanguageIdentifier:
    """Hybrid Language Identifier for Indic code-mixed and native text."""

    def __init__(self, abstention_threshold: float = 0.35, min_length_for_confidence: int = 4):
        self.abstention_threshold = abstention_threshold
        self.min_length_for_confidence = min_length_for_confidence

    def identify(self, text: str) -> LanguageProfileResult:
        """Identify language profile and code-mixing status of input text."""
        stripped = text.strip() if text else ""
        
        # 1. Edge case: Empty or near-empty text
        if not stripped:
            return LanguageProfileResult(
                primary_language="unknown",
                confidence=0.0,
                is_code_mixed=False,
                is_romanized=False,
                is_uncertain=True,
                scores={"unknown": 1.0},
                abstention_reason="Empty text input"
            )

        profile = analyze_text_profile(stripped)
        
        # 2. Native script checking (Devanagari, Telugu, Bengali)
        if profile.indic_script_ratio >= 0.40:
            dominant = profile.dominant_script
            is_code_mixed = profile.latin_ratio > 0.10
            
            if dominant == "Devanagari":
                conf = max(0.85, min(0.99, profile.indic_script_ratio))
                return LanguageProfileResult(
                    primary_language="hi_devanagari",
                    confidence=conf,
                    is_code_mixed=is_code_mixed,
                    is_romanized=False,
                    is_uncertain=False,
                    scores={"hi_devanagari": conf, "en": 1.0 - conf},
                    detected_scripts=profile.scripts_present,
                    script_profile=profile
                )
            elif dominant == "Telugu":
                conf = max(0.85, min(0.99, profile.indic_script_ratio))
                return LanguageProfileResult(
                    primary_language="te_telugu",
                    confidence=conf,
                    is_code_mixed=is_code_mixed,
                    is_romanized=False,
                    is_uncertain=False,
                    scores={"te_telugu": conf, "en": 1.0 - conf},
                    detected_scripts=profile.scripts_present,
                    script_profile=profile
                )
            elif dominant == "Bengali":
                conf = max(0.85, min(0.99, profile.indic_script_ratio))
                return LanguageProfileResult(
                    primary_language="bn_bengali",
                    confidence=conf,
                    is_code_mixed=is_code_mixed,
                    is_romanized=False,
                    is_uncertain=False,
                    scores={"bn_bengali": conf, "en": 1.0 - conf},
                    detected_scripts=profile.scripts_present,
                    script_profile=profile
                )

        # 3. Short input check
        if len(stripped) < self.min_length_for_confidence and profile.token_count <= 1:
            return LanguageProfileResult(
                primary_language="unknown",
                confidence=0.20,
                is_code_mixed=False,
                is_romanized=False,
                is_uncertain=True,
                scores={"unknown": 0.80, "en": 0.10, "hi_en_roman": 0.10},
                abstention_reason=f"Input length ({len(stripped)} chars) is below minimum confidence threshold",
                detected_scripts=profile.scripts_present,
                script_profile=profile
            )

        # 4. Romanized / Latin analysis (Hinglish, Telgish, Benglish, English)
        words = re.findall(r"[a-zA-Z]+", stripped.lower())
        if not words:
            # Only digits or symbols
            return LanguageProfileResult(
                primary_language="unknown",
                confidence=0.25,
                is_code_mixed=False,
                is_romanized=False,
                is_uncertain=True,
                scores={"unknown": 1.0},
                abstention_reason="No alphabetical tokens found in input",
                detected_scripts=profile.scripts_present,
                script_profile=profile
            )

        # Lexical matching counts
        hi_lex_hits = sum(1 for w in words if w in ROMAN_HINDI_LEXICON)
        te_lex_hits = sum(1 for w in words if w in ROMAN_TELUGU_LEXICON)
        bn_lex_hits = sum(1 for w in words if w in ROMAN_BENGALI_LEXICON)
        en_lex_hits = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)

        # Trigram scoring
        trigrams = _extract_trigrams(stripped)
        trigram_scores = {"hi_en_roman": 0.0, "te_en_roman": 0.0, "bn_en_roman": 0.0, "en": 0.0}
        for tg in trigrams:
            for lang, weights in TRIGRAM_WEIGHTS.items():
                if tg in weights:
                    trigram_scores[lang] += weights[tg]

        # Combine lexical hits with trigram evidence
        total_tokens = max(1, len(words))
        raw_scores = {
            "hi_en_roman": (hi_lex_hits * 3.0) + (trigram_scores["hi_en_roman"] * 0.4),
            "te_en_roman": (te_lex_hits * 3.5) + (trigram_scores["te_en_roman"] * 0.4),
            "bn_en_roman": (bn_lex_hits * 3.5) + (trigram_scores["bn_en_roman"] * 0.4),
            "en": (en_lex_hits * 1.5) + (trigram_scores["en"] * 0.3) + 0.5  # slight prior for English
        }

        # Softmax normalization
        max_val = max(raw_scores.values())
        exp_scores = {k: math.exp(v - max_val) for k, v in raw_scores.items()}
        sum_exp = sum(exp_scores.values())
        norm_scores = {k: round(v / sum_exp, 4) for k, v in exp_scores.items()}

        # Sort candidate languages
        sorted_candidates = sorted(norm_scores.items(), key=lambda item: item[1], reverse=True)
        top_lang, top_prob = sorted_candidates[0]
        second_lang, second_prob = sorted_candidates[1]

        # Margin and uncertainty
        margin = top_prob - second_prob
        is_uncertain = False
        abstention_reason = None

        if top_prob < self.abstention_threshold or (margin < 0.08 and top_prob < 0.45):
            is_uncertain = True
            abstention_reason = f"High ambiguity between {top_lang} ({top_prob:.2f}) and {second_lang} ({second_prob:.2f})"

        # Determine code-mixing and romanization
        is_romanized = top_lang in {"hi_en_roman", "te_en_roman", "bn_en_roman"}
        # Code mixed if an Indic Romanized language has English words or English text has Indic words
        is_code_mixed = False
        if is_romanized and (en_lex_hits > 0 or profile.indic_script_ratio > 0.05):
            is_code_mixed = True
        elif top_lang == "en" and (hi_lex_hits > 0 or te_lex_hits > 0 or bn_lex_hits > 0):
            is_code_mixed = True
            if hi_lex_hits >= 1 and hi_lex_hits >= en_lex_hits:
                top_lang = "hi_en_roman"
                is_romanized = True

        return LanguageProfileResult(
            primary_language=top_lang if not (is_uncertain and top_prob < 0.30) else "unknown",
            confidence=top_prob,
            is_code_mixed=is_code_mixed,
            is_romanized=is_romanized,
            is_uncertain=is_uncertain,
            scores=norm_scores,
            abstention_reason=abstention_reason,
            detected_scripts=profile.scripts_present,
            script_profile=profile
        )
