"""Text Normalization Engine for Code-Mixed Indic text.

Supports:
- Mode A: Minimal normalization (spelling correction, letter deduplication, spacing, punctuation).
- Mode B: English normalization (semantic rewriting into clear, natural English).
- Mode C: Native-script transliteration (Hinglish -> Devanagari, Telgish -> Telugu, Benglish -> Bengali).
- Strict entity-preservation verification (amounts, IDs, dates, numbers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class NormalizationResult:
    """Output of text normalization."""
    source_text: str
    normalized_text: str
    mode: str
    target_language: str
    preserved_entities: List[str]
    preservation_intact: bool
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_text": self.source_text,
            "normalized_text": self.normalized_text,
            "mode": self.mode,
            "target_language": self.target_language,
            "preserved_entities": self.preserved_entities,
            "preservation_intact": self.preservation_intact,
            "warnings": self.warnings
        }


# Informal spelling dictionary for code-mixed chat/support
INFORMAL_SPELLINGS = {
    # Hinglish spelling variants
    "nhi": "nahi",
    "nh": "nahi",
    "nai": "nahi",
    "ni": "nahi",
    "bhot": "bohot",
    "bht": "bohot",
    "boht": "bohot",
    "plz": "please",
    "pls": "please",
    "plzz": "please",
    "thx": "thanks",
    "thnx": "thanks",
    "ty": "thank you",
    "yr": "yaar",
    "kyu": "kyun",
    "kyn": "kyun",
    "kch": "kuch",
    "kuchh": "kuch",
    "achha": "accha",
    "thk": "theek",
    "abt": "about",
    "bcoz": "because",
    "coz": "because",
    "bcz": "because",
    "msg": "message",
    "acc": "account",
    "accnt": "account",
    "amt": "amount",
    "ref": "reference",
    "txn": "transaction",
    "tx": "transaction",
    "delvry": "delivery",
    "delivry": "delivery",
    "pymt": "payment",
    "pymnt": "payment",
    # Telgish variants
    "chala": "chaala",
    "kda": "kadha",
    "avvatle": "avvatledu",
    "chey": "cheyyi",
    # Benglish variants
    "hyeche": "hoyeche",
    "kora": "kora",
    "hbe": "hobe"
}

# Common phrase translations / rewrites for Mode B (English normalization)
COMMON_PHRASE_TRANSLATIONS = [
    # Payment and deduction
    (r"\b(?:bhai|yaar)?\s*payment deduct ho gaya par order confirm nahi dikha raha\b",
     "My payment was deducted, but the order is not showing as confirmed."),
    (r"\bmera order abhi tak deliver nhi hua\b",
     "My order has not been delivered yet."),
    (r"\bpaise account se kat gaye refund kab tak aayega\b",
     "Money was debited from my account, when will the refund arrive?"),
    (r"\bpayment cut ho gaya lekin order place nahi hua\b",
     "The payment was deducted, but the order was not placed."),
    (r"\bmera refund kab aayega\b",
     "When will I receive my refund?"),
    (r"\border cancel karna chahta hu\b",
     "I want to cancel my order."),
    (r"\border status kya hai\b",
     "What is the status of my order?"),
    (r"\blogin nahi ho raha hai\b",
     "I am unable to log in to my account."),
    (r"\bapp kaam nahi kar raha\b",
     "The mobile application is not working properly."),
    (r"\bwrong item deliver hua hai\b",
     "A wrong item was delivered to me."),
    (r"\bdelivery boy ne call nahi kiya\b",
     "The delivery executive did not call me."),
    
    # Telgish phrases
    (r"\bnaaku order deliver avvaledu\b",
     "My order has not been delivered to me."),
    (r"\bdabbu cut ayindi kani order confirm kaledu\b",
     "The money was deducted, but the order was not confirmed."),
    (r"\bnaa refund eppudu vastundi\b",
     "When will my refund arrive?"),
    (r"\border cancel cheyandi please\b",
     "Please cancel my order."),
    (r"\baccount login avvatledu\b",
     "I am unable to log in to my account."),
    
    # Benglish phrases
    (r"\btaka kete geche kintu order confirm hoyni\b",
     "Money was deducted, but the order was not confirmed."),
    (r"\bamar order ekhono deliver hoyni\b",
     "My order has not been delivered yet."),
    (r"\brefund kobe pabo\b",
     "When will I receive the refund?"),
    (r"\border cancel korte chai\b",
     "I want to cancel the order.")
]

# Lexical mapping for word-level English translation in Mode B
INDIC_TO_ENGLISH_WORDS = {
    "mera": "my", "meri": "my", "mere": "my", "mujhe": "me", "mujhko": "me",
    "humara": "our", "aapka": "your", "tumhara": "your", "kya": "what",
    "kyun": "why", "kab": "when", "kaha": "where", "kaise": "how",
    "nahi": "not", "par": "but", "lekin": "but", "aur": "and", "bhi": "also",
    "paise": "money", "rupaye": "money", "taka": "money", "dabbu": "money",
    "jaldi": "urgently", "theek": "okay", "accha": "good", "bohot": "very",
    "chahiye": "needed", "karo": "please do", "batao": "please tell",
    "naaku": "to me", "meeku": "to you", "eppudu": "when", "enduku": "why",
    "amar": "my", "tomar": "your", "kobe": "when", "keno": "why"
}

# Phonetic transliteration tables for Mode C
ROMAN_TO_DEVANAGARI = {
    "bhai": "भाई", "payment": "पेमेंट", "deduct": "डिडक्ट", "ho": "हो",
    "gaya": "गया", "par": "पर", "order": "ऑर्डर", "confirm": "कन्फर्म",
    "nahi": "नहीं", "dikha": "दिख", "raha": "रहा", "hai": "है",
    "mera": "मेरा", "meri": "मेरी", "mere": "मेरे", "abhi": "अभी",
    "tak": "तक", "deliver": "डिलीवर", "hua": "हुआ", "paise": "पैसे",
    "account": "अकाउंट", "se": "से", "kat": "कट", "gaye": "गए",
    "refund": "रिफंड", "kab": "कब", "aayega": "आएगा", "lekin": "लेकिन",
    "cancel": "कैंसल", "karna": "करना", "chahta": "चाहता", "hu": "हूँ",
    "status": "स्टेटस", "kya": "क्या", "login": "लॉगिन", "app": "ऐप",
    "kaam": "काम", "kar": "कर", "wrong": "रॉन्ग", "item": "आइटम",
    "delivery": "डिलीवरी", "boy": "बॉय", "ne": "ने", "call": "कॉल",
    "kiya": "किया", "chahiye": "चाहिए", "kripya": "कृपया", "theek": "ठीक",
    "bohot": "बहुत", "jaldi": "जल्दी", "problem": "प्रॉब्लम", "issue": "इशू",
    "support": "सपोर्ट", "help": "हेल्प", "shukriya": "शुक्रिया"
}

ROMAN_TO_TELUGU = {
    "naaku": "నాకు", "order": "ఆర్డర్", "deliver": "డెలివర్", "avvaledu": "అవ్వలేదు",
    "dabbu": "డబ్బు", "cut": "కట్", "ayindi": "అయింది", "kani": "కాని",
    "confirm": "కన్ఫర్మ్", "kaledu": "కాలేదు", "naa": "నా", "refund": "రీఫండ్",
    "eppudu": "ఎప్పుడు", "vastundi": "వస్తుంది", "cancel": "రద్దు", "cheyandi": "చేయండి",
    "please": "దయచేసి", "account": "ఖాతా", "login": "లాగిన్", "avvatledu": "అవ్వట్లేదు",
    "chala": "చాలా", "undi": "ఉంది", "ledu": "లేదు", "enti": "ఏంటి", "ela": "ఎలా"
}

ROMAN_TO_BENGALI = {
    "taka": "টাকা", "kete": "কেটে", "geche": "গেছে", "kintu": "কিন্তু",
    "order": "অর্ডার", "confirm": "কনফার্ম", "hoyni": "হয়নি", "amar": "আমার",
    "ekhono": "এখনো", "deliver": "ডেলিভার", "refund": "রিফান্ড", "kobe": "কবে",
    "pabo": "পাবো", "cancel": "ক্যানসেল", "korte": "করতে", "chai": "চাই",
    "dada": "দাদা", "bhai": "ভাই", "ektu": "একটু", "shob": "সব",
    "khub": "খুব", "bhalo": "ভালো", "ki": "কী", "keno": "কেন"
}


def deduplicate_repeated_chars(text: str) -> str:
    """Normalize exaggerated repeated characters (e.g. 'plzzzz' -> 'plz', 'bohooooot' -> 'bohot').
    
    Keeps at most 2 repeated characters for normal doubled letters (e.g., 'good', 'success').
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def extract_critical_entities(text: str) -> List[str]:
    """Extract numbers, currency amounts, codes, and IDs that must NEVER be altered."""
    entities: List[str] = []
    
    # Currency amounts: e.g. Rs. 500, Rs 500, ₹1500, 1000/-, 500 INR, 500 rupees
    amount_matches = re.findall(
        r"(?:(?:Rs\.?|INR|₹)\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:rs|inr|rupees|rupaye|taka|dabbu|/-)\b)",
        text,
        re.IGNORECASE
    )
    entities.extend(amount_matches)
    
    # Generic numbers/quantities (2+ digits)
    number_matches = re.findall(r"\b\d{2,}\b", text)
    entities.extend(number_matches)
    
    # Specific Order/Transaction IDs: ORD-12345, TXN98765, etc.
    id_matches = re.findall(r"\b[A-Za-z0-9_-]{6,}\b", text)
    for m in id_matches:
        if any(c.isdigit() for c in m) and any(c.isalpha() for c in m):
            entities.append(m)
            
    return list(dict.fromkeys(entities))


NEGATION_PATTERN = re.compile(
    r"\b("
    r"nahi|nhi|na|mat|ni|nai|kabhi\s+nahi|"
    r"ledu|vaddu|kadu|kaadu|avvaledu|kaledu|chudaledu|raaledu|ravatledu|avvatledu|"
    r"noy|hoyni|hoynai|paini|bujhini|asche\s+na|parina|"
    r"not|no|never|none|neither|nor|nothing|nowhere|failed|failure|unable|cannot|cant|can't|didn't|didnt|won't|wont|isn't|isnt|haven't|havent|hasn't|hasnt|don't|dont"
    r")\b",
    re.IGNORECASE
)

NATIVE_NEGATION_WORDS = {
    # Devanagari
    "नहीं", "नही", "ना", "मत", "न",
    # Telugu
    "లేదు", "కాలేదు", "రాలేదు", "అవ్వలేదు", "వద్దు", "కాదు", "అవ్వట్లేదు",
    # Bengali
    "হয়নি", "হয়নি", "না", "নয়", "পাইনি", "আসছে না"
}


def check_negation_polarity(text: str) -> bool:
    """Detect whether text contains negative polarity markers across code-mixed Indic, English, and native scripts."""
    if not text:
        return False
    if NEGATION_PATTERN.search(text):
        return True
    for word in NATIVE_NEGATION_WORDS:
        if word in text:
            return True
    return False


def verify_polarity_preservation(source_text: str, target_text: str) -> Tuple[bool, List[str]]:
    """Verify that negation polarity is preserved (preventing 'not received' -> 'received' corruptions)."""
    source_has_neg = check_negation_polarity(source_text)
    target_has_neg = check_negation_polarity(target_text)

    if source_has_neg != target_has_neg:
        if source_has_neg and not target_has_neg:
            warning = "Polarity inversion detected: Source text contains negation, but normalized text lost negation."
        else:
            warning = "Polarity inversion detected: Source text is affirmative, but normalized text introduced negation."
        return False, [warning]
    return True, []


def verify_entity_preservation(source_text: str, target_text: str) -> Tuple[bool, List[str], List[str]]:
    """Verify that all critical entities from source are preserved in target text."""
    critical_entities = extract_critical_entities(source_text)
    missing: List[str] = []
    
    for entity in critical_entities:
        # Check presence in target text
        if entity.lower() not in target_text.lower():
            # Check digits only in case format changed slightly
            digits = re.sub(r"\D", "", entity)
            if digits and len(digits) >= 2 and digits not in target_text:
                missing.append(entity)

    is_intact = len(missing) == 0
    warnings = [f"Critical entity dropped or corrupted: '{m}'" for m in missing]
    return is_intact, critical_entities, warnings


class TextNormalizer:
    """Multi-mode Indic text normalizer with entity preservation guarantees."""

    def __init__(self, strict_entity_preservation: bool = True):
        self.strict = strict_entity_preservation

    def normalize(
        self,
        text: str,
        mode: str = "minimal",
        target_language: str = "en",
        source_language: Optional[str] = None
    ) -> NormalizationResult:
        """Normalize text according to specified mode and target language."""
        if not text or not text.strip():
            return NormalizationResult(
                source_text=text,
                normalized_text="",
                mode=mode,
                target_language=target_language,
                preserved_entities=[],
                preservation_intact=True
            )

        stripped = text.strip()
        warnings: List[str] = []

        if mode == "minimal":
            norm = self._normalize_minimal(stripped)
        elif mode == "english":
            norm = self._normalize_to_english(stripped)
        elif mode == "native_script":
            norm = self._normalize_to_native_script(stripped, source_language or "hi_en_roman")
        else:
            warnings.append(f"Unknown mode '{mode}', defaulting to minimal normalization.")
            norm = self._normalize_minimal(stripped)

        # Entity preservation validation
        is_intact, entities, entity_warnings = verify_entity_preservation(stripped, norm)
        warnings.extend(entity_warnings)

        # Polarity preservation validation
        polarity_intact, polarity_warnings = verify_polarity_preservation(stripped, norm)
        warnings.extend(polarity_warnings)

        if (not is_intact or not polarity_intact) and self.strict and mode != "minimal":
            if not is_intact:
                warnings.append("Strict entity preservation failed: falling back to minimal normalization to prevent data loss.")
            if not polarity_intact:
                warnings.append("Strict polarity preservation failed: falling back to minimal normalization to prevent semantic inversion.")
            norm = self._normalize_minimal(stripped)
            is_intact = True

        return NormalizationResult(
            source_text=stripped,
            normalized_text=norm,
            mode=mode,
            target_language=target_language,
            preserved_entities=entities,
            preservation_intact=is_intact,
            warnings=warnings
        )

    def _normalize_minimal(self, text: str) -> str:
        """Mode A: Spelling correction, spacing, punctuation, character deduplication."""
        # 1. Deduplicate repeated letters
        text = deduplicate_repeated_chars(text)
        
        # 2. Token-level informal spelling correction
        words = text.split()
        normalized_words = []
        for w in words:
            # Preserve punctuation around word
            match = re.match(r"^([^\w]*)(.*?)([^\w]*)$", w)
            if match:
                prefix, core, suffix = match.groups()
                core_lower = core.lower()
                if core_lower in INFORMAL_SPELLINGS:
                    corrected = INFORMAL_SPELLINGS[core_lower]
                    # Preserve capitalization if original was capitalized
                    if core.istitle():
                        corrected = corrected.capitalize()
                    elif core.isupper() and len(core) > 1:
                        corrected = corrected.upper()
                    normalized_words.append(f"{prefix}{corrected}{suffix}")
                else:
                    normalized_words.append(w)
            else:
                normalized_words.append(w)

        result = " ".join(normalized_words)
        # Whitespace clean up
        result = re.sub(r"\s+", " ", result).strip()
        
        # Ensure ending punctuation if appropriate
        if result and result[-1] not in ".!?":
            # Add question mark if starts with question words
            q_words = {"kya", "kyun", "kab", "kaha", "kaise", "what", "why", "when", "how", "enti", "eppudu", "kobe"}
            first_word = result.split()[0].lower() if result.split() else ""
            if first_word in q_words:
                result += "?"
            else:
                result += "."
                
        return result

    def _normalize_to_english(self, text: str) -> str:
        """Mode B: Rewrite code-mixed text into natural, grammatically correct English."""
        clean_text = deduplicate_repeated_chars(text)
        
        # Check phrase match first
        for pattern, english_replacement in COMMON_PHRASE_TRANSLATIONS:
            if re.search(pattern, clean_text, re.IGNORECASE):
                # If entities like order ID or amounts are in original, splice them in
                entities = extract_critical_entities(clean_text)
                target = english_replacement
                if entities:
                    target_has_entities = all(e.lower() in target.lower() for e in entities)
                    if not target_has_entities:
                        # Append reference to entities
                        joined_entities = ", ".join(entities)
                        target = f"{target.rstrip('.')} (Ref: {joined_entities})."
                return target

        # Word-level transliteration + translation fallback
        words = clean_text.split()
        translated_tokens = []
        for w in words:
            match = re.match(r"^([^\w]*)(.*?)([^\w]*)$", w)
            if match:
                prefix, core, suffix = match.groups()
                c_lower = core.lower()
                # Check informal spelling first
                if c_lower in INFORMAL_SPELLINGS:
                    c_lower = INFORMAL_SPELLINGS[c_lower]
                # Check English translation
                if c_lower in INDIC_TO_ENGLISH_WORDS:
                    rep = INDIC_TO_ENGLISH_WORDS[c_lower]
                    translated_tokens.append(f"{prefix}{rep}{suffix}")
                else:
                    translated_tokens.append(w)
            else:
                translated_tokens.append(w)

        res = " ".join(translated_tokens)
        res = re.sub(r"\s+", " ", res).strip()
        if res and res[-1] not in ".!?":
            res += "."
        return res

    def _normalize_to_native_script(self, text: str, source_language: str) -> str:
        """Mode C: Transliterate Romanized text to target native script."""
        clean = deduplicate_repeated_chars(text)
        words = clean.split()
        
        if source_language.startswith("te") or "telugu" in source_language:
            mapping = ROMAN_TO_TELUGU
        elif source_language.startswith("bn") or "bengali" in source_language:
            mapping = ROMAN_TO_BENGALI
        else:
            mapping = ROMAN_TO_DEVANAGARI

        transliterated = []
        for w in words:
            match = re.match(r"^([^\w]*)(.*?)([^\w]*)$", w)
            if match:
                prefix, core, suffix = match.groups()
                c_lower = core.lower()
                if c_lower in INFORMAL_SPELLINGS:
                    c_lower = INFORMAL_SPELLINGS[c_lower]
                if c_lower in mapping:
                    transliterated.append(f"{prefix}{mapping[c_lower]}{suffix}")
                else:
                    transliterated.append(w)
            else:
                transliterated.append(w)

        return " ".join(transliterated)
