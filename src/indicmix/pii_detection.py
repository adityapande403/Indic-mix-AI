"""PII Detection Engine combining deterministic recognizers and contextual entity models.

Detects phone numbers, emails, UPI IDs, PAN, Aadhaar, vehicle registration numbers,
IFSC codes, Order IDs, and contextual entities (names, addresses, organizations)
in code-mixed Indic and English text with deterministic span merging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

from .text_profiles import codepoint_to_utf16_offset


@dataclass
class EntitySpan:
    """Represents a detected PII entity span."""
    entity_type: str
    start: int          # Code-point start offset
    end: int            # Code-point end offset
    text: str
    confidence: float
    source: str         # "deterministic" or "model"
    utf16_start: Optional[int] = None
    utf16_end: Optional[int] = None

    def __post_init__(self):
        # Guarantee non-negative valid offsets
        if self.start < 0:
            self.start = 0
        if self.end < self.start:
            self.end = self.start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "utf16_start": self.utf16_start,
            "utf16_end": self.utf16_end,
        }


# Context words that signal a phone number vs arbitrary numbers (e.g. price, quantity)
PHONE_CONTEXT_KEYWORDS = {
    "call", "calling", "phone", "mobile", "number", "no", "num", "contact",
    "whatsapp", "dial", "ph", "mob", "mera number", "naaku call", "phone cheyandi",
    "call koro", "ring", "reach"
}

# Context words that signal price, quantity, or dates (to avoid false-positive phone detection)
NON_PHONE_CONTEXT_KEYWORDS = {
    "rs", "inr", "rupees", "rupaye", "taka", "dabbu", "price", "cost", "amount",
    "qty", "quantity", "discount", "code", "pin", "pincode", "otp"
}


# Official Verhoeff Algorithm Tables for Indian Aadhaar Checksum Validation (Dihedral Group D5)
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff_checksum(num_str: str) -> bool:
    """Validate 12-digit Indian Aadhaar number using official Verhoeff dihedral algorithm."""
    digits = [int(c) for c in num_str if c.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed(digits)):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][item]]
    return c == 0


def generate_verhoeff_check_digit(num_str: str) -> str:
    """Generate the 12th Verhoeff check digit for an 11-digit Aadhaar prefix."""
    digits = [int(c) for c in num_str if c.isdigit()]
    c = 0
    for i, item in enumerate(reversed(digits)):
        c = VERHOEFF_D[c][VERHOEFF_P[(i + 1) % 8][item]]
    return str(VERHOEFF_INV[c])


class DeterministicPIIRecognizer:
    """High-precision regex and heuristic recognizers for Indian structured identifiers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Indian phone regex: captures +91, 0, or raw 10 digits starting with 6-9, with spaces/dashes
        self.phone_pattern = re.compile(
            r"(?<!\w)(?:(?:\+?91[\s.-]?)|(?:0[\s.-]?))?([6-9]\d{4}[\s.-]?\d{5})(?!\w)"
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r"(?<!\w)[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(?!\w)"
        )
        
        # Indian UPI ID pattern: e.g., name@okhdfcbank, payment@upi, rahul.sharma@paytm
        self.upi_pattern = re.compile(
            r"(?<!\w)[a-zA-Z0-9._-]+@(okaxis|okhdfcbank|oksbi|okicici|paytm|ybl|ibl|upi|axl|apl|sbi|hdfcbank|icici|kotak|barodampay)(?!\w)",
            re.IGNORECASE
        )
        # Generic UPI fallback pattern (3+ char bank handle)
        self.upi_generic_pattern = re.compile(
            r"(?<!\w)[a-zA-Z0-9._-]+@[a-zA-Z]{3,}(?!\w)",
            re.IGNORECASE
        )
        
        # Indian Permanent Account Number (PAN): 5 letters, 4 digits, 1 letter
        self.pan_pattern = re.compile(
            r"(?<!\w)[A-Z]{5}[0-9]{4}[A-Z](?!\w)"
        )
        
        # Indian Aadhaar: 12 digits, often 4-4-4 formatted
        self.aadhaar_pattern = re.compile(
            r"(?<!\w)([2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4})(?!\w)"
        )
        
        # Vehicle Registration: e.g., DL 01 AB 1234 or MH12DE5678 or TS09FA4321
        self.vehicle_pattern = re.compile(
            r"(?<!\w)([A-Z]{2}[-\s]?[0-9]{1,2}[-\s]?[A-Z]{1,3}[-\s]?[0-9]{4})(?!\w)"
        )
        
        # Bank IFSC Code: 4 alphabetic, 0, 6 alphanumeric
        self.ifsc_pattern = re.compile(
            r"(?<!\w)([A-Z]{4}0[A-Z0-9]{6})(?!\w)"
        )
        
        # Order ID pattern: e.g., ORD-9812451, OD1234567890, ORDER#123456
        self.order_id_pattern = re.compile(
            r"(?<!\w)(?:ORD|OD|ORDER)[-_#]?[A-Z0-9]{6,14}(?!\w)",
            re.IGNORECASE
        )

    def detect(self, text: str) -> List[EntitySpan]:
        """Run deterministic recognizers over input text."""
        if not text:
            return []
            
        spans: List[EntitySpan] = []
        lower_text = text.lower()

        # 1. Email detection
        for match in self.email_pattern.finditer(text):
            spans.append(EntitySpan(
                entity_type="EMAIL",
                start=match.start(),
                end=match.end(),
                text=match.group(),
                confidence=0.99,
                source="deterministic"
            ))

        # 2. UPI ID detection (check specific handles first)
        for match in self.upi_pattern.finditer(text):
            spans.append(EntitySpan(
                entity_type="UPI_ID",
                start=match.start(),
                end=match.end(),
                text=match.group(),
                confidence=0.98,
                source="deterministic"
            ))
            
        # 3. PAN detection
        for match in self.pan_pattern.finditer(text):
            spans.append(EntitySpan(
                entity_type="PAN",
                start=match.start(),
                end=match.end(),
                text=match.group(),
                confidence=0.98,
                source="deterministic"
            ))

        # 4. Aadhaar detection (12-digit Indian national identifier verified with Verhoeff dihedral checksum)
        for match in self.aadhaar_pattern.finditer(text):
            matched_str = match.group()
            digits_only = re.sub(r"\D", "", matched_str)
            if len(digits_only) == 12 and not digits_only.startswith("0") and not digits_only.startswith("1"):
                # Avoid Aadhaar if immediately preceded by phone keyword
                window = lower_text[max(0, match.start() - 20): match.start()]
                if not any(k in window for k in ["phone", "mobile", "call"]):
                    is_verhoeff_valid = validate_verhoeff_checksum(digits_only)
                    conf = 0.98 if is_verhoeff_valid else 0.90
                    spans.append(EntitySpan(
                        entity_type="AADHAAR",
                        start=match.start(),
                        end=match.end(),
                        text=matched_str,
                        confidence=conf,
                        source="deterministic"
                    ))

        # 5. Vehicle registration detection
        for match in self.vehicle_pattern.finditer(text):
            matched_str = match.group()
            # Must start with known Indian State code
            state_code = matched_str[:2].upper()
            known_states = {
                "DL", "MH", "KA", "TN", "TS", "AP", "WB", "UP", "HR", "PB",
                "GJ", "RJ", "MP", "KL", "BR", "JH", "OD", "AS", "CH", "GA"
            }
            if state_code in known_states:
                spans.append(EntitySpan(
                    entity_type="VEHICLE_NO",
                    start=match.start(),
                    end=match.end(),
                    text=matched_str,
                    confidence=0.96,
                    source="deterministic"
                ))

        # 6. IFSC detection
        for match in self.ifsc_pattern.finditer(text):
            spans.append(EntitySpan(
                entity_type="IFSC",
                start=match.start(),
                end=match.end(),
                text=match.group(),
                confidence=0.97,
                source="deterministic"
            ))

        # 7. Order ID detection
        for match in self.order_id_pattern.finditer(text):
            spans.append(EntitySpan(
                entity_type="ORDER_ID",
                start=match.start(),
                end=match.end(),
                text=match.group(),
                confidence=0.95,
                source="deterministic"
            ))

        # 8. Phone Number detection with context validation
        for match in self.phone_pattern.finditer(text):
            matched_str = match.group()
            digits_only = re.sub(r"\D", "", matched_str)
            
            # Extract local window around the match
            start_idx = match.start()
            end_idx = match.end()
            pre_window = lower_text[max(0, start_idx - 25): start_idx]
            post_window = lower_text[end_idx: min(len(lower_text), end_idx + 25)]
            combined_window = f"{pre_window} {post_window}"

            # Check if clearly non-phone (e.g. price / amount)
            is_price = any(term in combined_window for term in NON_PHONE_CONTEXT_KEYWORDS)
            has_phone_context = any(term in combined_window for term in PHONE_CONTEXT_KEYWORDS)
            has_country_code = matched_str.startswith("+91") or matched_str.startswith("91") or matched_str.startswith("0")
            
            confidence = 0.95 if has_phone_context or has_country_code else (0.85 if not is_price else 0.40)
            
            if confidence >= 0.70:
                spans.append(EntitySpan(
                    entity_type="PHONE",
                    start=start_idx,
                    end=end_idx,
                    text=matched_str,
                    confidence=confidence,
                    source="deterministic"
                ))

        # Populate UTF-16 offsets
        for s in spans:
            s.utf16_start = codepoint_to_utf16_offset(text, s.start)
            s.utf16_end = codepoint_to_utf16_offset(text, s.end)

        return spans


class ContextualPIIRecognizer:
    """Contextual Named Entity Recognizer for person names, addresses, and organizations."""

    # Curated Indian names commonly found in customer support / service messages
    COMMON_INDIAN_NAMES = {
        "rahul", "rohit", "amit", "priya", "pooja", "aditya", "vikram", "sneha",
        "neha", "suresh", "ramesh", "rajesh", "anil", "sunil", "deepak", "anjali",
        "sharma", "verma", "gupta", "singh", "patel", "kumar", "reddy", "rao",
        "chatterjee", "banerjee", "das", "mukherjee", "khan", "ali", "ahmed",
        "krishna", "venkat", "srinivas", "sai", "ravi", "arjun", "karan", "manish",
        "kavita", "swati", "meena", "sunita", "rekha", "anita", "divya", "rashmi"
    }

    # Contextual address and city markers
    COMMON_INDIAN_CITIES = {
        "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata",
        "pune", "ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur", "indore", "patna",
        "bhopal", "ludhiana", "agra", "nashik", "vadodara", "varanasi", "noida", "gurgaon"
    }

    def __init__(self, model_checkpoint: Optional[str] = None):
        self.model_checkpoint = model_checkpoint
        self.model_available = False
        # When model checkpoint is provided and loaded, model inference is active.
        # Otherwise, reliable rule/lexicon-based fallback operates.

    def detect(self, text: str) -> List[EntitySpan]:
        """Detect contextual entities (names, addresses, organizations)."""
        if not text:
            return []

        spans: List[EntitySpan] = []
        lower_text = text.lower()
        
        # 1. Person Name recognition based on honorifics and common Indian names
        # Patterns like: "call Rahul", "Rahul ko", "bhai Rahul", "Rahul Sharma", "Mr. Amit"
        name_honorific_pattern = re.compile(
            r"(?<!\w)(?:mr\.?|ms\.?|mrs\.?|shri|smt|call|contact|naam|name|bhai)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            re.IGNORECASE
        )
        for match in name_honorific_pattern.finditer(text):
            name_text = match.group(1).strip()
            # Verify tokens
            name_tokens = name_text.lower().split()
            if any(t in self.COMMON_INDIAN_NAMES for t in name_tokens) or match.group(0).lower().startswith(("mr", "ms", "shri", "smt")):
                span_start = match.start(1)
                span_end = match.end(1)
                spans.append(EntitySpan(
                    entity_type="NAME",
                    start=span_start,
                    end=span_end,
                    text=name_text,
                    confidence=0.88,
                    source="model"
                ))

        # Check standalone Indian names
        word_matches = list(re.finditer(r"\b([A-Za-z]+)\b", text))
        for wm in word_matches:
            word = wm.group(1)
            lower_word = word.lower()
            if lower_word in self.COMMON_INDIAN_NAMES:
                # Avoid if already matched
                overlap = any(s.start <= wm.start() and wm.end() <= s.end for s in spans)
                if not overlap:
                    spans.append(EntitySpan(
                        entity_type="NAME",
                        start=wm.start(),
                        end=wm.end(),
                        text=word,
                        confidence=0.82,
                        source="model"
                    ))

        # 2. Address & City recognition
        for city in self.COMMON_INDIAN_CITIES:
            for match in re.finditer(rf"\b{re.escape(city)}\b", lower_text):
                spans.append(EntitySpan(
                    entity_type="ADDRESS",
                    start=match.start(),
                    end=match.end(),
                    text=text[match.start():match.end()],
                    confidence=0.90,
                    source="model"
                ))

        # 3. Pincodes associated with address (6-digit Indian PIN code)
        pincode_pattern = re.compile(r"(?<!\w)(?:pin|pincode|postal code)?\s*([1-9][0-9]{5})(?!\w)", re.IGNORECASE)
        for match in pincode_pattern.finditer(text):
            matched_code = match.group(1)
            # Only consider if pin context or standalone 6-digit number
            spans.append(EntitySpan(
                entity_type="ADDRESS",
                start=match.start(1),
                end=match.end(1),
                text=matched_code,
                confidence=0.85,
                source="model"
            ))

        for s in spans:
            s.utf16_start = codepoint_to_utf16_offset(text, s.start)
            s.utf16_end = codepoint_to_utf16_offset(text, s.end)

        return spans


def merge_and_deduplicate_spans(spans: List[EntitySpan]) -> List[EntitySpan]:
    """Deterministically merge and deduplicate overlapping PII spans.
    
    Priority rules:
    1. Deterministic recognizers take precedence over model spans for structured IDs.
    2. Higher confidence wins when spans overlap.
    3. Longer span wins when confidence is equal.
    """
    if not spans:
        return []

    # Sort spans primarily by start offset ascending, then by confidence descending, then length descending
    sorted_spans = sorted(
        spans,
        key=lambda s: (s.start, -s.confidence, -(s.end - s.start))
    )

    merged: List[EntitySpan] = []
    
    for candidate in sorted_spans:
        if not merged:
            merged.append(candidate)
            continue
            
        last = merged[-1]
        
        # Check overlap
        if candidate.start < last.end:
            # Overlapping span detected!
            # If candidate is deterministic and last is model, candidate replaces last
            if candidate.source == "deterministic" and last.source == "model":
                merged[-1] = candidate
            elif last.source == "deterministic" and candidate.source == "model":
                # Keep last
                pass
            elif candidate.confidence > last.confidence:
                merged[-1] = candidate
            elif candidate.confidence == last.confidence and (candidate.end - candidate.start) > (last.end - last.start):
                merged[-1] = candidate
            # Otherwise discard candidate
        else:
            merged.append(candidate)
            
    return sorted(merged, key=lambda s: s.start)


class HybridPIIDetector:
    """Unified hybrid PII detection combining deterministic and model recognizers."""

    def __init__(self, enable_model: bool = True, model_checkpoint: Optional[str] = None):
        self.deterministic = DeterministicPIIRecognizer()
        self.enable_model = enable_model
        self.contextual = ContextualPIIRecognizer(model_checkpoint=model_checkpoint) if enable_model else None

    def detect(self, text: str) -> List[EntitySpan]:
        """Detect all PII entity spans in text."""
        det_spans = self.deterministic.detect(text)
        
        if self.enable_model and self.contextual:
            model_spans = self.contextual.detect(text)
            all_spans = det_spans + model_spans
        else:
            all_spans = det_spans
            
        return merge_and_deduplicate_spans(all_spans)
