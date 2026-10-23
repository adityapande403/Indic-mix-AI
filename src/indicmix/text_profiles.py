"""Text profiling and Unicode script analysis for Indic and code-mixed text.

Handles Unicode code-point vs UTF-16 offset mapping, script distribution analysis,
and lexical token characterization.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TextProfile:
    """Statistical profile of input text."""
    char_count: int
    token_count: int
    dominant_script: str
    script_distribution: Dict[str, float]
    latin_ratio: float
    indic_script_ratio: float
    digit_ratio: float
    whitespace_ratio: float
    is_multiscript: bool
    scripts_present: List[str] = field(default_factory=list)


def get_character_script(ch: str) -> str:
    """Identify the Unicode script or character category for a single character."""
    code = ord(ch)
    
    # Common Indic Unicode blocks
    if 0x0900 <= code <= 0x097F:
        return "Devanagari"
    elif 0x0C00 <= code <= 0x0C7F:
        return "Telugu"
    elif 0x0980 <= code <= 0x09FF:
        return "Bengali"
    elif 0x0B80 <= code <= 0x0BFF:
        return "Tamil"
    elif 0x0C80 <= code <= 0x0CFF:
        return "Kannada"
    elif 0x0D00 <= code <= 0x0D7F:
        return "Malayalam"
    elif 0x0A80 <= code <= 0x0AFF:
        return "Gujarati"
    elif 0x0A00 <= code <= 0x0A7F:
        return "Gurmukhi"
    elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
        return "Arabic_Urdu"
    
    # Latin
    if (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A) or (0x00C0 <= code <= 0x024F):
        return "Latin"
    
    # General categories
    cat = unicodedata.category(ch)
    if cat.startswith("N"):
        return "Number"
    elif cat.startswith("P"):
        return "Punctuation"
    elif cat.startswith("Z"):
        return "Whitespace"
    elif cat.startswith("S"):
        return "Symbol"
    
    return "Other"


def analyze_text_profile(text: str) -> TextProfile:
    """Compute script distribution, token count, and mixing ratios for text."""
    if not text:
        return TextProfile(
            char_count=0,
            token_count=0,
            dominant_script="Empty",
            script_distribution={},
            latin_ratio=0.0,
            indic_script_ratio=0.0,
            digit_ratio=0.0,
            whitespace_ratio=0.0,
            is_multiscript=False,
            scripts_present=[]
        )

    char_counts: Dict[str, int] = {}
    tokens = text.split()
    total_chars = len(text)
    
    indic_scripts = {"Devanagari", "Telugu", "Bengali", "Tamil", "Kannada", "Malayalam", "Gujarati", "Gurmukhi", "Arabic_Urdu"}
    
    for ch in text:
        script = get_character_script(ch)
        char_counts[script] = char_counts.get(script, 0) + 1
        
    script_dist = {script: count / total_chars for script, count in char_counts.items()}
    
    latin_ratio = script_dist.get("Latin", 0.0)
    digit_ratio = script_dist.get("Number", 0.0)
    whitespace_ratio = script_dist.get("Whitespace", 0.0)
    indic_ratio = sum(script_dist.get(s, 0.0) for s in indic_scripts)
    
    # Major scripts present excluding whitespace, punctuation, numbers
    alphabetical_scripts = [
        s for s, count in char_counts.items()
        if s not in {"Whitespace", "Punctuation", "Number", "Symbol", "Other"} and count > 0
    ]
    
    # Dominant script among alphabets
    if alphabetical_scripts:
        dominant_script = max(alphabetical_scripts, key=lambda s: char_counts[s])
    else:
        dominant_script = max(char_counts.keys(), key=lambda s: char_counts[s]) if char_counts else "Unknown"
        
    is_multiscript = len(alphabetical_scripts) > 1

    return TextProfile(
        char_count=total_chars,
        token_count=len(tokens),
        dominant_script=dominant_script,
        script_distribution=script_dist,
        latin_ratio=latin_ratio,
        indic_script_ratio=indic_ratio,
        digit_ratio=digit_ratio,
        whitespace_ratio=whitespace_ratio,
        is_multiscript=is_multiscript,
        scripts_present=alphabetical_scripts
    )


def codepoint_to_utf16_offset(text: str, cp_offset: int) -> int:
    """Convert Python Unicode code-point offset to UTF-16 code unit offset.
    
    JavaScript and browsers index strings by UTF-16 code units (where surrogate pairs
    occupy 2 units). This function guarantees exact parity across runtimes.
    """
    if cp_offset <= 0:
        return 0
    clamped_offset = min(cp_offset, len(text))
    prefix = text[:clamped_offset]
    return len(prefix.encode("utf-16-le")) // 2


def utf16_to_codepoint_offset(text: str, utf16_offset: int) -> int:
    """Convert UTF-16 code unit offset to Python Unicode code-point offset."""
    if utf16_offset <= 0:
        return 0
    
    u16_units = 0
    for idx, ch in enumerate(text):
        if u16_units >= utf16_offset:
            return idx
        code = ord(ch)
        u16_units += 2 if code > 0xFFFF else 1
    return len(text)
