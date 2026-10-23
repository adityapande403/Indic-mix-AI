# IndicMix AI — Browser Inference Demonstration

This directory contains an in-browser client-side demonstration of **IndicMix AI**.

## Key Features
- **100% Client-Side Privacy**: Source text never leaves the user's browser device. All inference runs locally via WebAssembly / JavaScript.
- **Code-Mixed Language Identification**: Accurately disambiguates Hinglish, Telgish, Benglish, English, and native Indic scripts (Devanagari, Telugu, Bengali).
- **Intent Classification & Uncertainty Abstention**: Calibrated intent predictions with unknown-intent handling.
- **Content Moderation & Harm Prevention**: Flags abuse, threats, harassment, and spam while strictly protecting harmless colloquial slang (`bhai`, `yaar`, `dada`, `anna`).
- **Sensitive PII Detection & Interactive Redaction**: Identifies Indian mobile numbers, emails, UPI IDs, PAN, and Order IDs with an interactive redaction toggle.
- **Multi-Mode Normalization**: Supports Mode A (minimal spelling & punctuation), Mode B (English normalization), and Mode C (native script transliteration).

## How to Run Locally

Start a lightweight HTTP server in this directory:

```bash
# Using Python
python -m http.server 8080 --directory .

# Or using Node
npx serve .
```

Open `http://localhost:8080` in your web browser.
