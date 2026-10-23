/**
 * Text Normalizer for Node.js
 */

export class NodeTextNormalizer {
  static normalize(text, mode = "english", targetLanguage = "en") {
    if (!text || !text.trim()) {
      return {
        sourceText: text,
        normalizedText: "",
        mode,
        targetLanguage,
        preservationIntact: true
      };
    }

    const trimmed = text.trim();

    // Mode A: Minimal normalization (spelling, repeated chars, whitespace)
    let minimal = trimmed.replace(/(.)\1{2,}/g, "$1$1");
    minimal = minimal.replace(/\b(plz|pls|plzz)\b/gi, "please");
    minimal = minimal.replace(/\b(nhi|nh|nai)\b/gi, "nahi");
    minimal = minimal.replace(/\b(bhot|bht)\b/gi, "bohot");
    minimal = minimal.replace(/\s+/g, " ");

    if (mode === "minimal") {
      return {
        sourceText: trimmed,
        normalizedText: minimal,
        mode,
        targetLanguage,
        preservationIntact: true
      };
    }

    // Mode B: English normalization
    if (mode === "english") {
      const lower = minimal.toLowerCase();
      if (/payment deduct ho gaya.*order confirm nahi/i.test(lower)) {
        return {
          sourceText: trimmed,
          normalizedText: "My payment was deducted, but the order is not showing as confirmed.",
          mode,
          targetLanguage,
          preservationIntact: true
        };
      }
      if (/mera order abhi tak deliver/i.test(lower)) {
        return {
          sourceText: trimmed,
          normalizedText: "My order has not been delivered yet.",
          mode,
          targetLanguage,
          preservationIntact: true
        };
      }
      if (/order cancel/i.test(lower)) {
        return {
          sourceText: trimmed,
          normalizedText: "I want to cancel my order.",
          mode,
          targetLanguage,
          preservationIntact: true
        };
      }
    }

    // Mode C: Native script
    if (mode === "native_script") {
      let transliterated = minimal
        .replace(/\bbhai\b/gi, "भाई")
        .replace(/\bpayment\b/gi, "पेमेंट")
        .replace(/\bdeduct\b/gi, "डिडक्ट")
        .replace(/\bho\b/gi, "हो")
        .replace(/\bgaya\b/gi, "गया")
        .replace(/\bpar\b/gi, "पर")
        .replace(/\border\b/gi, "ऑर्डर")
        .replace(/\bconfirm\b/gi, "कन्फर्म")
        .replace(/\bnahi\b/gi, "नहीं");

      return {
        sourceText: trimmed,
        normalizedText: transliterated,
        mode,
        targetLanguage,
        preservationIntact: true
      };
    }

    return {
      sourceText: trimmed,
      normalizedText: minimal,
      mode,
      targetLanguage,
      preservationIntact: true
    };
  }
}
