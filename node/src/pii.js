/**
 * PII Detection and Redaction for Node.js
 */

export class NodePIIDetector {
  static detect(text) {
    if (!text) return [];

    const spans = [];

    // Phone: Indian 10-digit mobile
    const phoneRegex = /(?<!\w)(?:\+?91[\s.-]?)?([6-9]\d{4}[\s.-]?\d{5})(?!\w)/g;
    let match;
    while ((match = phoneRegex.exec(text)) !== null) {
      spans.push({
        type: "PHONE",
        start: match.index,
        end: match.index + match[0].length,
        text: match[0],
        confidence: 0.96
      });
    }

    // Email
    const emailRegex = /(?<!\w)[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+(?!\w)/g;
    while ((match = emailRegex.exec(text)) !== null) {
      spans.push({
        type: "EMAIL",
        start: match.index,
        end: match.index + match[0].length,
        text: match[0],
        confidence: 0.99
      });
    }

    // UPI ID
    const upiRegex = /(?<!\w)[a-zA-Z0-9._-]+@(okaxis|okhdfcbank|oksbi|paytm|ybl|upi)(?!\w)/gi;
    while ((match = upiRegex.exec(text)) !== null) {
      spans.push({
        type: "UPI_ID",
        start: match.index,
        end: match.index + match[0].length,
        text: match[0],
        confidence: 0.98
      });
    }

    // Order ID
    const orderRegex = /(?<!\w)(?:ORD|OD|ORDER)[-_#]?[A-Z0-9]{6,14}(?!\w)/gi;
    while ((match = orderRegex.exec(text)) !== null) {
      spans.push({
        type: "ORDER_ID",
        start: match.index,
        end: match.index + match[0].length,
        text: match[0],
        confidence: 0.95
      });
    }

    // Sort by start position
    return spans.sort((a, b) => a.start - b.start);
  }

  static redact(text, spans = null) {
    if (!text) return { originalText: text, redactedText: text, count: 0 };
    const detected = spans || NodePIIDetector.detect(text);
    if (!detected.length) {
      return { originalText: text, redactedText: text, count: 0 };
    }

    // Sort in reverse order of start index to safely slice
    const sortedRev = [...detected].sort((a, b) => b.start - a.start);
    let result = text;
    for (const s of sortedRev) {
      const mask = `[${s.type}]`;
      result = result.substring(0, s.start) + mask + result.substring(s.end);
    }

    return {
      originalText: text,
      redactedText: result,
      count: detected.length,
      spans: detected
    };
  }
}
