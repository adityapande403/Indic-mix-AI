/**
 * ONNX-backed Intent Classifier for Node.js
 */

import fs from "node:fs";
import * as ort from "onnxruntime-node";

export class NodeIntentClassifier {
  constructor(session, vocab, metadata) {
    this.session = session;
    this.vocab = vocab;
    this.metadata = metadata;
    this.labels = metadata.labels || [];
    this.maxLength = metadata.max_length || 64;
  }

  static async load(modelPath, vocabPath, metadataPath) {
    let session = null;
    if (fs.existsSync(modelPath)) {
      session = await ort.InferenceSession.create(modelPath);
    }
    const vocabData = JSON.parse(fs.readFileSync(vocabPath, "utf-8"));
    const metadata = fs.existsSync(metadataPath)
      ? JSON.parse(fs.readFileSync(metadataPath, "utf-8"))
      : { labels: ["greeting", "order_status", "payment_issue", "refund_request", "cancellation_request", "unknown"] };

    return new NodeIntentClassifier(session, vocabData.w2i, metadata);
  }

  encode(text) {
    const words = text.toLowerCase().split(/\s+/);
    const tokens = [];

    for (const w of words) {
      tokens.push(this.vocab[w] ?? this.vocab["<unk>"] ?? 1);
      // Subword character n-grams
      for (const n of [3, 4]) {
        if (w.length >= n) {
          for (let i = 0; i <= w.length - n; i++) {
            const sub = w.substring(i, i + n);
            tokens.push(this.vocab[sub] ?? this.vocab["<unk>"] ?? 1);
          }
        }
      }
    }

    const trimmed = tokens.slice(0, this.maxLength);
    const padLen = this.maxLength - trimmed.length;
    const inputIds = trimmed.concat(new Array(padLen).fill(0));
    const attentionMask = new Array(trimmed.length).fill(1).concat(new Array(padLen).fill(0));

    return { inputIds, attentionMask };
  }

  async classify(text) {
    if (!text || !text.trim()) {
      return {
        intent: "unknown",
        confidence: 0.0,
        isAbstained: true,
        reason: "Empty text"
      };
    }

    if (!this.session) {
      // Fallback rule classification if model session is unavailable
      const lower = text.toLowerCase();
      if (/payment|deduct|kat gaye/i.test(lower)) {
        return { intent: "payment_issue", confidence: 0.92, isAbstained: false };
      }
      if (/order|track|deliver/i.test(lower)) {
        return { intent: "order_status", confidence: 0.88, isAbstained: false };
      }
      return { intent: "unknown", confidence: 0.2, isAbstained: true };
    }

    const { inputIds, attentionMask } = this.encode(text);
    const tensorIds = new ort.Tensor("int64", BigInt64Array.from(inputIds.map(BigInt)), [1, this.maxLength]);
    const tensorMask = new ort.Tensor("int64", BigInt64Array.from(attentionMask.map(BigInt)), [1, this.maxLength]);

    const feeds = {
      input_ids: tensorIds,
      attention_mask: tensorMask
    };

    const results = await this.session.run(feeds);
    const logits = Array.from(results.logits.data);

    // Softmax
    const maxVal = Math.max(...logits);
    const expVals = logits.map((v) => Math.exp(v - maxVal));
    const sumExp = expVals.reduce((a, b) => a + b, 0);
    const probs = expVals.map((v) => v / sumExp);

    let topIdx = 0;
    let maxProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        topIdx = i;
      }
    }

    const intent = this.labels[topIdx] || "unknown";
    const allScores = {};
    for (let i = 0; i < this.labels.length; i++) {
      allScores[this.labels[i]] = probs[i] || 0.0;
    }

    return {
      intent,
      confidence: parseFloat(maxProb.toFixed(4)),
      allScores,
      isAbstained: maxProb < 0.35
    };
  }
}
