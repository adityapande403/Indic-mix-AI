/**
 * IndicMix AI Node.js Entrypoint
 */

import { NodeIntentClassifier } from "./classifier.js";
import { NodeTextNormalizer } from "./normalizer.js";
import { NodePIIDetector } from "./pii.js";

export class IndicMix {
  constructor(classifier) {
    this.classifier = classifier;
  }

  static async load(options = {}) {
    const modelPath = options.intentModel || "../models/onnx/classifier_quantized.onnx";
    const vocabPath = options.vocabPath || "../models/checkpoints/classifier/vocab.json";
    const metadataPath = options.metadataPath || "../models/checkpoints/classifier/model_metadata.json";

    const classifier = await NodeIntentClassifier.load(modelPath, vocabPath, metadataPath);
    return new IndicMix(classifier);
  }

  async classifyIntent(text) {
    return this.classifier.classify(text);
  }

  detectPII(text) {
    return NodePIIDetector.detect(text);
  }

  redactPII(text) {
    return NodePIIDetector.redact(text);
  }

  normalize(text, mode = "english", targetLanguage = "en") {
    return NodeTextNormalizer.normalize(text, mode, targetLanguage);
  }

  async analyze(text, tasks = ["intent", "pii", "normalization"]) {
    const res = { sourceText: text };
    if (tasks.includes("intent")) {
      res.intent = await this.classifyIntent(text);
    }
    if (tasks.includes("pii")) {
      res.pii = this.detectPII(text);
      res.redacted = this.redactPII(text).redactedText;
    }
    if (tasks.includes("normalization")) {
      res.normalization = this.normalize(text);
    }
    return res;
  }
}

export { NodeIntentClassifier, NodeTextNormalizer, NodePIIDetector };
