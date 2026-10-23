/**
 * IndicMix Node.js Example Usage
 */

import { IndicMix } from "./index.js";

async function main() {
  console.log("Loading IndicMix in Node.js...");
  const engine = await IndicMix.load({
    intentModel: "../models/onnx/classifier_quantized.onnx",
    vocabPath: "../models/checkpoints/classifier/vocab.json",
    metadataPath: "../models/checkpoints/classifier/model_metadata.json"
  });

  const sampleText = "bhai payment deduct ho gaya par order confirm nahi dikha raha call 9876543210";
  console.log("\nInput:", sampleText);

  const intent = await engine.classifyIntent(sampleText);
  console.log("Intent Result:", intent);

  const pii = engine.detectPII(sampleText);
  console.log("PII Detected:", pii);

  const redaction = engine.redactPII(sampleText);
  console.log("Redacted Text:", redaction.redactedText);

  const normalized = engine.normalize(sampleText, "english");
  console.log("Normalized English:", normalized.normalizedText);
}

main().catch(console.error);
