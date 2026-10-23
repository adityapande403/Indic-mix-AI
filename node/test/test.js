/**
 * Test Suite for Node.js IndicMix implementation
 */

import test from "node:test";
import assert from "node:assert/strict";
import { IndicMix } from "../src/index.js";
import { NodePIIDetector } from "../src/pii.js";
import { NodeTextNormalizer } from "../src/normalizer.js";

test("PII Detection in Node.js", () => {
  const text = "Call 9876543210 or email test@example.com for order ORD-998811";
  const spans = NodePIIDetector.detect(text);
  assert.equal(spans.length, 3);
  
  const types = spans.map((s) => s.type);
  assert.ok(types.includes("PHONE"));
  assert.ok(types.includes("EMAIL"));
  assert.ok(types.includes("ORDER_ID"));
});

test("PII Redaction in Node.js", () => {
  const text = "mera number 9876543210 hai";
  const result = NodePIIDetector.redact(text);
  assert.ok(result.redactedText.includes("[PHONE]"));
  assert.ok(!result.redactedText.includes("9876543210"));
});

test("Text Normalization in Node.js", () => {
  const text = "bhaiii payment deduct ho gaya plzzzz help";
  const minimal = NodeTextNormalizer.normalize(text, "minimal");
  assert.ok(!minimal.normalizedText.includes("plzzzz"));
  assert.ok(!minimal.normalizedText.includes("bhaiii"));
});

test("IndicMix Engine Loading and Intent Classification in Node.js", async () => {
  const engine = await IndicMix.load({
    intentModel: "../models/onnx/classifier_quantized.onnx",
    vocabPath: "../models/checkpoints/classifier/vocab.json",
    metadataPath: "../models/checkpoints/classifier/model_metadata.json"
  });

  const res = await engine.classifyIntent("bhai payment deduct ho gaya");
  assert.ok(res.intent);
  assert.ok(typeof res.confidence === "number");
});
