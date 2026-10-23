"""Comprehensive multi-task evaluation harness for IndicMix AI.

Evaluates:
1. Intent Classification (Macro-F1, Weighted-F1, per-class P/R/F1, Confusion Matrix, ECE)
2. Moderation (Per-category F1, False Positive Rate, False Negative Rate, Macro-F1)
3. PII Detection (Exact span match P/R/F1, Redaction accuracy)
4. Text Normalization (CER, Exact Match, Strict Entity Preservation)
Outputs evaluation reports in JSON and Markdown formats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from indicmix.calibration import compute_expected_calibration_error
from indicmix.metrics import (
    compute_classification_metrics,
    compute_multilabel_moderation_metrics,
    compute_normalization_metrics,
    compute_pii_span_metrics
)
from indicmix.pipeline import IndicMix


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def run_evaluation(
    intent_test_path: str = "data/processed/intent/test.jsonl",
    mod_test_path: str = "data/processed/moderation/test.jsonl",
    pii_test_path: str = "data/processed/pii/test.jsonl",
    norm_test_path: str = "data/processed/normalization/test.jsonl",
    output_dir: str = "evaluation/results"
) -> Dict[str, Any]:
    print("=== Running IndicMix AI Multi-Task Evaluation ===")
    os.makedirs(output_dir, exist_ok=True)
    engine = IndicMix()

    results: Dict[str, Any] = {}

    # 1. Evaluate Intent Classification
    intent_data = load_jsonl(intent_test_path)
    if intent_data:
        print(f"Evaluating Intent Classification ({len(intent_data)} test samples)...")
        y_true, y_pred, confs = [], [], []
        for d in intent_data:
            res = engine.classify_intent(d["text"])
            y_true.append(d["label"])
            y_pred.append(res.intent)
            confs.append(res.confidence)

        intent_metrics = compute_classification_metrics(y_true, y_pred)
        acc_array = np.array([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
        ece, mce, _, _, _ = compute_expected_calibration_error(np.array(confs), acc_array)
        intent_metrics["calibration"] = {
            "expected_calibration_error": round(ece, 4),
            "maximum_calibration_error": round(mce, 4)
        }
        results["intent"] = intent_metrics
        print(f"Intent Macro-F1: {intent_metrics.get('macro_f1', 0):.4f}, Accuracy: {intent_metrics.get('accuracy', 0):.4f}, ECE: {ece:.4f}")

    # 2. Evaluate Moderation
    mod_data = load_jsonl(mod_test_path)
    if mod_data:
        print(f"Evaluating Content Moderation ({len(mod_data)} test samples)...")
        categories = list(engine.moderation_engine.categories)
        y_true_bin = []
        y_pred_bin = []

        for d in mod_data:
            gold_cats = d.get("categories", {})
            gold_vec = [gold_cats.get(c, 0) for c in categories]
            y_true_bin.append(gold_vec)

            mod_res = engine.moderate(d["text"])
            pred_vec = [1 if c in mod_res.flagged_categories else 0 for c in categories]
            y_pred_bin.append(pred_vec)

        mod_metrics = compute_multilabel_moderation_metrics(
            np.array(y_true_bin),
            np.array(y_pred_bin),
            categories
        )
        results["moderation"] = mod_metrics
        print(f"Moderation Macro-F1: {mod_metrics.get('macro_f1', 0):.4f}")

    # 3. Evaluate PII Detection & Redaction
    pii_data = load_jsonl(pii_test_path)
    if pii_data:
        print(f"Evaluating PII Detection & Redaction ({len(pii_data)} test samples)...")
        gold_spans_all = []
        pred_spans_all = []

        for d in pii_data:
            gold_spans = d.get("entities", [])
            gold_spans_all.append(gold_spans)

            detected = engine.detect_pii(d["text"])
            pred_spans_all.append([s.to_dict() for s in detected])

        pii_metrics = compute_pii_span_metrics(gold_spans_all, pred_spans_all)
        results["pii"] = pii_metrics
        print(f"PII Exact Span Match F1: {pii_metrics.get('exact_match_f1', 0):.4f}")

    # 4. Evaluate Text Normalization
    norm_data = load_jsonl(norm_test_path)
    if norm_data:
        print(f"Evaluating Text Normalization ({len(norm_data)} test samples)...")
        references, hypotheses, sources = [], [], []

        for d in norm_data:
            src = d["source_text"]
            tgt = d["target_text"]
            mode = d.get("mode", "english")
            tgt_lang = d.get("target_language", "en")

            norm_res = engine.normalize(src, mode=mode, target_language=tgt_lang)
            sources.append(src)
            references.append(tgt)
            hypotheses.append(norm_res.normalized_text)

        norm_metrics = compute_normalization_metrics(references, hypotheses, sources)
        results["normalization"] = norm_metrics
        print(f"Normalization Entity Preservation: {norm_metrics.get('entity_preservation_rate', 0):.4f}, CER: {norm_metrics.get('character_error_rate', 0):.4f}")

    # Save JSON Report
    json_path = os.path.join(output_dir, "evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown Report
    md_path = os.path.join(output_dir, "evaluation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# IndicMix AI: Comprehensive Evaluation Report\n\n")
        f.write(f"Generated on test splits.\n\n")
        
        if "intent" in results:
            f.write("## 1. Intent Classification\n")
            f.write(f"- **Accuracy**: {results['intent'].get('accuracy', 0):.4f}\n")
            f.write(f"- **Macro-F1**: {results['intent'].get('macro_f1', 0):.4f}\n")
            f.write(f"- **Weighted-F1**: {results['intent'].get('weighted_f1', 0):.4f}\n")
            f.write(f"- **ECE (Expected Calibration Error)**: {results['intent']['calibration']['expected_calibration_error']:.4f}\n\n")

        if "moderation" in results:
            f.write("## 2. Content Moderation\n")
            f.write(f"- **Macro-F1**: {results['moderation'].get('macro_f1', 0):.4f}\n")
            f.write("| Category | Precision | Recall | F1 | FPR | FNR |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for cat, m in results['moderation'].get('per_category', {}).items():
                f.write(f"| {cat} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['false_positive_rate']:.3f} | {m['false_negative_rate']:.3f} |\n")
            f.write("\n")

        if "pii" in results:
            f.write("## 3. PII Detection\n")
            f.write(f"- **Exact Span Precision**: {results['pii'].get('exact_match_precision', 0):.4f}\n")
            f.write(f"- **Exact Span Recall**: {results['pii'].get('exact_match_recall', 0):.4f}\n")
            f.write(f"- **Exact Span F1**: {results['pii'].get('exact_match_f1', 0):.4f}\n\n")

        if "normalization" in results:
            f.write("## 4. Text Normalization\n")
            f.write(f"- **Exact Match Accuracy**: {results['normalization'].get('exact_match_accuracy', 0):.4f}\n")
            f.write(f"- **Character Error Rate (CER)**: {results['normalization'].get('character_error_rate', 0):.4f}\n")
            f.write(f"- **Entity Preservation Rate**: {results['normalization'].get('entity_preservation_rate', 0):.4f}\n\n")

    print(f"Evaluation reports written to: {json_path} and {md_path}")
    return results


if __name__ == "__main__":
    run_evaluation()
