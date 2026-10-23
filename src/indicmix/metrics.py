"""Evaluation Metrics Suite for IndicMix AI.

Computes comprehensive metrics for:
- Multiclass Intent Classification
- Multilabel Moderation
- PII Entity and Span-Level Token Classification
- Text Normalization and Entity Preservation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute per-class and aggregated classification metrics."""
    if not y_true or not y_pred:
        return {}

    all_labels = sorted(list(set(y_true) | set(y_pred))) if labels is None else labels
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    num_classes = len(all_labels)

    # Build confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    total_samples = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total_samples if total_samples > 0 else 0.0

    per_class = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

    for idx, label in enumerate(all_labels):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        support = int(np.sum(cm[idx, :]))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        per_class[label] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "support": support
        }

        macro_p += p
        macro_r += r
        macro_f1 += f1

        weight = support / total_samples if total_samples > 0 else 0.0
        weighted_p += p * weight
        weighted_r += r * weight
        weighted_f1 += f1 * weight

    num_eval_classes = max(1, num_classes)
    macro_p /= num_eval_classes
    macro_r /= num_eval_classes
    macro_f1 /= num_eval_classes

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": all_labels,
        "total_samples": total_samples
    }


def compute_multilabel_moderation_metrics(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    categories: List[str]
) -> Dict[str, Any]:
    """Compute per-category precision, recall, F1, FPR, FNR for moderation."""
    per_category = {}
    macro_f1_sum = 0.0

    for idx, cat in enumerate(categories):
        y_t = y_true_binary[:, idx]
        y_p = y_pred_binary[:, idx]

        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        per_category[cat] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4),
            "support": int(np.sum(y_t == 1))
        }
        macro_f1_sum += f1

    macro_f1 = macro_f1_sum / max(1, len(categories))
    return {
        "macro_f1": round(float(macro_f1), 4),
        "per_category": per_category,
        "total_evaluated": len(y_true_binary)
    }


def compute_pii_span_metrics(
    true_spans_list: List[List[Dict[str, Any]]],
    pred_spans_list: List[List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Compute exact and entity-level precision, recall, and F1 for PII spans."""
    total_gold = 0
    total_pred = 0
    exact_matches = 0
    type_matches = 0

    for gold_spans, pred_spans in zip(true_spans_list, pred_spans_list):
        total_gold += len(gold_spans)
        total_pred += len(pred_spans)

        # Build tuples (type, start, end)
        gold_set = {(s["type"], s["start"], s["end"]) for s in gold_spans}
        pred_set = {(s["type"], s["start"], s["end"]) for s in pred_spans}

        exact_matches += len(gold_set & pred_set)

    precision = exact_matches / total_pred if total_pred > 0 else 0.0
    recall = exact_matches / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "exact_match_precision": round(float(precision), 4),
        "exact_match_recall": round(float(recall), 4),
        "exact_match_f1": round(float(f1), 4),
        "total_gold_spans": total_gold,
        "total_pred_spans": total_pred,
        "exact_matches": exact_matches
    }


def compute_normalization_metrics(
    references: List[str],
    hypotheses: List[str],
    source_texts: List[str]
) -> Dict[str, Any]:
    """Compute character error rate, exact match, and critical entity preservation."""
    total = len(references)
    if total == 0:
        return {}

    exact_matches = sum(1 for r, h in zip(references, hypotheses) if r.strip().lower() == h.strip().lower())
    
    # Character Error Rate (Levenshtein-based)
    total_ref_chars = sum(len(r) for r in references)
    total_char_dist = 0
    for r, h in zip(references, hypotheses):
        total_char_dist += _levenshtein_distance(r.lower(), h.lower())

    cer = total_char_dist / max(1, total_ref_chars)

    # Entity preservation across source and hypothesis
    from .normalization import extract_critical_entities
    total_entities = 0
    preserved_entities = 0

    for src, hyp in zip(source_texts, hypotheses):
        src_ents = extract_critical_entities(src)
        total_entities += len(src_ents)
        for ent in src_ents:
            if ent.lower() in hyp.lower():
                preserved_entities += 1

    preservation_rate = preserved_entities / max(1, total_entities) if total_entities > 0 else 1.0

    return {
        "exact_match_accuracy": round(exact_matches / total, 4),
        "character_error_rate": round(cer, 4),
        "entity_preservation_rate": round(preservation_rate, 4),
        "total_samples": total,
        "total_tracked_entities": total_entities,
        "preserved_entities": preserved_entities
    }


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]
