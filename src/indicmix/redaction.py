"""PII Redaction Engine.

Applies configurable, deterministic redaction masks to sensitive entity spans
with offset preservation and overlapping entity resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .pii_detection import EntitySpan, merge_and_deduplicate_spans


@dataclass
class RedactionResult:
    """Result of PII redaction."""
    original_text: str
    redacted_text: str
    spans_redacted: List[EntitySpan]
    entity_counts: Dict[str, int]
    num_entities_redacted: int


class PIIRedactor:
    """Configurable redactor for PII entities."""

    DEFAULT_MASKS = {
        "PHONE": "[PHONE]",
        "EMAIL": "[EMAIL]",
        "UPI_ID": "[UPI_ID]",
        "PAN": "[PAN]",
        "AADHAAR": "[AADHAAR]",
        "VEHICLE_NO": "[VEHICLE_NO]",
        "IFSC": "[IFSC]",
        "ORDER_ID": "[ORDER_ID]",
        "NAME": "[NAME]",
        "ADDRESS": "[ADDRESS]",
        "ORGANIZATION": "[ORGANIZATION]"
    }

    def __init__(
        self,
        redact_names: bool = True,
        custom_masks: Optional[Dict[str, str]] = None,
        mask_format: str = "[{tag}]"
    ):
        self.redact_names = redact_names
        self.masks = dict(self.DEFAULT_MASKS)
        if custom_masks:
            self.masks.update(custom_masks)
        self.mask_format = mask_format

    def get_mask(self, entity_type: str) -> str:
        """Get replacement string mask for an entity type."""
        if entity_type in self.masks:
            return self.masks[entity_type]
        return self.mask_format.format(tag=entity_type)

    def redact(
        self,
        text: str,
        spans: List[EntitySpan],
        redact_names: Optional[bool] = None
    ) -> RedactionResult:
        """Redact sensitive entity spans from input text.
        
        Args:
            text: Original string.
            spans: List of detected EntitySpan objects.
            redact_names: Override instance default for name redaction.
            
        Returns:
            RedactionResult with redacted text and audit metadata.
        """
        if not text or not spans:
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                spans_redacted=[],
                entity_counts={},
                num_entities_redacted=0
            )

        should_redact_names = self.redact_names if redact_names is None else redact_names
        
        # Filter spans if name redaction is disabled
        filtered_spans = [
            s for s in spans
            if s.entity_type != "NAME" or should_redact_names
        ]

        # Ensure deduplicated and non-overlapping
        resolved_spans = merge_and_deduplicate_spans(filtered_spans)

        # Sort in reverse order of code-point start so replacements do not alter preceding indices
        sorted_rev = sorted(resolved_spans, key=lambda s: s.start, reverse=True)

        redacted_chars = list(text)
        entity_counts: Dict[str, int] = {}
        applied_spans: List[EntitySpan] = []

        for span in sorted_rev:
            # Code-point slice bounds
            start = max(0, min(span.start, len(redacted_chars)))
            end = max(start, min(span.end, len(redacted_chars)))
            
            mask = self.get_mask(span.entity_type)
            redacted_chars[start:end] = list(mask)
            
            entity_counts[span.entity_type] = entity_counts.get(span.entity_type, 0) + 1
            applied_spans.append(span)

        # Re-sort applied spans in natural forward order for reporting
        applied_spans.reverse()

        return RedactionResult(
            original_text=text,
            redacted_text="".join(redacted_chars),
            spans_redacted=applied_spans,
            entity_counts=entity_counts,
            num_entities_redacted=len(applied_spans)
        )
