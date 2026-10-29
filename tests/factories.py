"""Shared builders for valid wire payloads used across test modules."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from multimodal_vision_pipeline.schemas import ALL_RULES, PipelineModel

NOW = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
JOB_ID = UUID("6f1c2a52-5d0e-4c3e-9b7a-1f2e3d4c5b6a")
SHA256 = "a" * 64


def parse[M: PipelineModel](model: type[M], data: object) -> M:
    """Validate ``data`` the way it arrives over the wire (JSON text)."""
    return model.model_validate_json(json.dumps(data))


def anchor(page: int = 1, bbox: tuple[float, ...] = (0.1, 0.1, 0.5, 0.2)) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    return {"page_number": page, "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}}


def ef(
    value: object,
    *,
    status: str = "extracted",
    confidence: float = 0.95,
    anchor_: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"value": value, "status": status, "confidence": confidence, "anchor": anchor_}


def missing() -> dict[str, Any]:
    return ef(None, status="missing", confidence=0.0)


def invoice_data(**overrides: object) -> dict[str, Any]:
    """A self-consistent two-page invoice.

    2 x 12.50 = 25.00; 1.5 x 19.99 = 29.985 -> 29.99 (half-up); subtotal 54.99;
    23% of 54.99 = 12.6477 -> 12.65; total 54.99 + 12.65 - 0 = 67.64.
    """
    data: dict[str, Any] = {
        "document_kind": "invoice",
        "model": "claude-sonnet-5-5",
        "status": "complete",
        "page_count": 2,
        "vendor": {
            "name": ef("ACME Sp. z o.o.", anchor_=anchor(1)),
            "tax_id": ef("PL5250001009"),
            "address": ef("ul. Prosta 1, 00-001 Warszawa"),
        },
        "document_number": ef("FV/2026/10/0042"),
        "issue_date": ef("2026-10-29"),
        "currency": ef("PLN", status="inferred", confidence=0.8),
        "line_items": [
            {
                "description": "Widget",
                "quantity": "2",
                "unit_price_cents": 1250,
                "amount_cents": 2500,
                "tax_rate": "23",
                "confidence": 0.97,
                "anchor": anchor(1, (0.05, 0.4, 0.95, 0.45)),
            },
            {
                "description": "Copper wire (kg)",
                "quantity": "1.5",
                "unit_price_cents": 1999,
                "amount_cents": 2999,
                "tax_rate": "23",
                "confidence": 0.91,
                "anchor": anchor(2, (0.05, 0.1, 0.95, 0.15)),
            },
        ],
        "tax_breakdown": [
            {"rate": "23", "taxable_base_cents": 5499, "tax_cents": 1265, "confidence": 0.9}
        ],
        "subtotal": ef(5499),
        "tax_total": ef(1265),
        "discount": ef(0, status="inferred", confidence=0.6),
        "total": ef(6764, anchor_=anchor(2, (0.6, 0.8, 0.95, 0.85))),
    }
    data.update(overrides)
    return data


def report_data(**overrides: object) -> dict[str, Any]:
    data: dict[str, Any] = {
        "status": "passed",
        "policy": {"tolerance_cents": 1, "rounding": "half_up", "rules": sorted(ALL_RULES)},
        "evaluated_rules": sorted(ALL_RULES),
        "skipped_rules": [],
        "discrepancies": [],
    }
    data.update(overrides)
    return data


def descriptor_data(**overrides: object) -> dict[str, Any]:
    data: dict[str, Any] = {
        "filename": "invoice.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 48_213,
        "sha256": SHA256,
        "page_count": 2,
    }
    data.update(overrides)
    return data


def page_data(page: int) -> dict[str, Any]:
    return {
        "page_number": page,
        "width_px": 1654,
        "height_px": 2339,
        "dpi": 200,
        "color_mode": "RGB",
    }


def job_data(**overrides: object) -> dict[str, Any]:
    data: dict[str, Any] = {
        "job_id": str(JOB_ID),
        "status": "succeeded",
        "document": descriptor_data(),
        "created_at": NOW.isoformat(),
        "updated_at": (NOW + timedelta(seconds=9)).isoformat(),
        "pages": [page_data(1), page_data(2)],
        "extraction": invoice_data(),
        "reconciliation": report_data(),
        "error": None,
    }
    data.update(overrides)
    return data


def model_reply(document_number: str = "FV/2026/10/0042") -> str:
    """Structured-output text a vision model would return; anchors removed so it fits
    any page selection."""
    data = invoice_data(document_number=ef(document_number))
    data["vendor"]["name"] = ef("ACME Sp. z o.o.")
    data["total"] = ef(6764)
    for item in data["line_items"]:
        item["anchor"] = None
    del data["model"], data["page_count"]
    return json.dumps(data)


def messages_api_body(text: str, *, model: str = "claude-sonnet-5-5") -> dict[str, Any]:
    """A successful Messages API response carrying ``text`` as its only text block."""
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1000, "output_tokens": 200},
    }
