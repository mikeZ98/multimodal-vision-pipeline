"""Exact arithmetic reconciliation of an extraction result against a policy.

Every rule either evaluates (possibly producing discrepancies) or is skipped because
the values it needs were not extracted. Nothing here uses ``float``.
"""

from collections.abc import Callable

from multimodal_vision_pipeline.core.errors import ReconciliationOverflowError
from multimodal_vision_pipeline.schemas import (
    MAX_CENTS,
    Discrepancy,
    ExtractedField,
    ExtractionResult,
    FieldStatus,
    ReconciliationPolicy,
    ReconciliationReport,
    ReconciliationRule,
    ReconciliationStatus,
)

type Check = Callable[[ExtractionResult, ReconciliationPolicy], list[Discrepancy] | None]
"""Returns discrepancies found, or ``None`` when the rule cannot be evaluated."""


def _compare(
    policy: ReconciliationPolicy,
    rule: ReconciliationRule,
    *,
    expected: int,
    actual: int,
    location: str,
    message: str,
) -> Discrepancy | None:
    if abs(expected) > MAX_CENTS:
        raise ReconciliationOverflowError(
            f"{rule} at {location or '/'}: expected value {expected} exceeds ±{MAX_CENTS} cents"
        )
    if policy.within_tolerance(expected, actual):
        return None
    return Discrepancy(
        rule=rule,
        expected_cents=expected,
        actual_cents=actual,
        delta_cents=actual - expected,
        location=location,
        message=message,
    )


def _amount(field: ExtractedField[int]) -> int | None:
    return field.value if field.status.has_value else None


def _check_line_items(
    result: ExtractionResult, policy: ReconciliationPolicy
) -> list[Discrepancy] | None:
    if not result.line_items:
        return None
    found: list[Discrepancy] = []
    for index, item in enumerate(result.line_items):
        expected = policy.expected_line_amount(item.quantity, item.unit_price_cents)
        discrepancy = _compare(
            policy,
            ReconciliationRule.LINE_ITEM_AMOUNT,
            expected=expected,
            actual=item.amount_cents,
            location=f"/line_items/{index}/amount_cents",
            message=(
                f"{item.quantity} x {item.unit_price_cents} = {expected} "
                f"({policy.rounding}), extracted {item.amount_cents}"
            ),
        )
        if discrepancy is not None:
            found.append(discrepancy)
    return found


def _check_subtotal(
    result: ExtractionResult, policy: ReconciliationPolicy
) -> list[Discrepancy] | None:
    subtotal = _amount(result.subtotal)
    if not result.line_items or subtotal is None:
        return None
    expected = sum(item.amount_cents for item in result.line_items)
    discrepancy = _compare(
        policy,
        ReconciliationRule.LINE_ITEMS_SUBTOTAL,
        expected=expected,
        actual=subtotal,
        location="/subtotal/value",
        message=f"sum of {len(result.line_items)} line item(s) = {expected}, subtotal {subtotal}",
    )
    return [] if discrepancy is None else [discrepancy]


def _check_tax_rates(
    result: ExtractionResult, policy: ReconciliationPolicy
) -> list[Discrepancy] | None:
    if not result.tax_breakdown:
        return None
    found: list[Discrepancy] = []
    for index, line in enumerate(result.tax_breakdown):
        expected = policy.expected_tax(line.taxable_base_cents, line.rate)
        discrepancy = _compare(
            policy,
            ReconciliationRule.TAX_RATE_APPLICATION,
            expected=expected,
            actual=line.tax_cents,
            location=f"/tax_breakdown/{index}/tax_cents",
            message=(
                f"{line.rate}% of {line.taxable_base_cents} = {expected} "
                f"({policy.rounding}), extracted {line.tax_cents}"
            ),
        )
        if discrepancy is not None:
            found.append(discrepancy)
    return found


def _check_tax_total(
    result: ExtractionResult, policy: ReconciliationPolicy
) -> list[Discrepancy] | None:
    tax_total = _amount(result.tax_total)
    if not result.tax_breakdown or tax_total is None:
        return None
    expected = sum(line.tax_cents for line in result.tax_breakdown)
    discrepancy = _compare(
        policy,
        ReconciliationRule.TAX_BREAKDOWN_TOTAL,
        expected=expected,
        actual=tax_total,
        location="/tax_total/value",
        message=(
            f"sum of {len(result.tax_breakdown)} tax line(s) = {expected}, tax total {tax_total}"
        ),
    )
    return [] if discrepancy is None else [discrepancy]


def _check_grand_total(
    result: ExtractionResult, policy: ReconciliationPolicy
) -> list[Discrepancy] | None:
    subtotal, tax, total = (
        _amount(result.subtotal),
        _amount(result.tax_total),
        _amount(result.total),
    )
    # A discount that is not printed is zero; one that is printed but unreadable is unknown.
    discount = 0 if result.discount.status is FieldStatus.MISSING else _amount(result.discount)
    if subtotal is None or tax is None or total is None or discount is None:
        return None
    expected = policy.expected_total(subtotal, tax, discount)
    discrepancy = _compare(
        policy,
        ReconciliationRule.GRAND_TOTAL,
        expected=expected,
        actual=total,
        location="/total/value",
        message=f"{subtotal} + {tax} - {discount} = {expected}, total {total}",
    )
    return [] if discrepancy is None else [discrepancy]


CHECKS: dict[ReconciliationRule, Check] = {
    ReconciliationRule.LINE_ITEM_AMOUNT: _check_line_items,
    ReconciliationRule.LINE_ITEMS_SUBTOTAL: _check_subtotal,
    ReconciliationRule.TAX_RATE_APPLICATION: _check_tax_rates,
    ReconciliationRule.TAX_BREAKDOWN_TOTAL: _check_tax_total,
    ReconciliationRule.GRAND_TOTAL: _check_grand_total,
}


def reconcile(result: ExtractionResult, policy: ReconciliationPolicy) -> ReconciliationReport:
    """Run every rule in ``policy`` against ``result`` and summarise the outcome."""
    evaluated: set[ReconciliationRule] = set()
    skipped: set[ReconciliationRule] = set()
    discrepancies: list[Discrepancy] = []
    for rule in ReconciliationRule:  # declaration order keeps reports deterministic
        if rule not in policy.rules:
            continue
        found = CHECKS[rule](result, policy)
        if found is None:
            skipped.add(rule)
        else:
            evaluated.add(rule)
            discrepancies.extend(found)
    if not evaluated:
        status = ReconciliationStatus.SKIPPED
    elif discrepancies:
        status = ReconciliationStatus.FAILED
    else:
        status = ReconciliationStatus.PASSED
    return ReconciliationReport(
        status=status,
        policy=policy,
        evaluated_rules=frozenset(evaluated),
        skipped_rules=frozenset(skipped),
        discrepancies=tuple(discrepancies),
    )
