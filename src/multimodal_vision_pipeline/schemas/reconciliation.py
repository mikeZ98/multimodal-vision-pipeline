"""Mathematical post-processing rules and reconciliation outcomes.

Arithmetic is exact: amounts are integer cents and rates/quantities are ``Decimal``.
Rounding to whole cents happens once per derived value, using the policy's mode.
"""

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext
from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from multimodal_vision_pipeline.schemas._base import (
    Cents,
    PipelineModel,
    Quantity,
    SignedCents,
    TaxRate,
)

MAX_TOLERANCE_CENTS = 100
_HUNDRED = Decimal(100)
_PRECISION = 60
"""Decimal context precision; comfortably above MAX_CENTS x max Quantity digits."""

JsonPointer = Annotated[str, StringConstraints(max_length=256, pattern=r"^(/([^~/]|~[01])*)*$")]
"""RFC 6901 JSON Pointer into the extraction result, e.g. ``/line_items/2``."""


class RoundingMode(StrEnum):
    """How a derived amount is rounded to whole cents."""

    HALF_UP = "half_up"
    HALF_EVEN = "half_even"

    @property
    def decimal_rounding(self) -> str:
        return ROUND_HALF_UP if self is RoundingMode.HALF_UP else ROUND_HALF_EVEN


class ReconciliationRule(StrEnum):
    """Arithmetic checks applied to an extraction result."""

    LINE_ITEM_AMOUNT = "line_item_amount"
    """``quantity * unit_price == amount`` for every line item."""
    LINE_ITEMS_SUBTOTAL = "line_items_subtotal"
    """``sum(line_item.amount) == subtotal``."""
    TAX_RATE_APPLICATION = "tax_rate_application"
    """``taxable_base * rate / 100 == tax`` for every tax line."""
    TAX_BREAKDOWN_TOTAL = "tax_breakdown_total"
    """``sum(tax_line.tax) == tax_total``."""
    GRAND_TOTAL = "grand_total"
    """``subtotal + tax_total - discount == total``."""


ALL_RULES = frozenset(ReconciliationRule)


class ReconciliationPolicy(PipelineModel):
    """Which rules run, and how much rounding slack each comparison allows."""

    tolerance_cents: int = Field(default=1, ge=0, le=MAX_TOLERANCE_CENTS)
    rounding: RoundingMode = RoundingMode.HALF_UP
    rules: frozenset[ReconciliationRule] = Field(default=ALL_RULES, min_length=1)

    def to_cents(self, amount: Decimal) -> int:
        """Round an exact amount in cents to a whole number of cents."""
        if not amount.is_finite():
            raise ValueError(f"cannot round non-finite amount {amount}")
        with localcontext(prec=_PRECISION):
            return int(amount.quantize(Decimal(1), rounding=self.rounding.decimal_rounding))

    def expected_line_amount(self, quantity: Quantity, unit_price_cents: Cents) -> int:
        with localcontext(prec=_PRECISION):
            return self.to_cents(quantity * unit_price_cents)

    def expected_tax(self, taxable_base_cents: Cents, rate: TaxRate) -> int:
        with localcontext(prec=_PRECISION):
            return self.to_cents(Decimal(taxable_base_cents) * rate / _HUNDRED)

    @staticmethod
    def expected_total(subtotal_cents: Cents, tax_cents: Cents, discount_cents: Cents) -> int:
        return subtotal_cents + tax_cents - discount_cents

    def within_tolerance(self, expected_cents: int, actual_cents: int) -> bool:
        return abs(actual_cents - expected_cents) <= self.tolerance_cents


class ReconciliationStatus(StrEnum):
    PASSED = "passed"
    """At least one rule ran and none found a discrepancy."""
    FAILED = "failed"
    """At least one discrepancy beyond tolerance."""
    SKIPPED = "skipped"
    """No rule could run (e.g. totals missing from the extraction)."""


class Discrepancy(PipelineModel):
    """An arithmetic mismatch beyond tolerance; ``delta_cents = actual - expected``."""

    rule: ReconciliationRule
    expected_cents: SignedCents
    actual_cents: SignedCents
    delta_cents: SignedCents
    location: JsonPointer = ""
    message: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _check_delta(self) -> Self:
        if self.delta_cents != self.actual_cents - self.expected_cents:
            raise ValueError("delta_cents must equal actual_cents - expected_cents")
        if self.delta_cents == 0:
            raise ValueError("a discrepancy must have a non-zero delta")
        return self


class ReconciliationReport(PipelineModel):
    """Outcome of running a policy's rules against one extraction result.

    Every policy rule is either evaluated or skipped (never both); discrepancies only
    come from evaluated rules and must exceed the policy tolerance.
    """

    status: ReconciliationStatus
    policy: ReconciliationPolicy
    evaluated_rules: frozenset[ReconciliationRule]
    skipped_rules: frozenset[ReconciliationRule] = frozenset()
    discrepancies: tuple[Discrepancy, ...] = Field(default=(), max_length=2048)

    @model_validator(mode="after")
    def _check_outcome(self) -> Self:
        if self.evaluated_rules & self.skipped_rules:
            raise ValueError("a rule cannot be both evaluated and skipped")
        if self.evaluated_rules | self.skipped_rules != self.policy.rules:
            raise ValueError("evaluated and skipped rules must partition the policy rules")
        for d in self.discrepancies:
            if d.rule not in self.evaluated_rules:
                raise ValueError(f"discrepancy for rule {d.rule} that was not evaluated")
            if abs(d.delta_cents) <= self.policy.tolerance_cents:
                raise ValueError(
                    f"delta {d.delta_cents} is within tolerance "
                    f"{self.policy.tolerance_cents} and is not a discrepancy"
                )
        expected = (
            ReconciliationStatus.SKIPPED
            if not self.evaluated_rules
            else ReconciliationStatus.FAILED
            if self.discrepancies
            else ReconciliationStatus.PASSED
        )
        if self.status is not expected:
            raise ValueError(f"status must be {expected} for this outcome, got {self.status}")
        return self

    @property
    def max_abs_delta_cents(self) -> int:
        return max((abs(d.delta_cents) for d in self.discrepancies), default=0)
