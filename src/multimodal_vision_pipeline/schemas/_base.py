"""Shared base model and constrained primitives used by every schema in the package."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

MAX_CENTS = 10**15
"""Upper bound for any monetary amount in minor units (10 trillion major units)."""


class PipelineModel(BaseModel):
    """Immutable, strictly validated base for every domain contract.

    Strict mode means no type coercion: in Python mode values must already have
    the declared type; over the wire use ``model_dump_json``/``model_validate_json``,
    where JSON-native encodings (ISO-8601 dates, UUID strings, decimal strings) apply.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
"""Timezone-aware datetime normalised to UTC; naive datetimes are rejected."""

Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
"""Model-reported confidence in ``[0.0, 1.0]``."""

NormalizedCoord = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
"""Coordinate as a fraction of page width or height, origin at the top-left corner."""

Cents = Annotated[int, Field(ge=0, le=MAX_CENTS)]
"""Non-negative monetary amount in integer minor units (cents); never a float."""

SignedCents = Annotated[int, Field(ge=-MAX_CENTS, le=MAX_CENTS)]
"""Monetary amount in minor units that may be negative (expected values, deltas)."""

TaxRate = Annotated[
    Decimal, Field(ge=0, le=100, max_digits=7, decimal_places=4, allow_inf_nan=False)
]
"""Tax rate as an exact percentage, e.g. ``Decimal("23")`` or ``Decimal("8.875")``."""

Quantity = Annotated[Decimal, Field(gt=0, max_digits=18, decimal_places=6, allow_inf_nan=False)]
"""Strictly positive exact quantity (fractional units such as ``2.5`` kg allowed)."""

CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
"""ISO 4217 alphabetic currency code."""

NonEmptyStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
]

IdempotencyKey = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
]
