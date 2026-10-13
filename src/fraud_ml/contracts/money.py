"""Money helpers at the contract boundary.

Amounts are exact ``Decimal`` values in the contracts and the API. Inside data frames they are
stored as ``float64`` for the numerical libraries; ``to_decimal`` restores the exact value by
quantising to the currency's minor unit, which is lossless for any amount that respects the
contract bounds (``MAX_AMOUNT`` with at most two decimals is far inside float64's 15-digit
precision).
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

from fraud_ml.contracts.enums import CURRENCY_EXPONENT, Currency

#: Illustrative static conversion factors to a common reference unit, used only so that
#: amount features are comparable across currencies. These are not market FX rates.
CURRENCY_TO_REFERENCE: dict[str, float] = {
    Currency.GBP: 1.0,
    Currency.USD: 0.80,
    Currency.EUR: 0.85,
    Currency.CHF: 0.90,
    Currency.SGD: 0.60,
    Currency.AUD: 0.52,
    Currency.JPY: 0.0055,
}


def quantum(currency: Currency | str) -> Decimal:
    exponent = CURRENCY_EXPONENT[Currency(currency)]
    return Decimal(1).scaleb(-exponent)


def to_decimal(value: float | int | str | Decimal, currency: Currency | str) -> Decimal:
    """Convert a frame value back to an exact contract amount."""
    text = value if isinstance(value, Decimal | str) else repr(float(value))
    return Decimal(text).quantize(quantum(currency), rounding=ROUND_HALF_EVEN)


def to_float(value: Decimal | float | int) -> float:
    return float(value)
