"""Reproducible synthetic transaction generator with injected fraud-like patterns.

Everything here is fictional. Accounts, counterparties, devices and amounts are drawn from
illustrative distributions; the fraud patterns are stylised versions of textbook typologies
and the ``fraud_type`` labels are generator conventions, not regulatory definitions.

Design points that matter for the ML work:

* the same ``seed`` and config always produce the same frame (NumPy ``Generator`` only);
* legitimate behaviour deliberately overlaps with fraud (legit large payments, legit new
  beneficiaries, night-owl accounts, travel, new phones) so the labels are not trivially
  separable and no single feature is a copy of the label;
* the generator emits a *raw event log* and then derives every history field through the
  same point-in-time routine the adapter uses (``compute_history_features``), so the
  training data cannot contain information from later transactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fraud_ml.contracts.enums import (
    SCHEMA_VERSION,
    Channel,
    Currency,
    CustomerSegment,
    FraudType,
    IpRiskBand,
    TransactionType,
)
from fraud_ml.contracts.money import CURRENCY_TO_REFERENCE
from fraud_ml.features.velocity import compute_history_features

Rng = np.random.Generator

COUNTRY_WEIGHTS: dict[str, float] = {
    "GB": 0.42, "US": 0.14, "DE": 0.08, "FR": 0.06, "SG": 0.05, "AU": 0.05,
    "CH": 0.04, "JP": 0.04, "NL": 0.04, "IE": 0.04, "ES": 0.04,
}  # fmt: skip
COUNTRY_CURRENCY: dict[str, Currency] = {
    "GB": Currency.GBP, "US": Currency.USD, "DE": Currency.EUR, "FR": Currency.EUR,
    "SG": Currency.SGD, "AU": Currency.AUD, "CH": Currency.CHF, "JP": Currency.JPY,
    "NL": Currency.EUR, "IE": Currency.EUR, "ES": Currency.EUR,
}  # fmt: skip
FOREIGN_COUNTRIES: tuple[str, ...] = (
    "GB", "US", "DE", "FR", "SG", "AU", "CH", "JP", "NL", "IE", "ES", "IT", "PT", "PL", "CA",
    "HK", "AE", "TR", "BR", "NG", "IN", "MX", "ZA",
)  # fmt: skip
MERCHANT_CATEGORIES: tuple[str, ...] = (
    "GROCERY", "FUEL", "RESTAURANT", "RETAIL", "TRAVEL", "DIGITAL_GOODS", "UTILITIES",
    "ELECTRONICS", "ENTERTAINMENT", "PHARMACY", "GAMBLING", "CRYPTO_EXCHANGE", "MONEY_TRANSFER",
)  # fmt: skip
SEGMENT_PROFILE: dict[CustomerSegment, tuple[float, float, float]] = {
    # (daily activity rate, median amount in reference units, weight)
    CustomerSegment.RETAIL: (0.35, 45.0, 0.75),
    CustomerSegment.PREMIUM: (0.55, 140.0, 0.15),
    CustomerSegment.BUSINESS: (1.30, 420.0, 0.10),
}
TYPE_WEIGHTS: dict[TransactionType, float] = {
    TransactionType.TRANSFER: 0.28,
    TransactionType.PAYMENT: 0.22,
    TransactionType.CARD_PURCHASE: 0.30,
    TransactionType.DIRECT_DEBIT: 0.08,
    TransactionType.WITHDRAWAL: 0.05,
    TransactionType.DEPOSIT: 0.06,
    TransactionType.REFUND: 0.01,
}
CHANNELS_BY_TYPE: dict[TransactionType, tuple[tuple[Channel, ...], tuple[float, ...]]] = {
    TransactionType.TRANSFER: (
        (Channel.MOBILE, Channel.ONLINE, Channel.API, Channel.BRANCH), (0.45, 0.40, 0.10, 0.05)
    ),
    TransactionType.PAYMENT: ((Channel.ONLINE, Channel.MOBILE, Channel.API), (0.50, 0.40, 0.10)),
    TransactionType.CARD_PURCHASE: ((Channel.CARD, Channel.ONLINE), (0.80, 0.20)),
    TransactionType.DIRECT_DEBIT: ((Channel.BATCH,), (1.0,)),
    TransactionType.WITHDRAWAL: ((Channel.ATM, Channel.BRANCH), (0.85, 0.15)),
    TransactionType.DEPOSIT: (
        (Channel.BRANCH, Channel.BATCH, Channel.ONLINE, Channel.ATM), (0.35, 0.35, 0.20, 0.10)
    ),
    TransactionType.REFUND: ((Channel.CARD, Channel.BATCH), (0.7, 0.3)),
}  # fmt: skip
IP_BANDS: tuple[str, ...] = tuple(IpRiskBand)
NORMAL_IP_WEIGHTS = (0.86, 0.10, 0.02, 0.02)
RISKY_IP_WEIGHTS = (0.30, 0.35, 0.30, 0.05)
NO_IP_CHANNELS: frozenset[str] = frozenset(
    {Channel.BRANCH, Channel.ATM, Channel.BATCH, Channel.CARD}
)
STRUCTURING_THRESHOLDS = (1_000.0, 5_000.0, 10_000.0)


@dataclass(frozen=True)
class GeneratorConfig:
    transactions: int = 50_000
    fraud_rate: float = 0.01
    seed: int = 42
    start: date = date(2025, 1, 1)
    days: int = 180
    accounts: int | None = None

    def __post_init__(self) -> None:
        if self.transactions < 100:
            raise ValueError("transactions must be at least 100")
        if not 0 <= self.fraud_rate <= 0.5:
            raise ValueError("fraud_rate must lie in [0, 0.5]")
        if self.days < 7:
            raise ValueError("days must be at least 7")

    @property
    def n_accounts(self) -> int:
        if self.accounts is not None:
            return max(10, self.accounts)
        return max(50, self.transactions // 40)


@dataclass(frozen=True)
class GenerationSummary:
    transactions: int
    fraud_rows: int
    fraud_rate: float
    accounts: int
    start: str
    end: str
    fraud_type_counts: dict[str, int] = field(default_factory=dict)
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "transactions": self.transactions,
            "fraud_rows": self.fraud_rows,
            "fraud_rate": round(self.fraud_rate, 6),
            "accounts": self.accounts,
            "start": self.start,
            "end": self.end,
            "fraud_type_counts": self.fraud_type_counts,
            "seed": self.seed,
        }


@dataclass
class _Account:
    account_id: str
    country: str
    currency: Currency
    segment: CustomerSegment
    open_date: datetime
    rate: float
    median_amount: float  # in account currency
    preferred_hour: float
    hour_sigma: float
    cross_border_share: float
    counterparties: list[str]
    devices: list[str]
    channel_bias: TransactionType  # dominant transaction style
    active_windows: list[tuple[float, float]] | None  # fraction-of-window spans for bursty accounts


def _choice(
    rng: Rng, options: tuple[Any, ...] | list[Any], weights: tuple[float, ...] | list[float]
) -> Any:
    probabilities = np.asarray(weights, dtype="float64")
    return options[int(rng.choice(len(options), p=probabilities / probabilities.sum()))]


def _make_accounts(cfg: GeneratorConfig, rng: Rng) -> list[_Account]:
    start = datetime.combine(cfg.start, datetime.min.time(), tzinfo=UTC)
    countries = list(COUNTRY_WEIGHTS)
    country_p = np.asarray(list(COUNTRY_WEIGHTS.values()))
    segments = list(SEGMENT_PROFILE)
    segment_p = np.asarray([SEGMENT_PROFILE[s][2] for s in segments])
    merchants = [f"CP-SYN-M{i:05d}" for i in range(400)]
    accounts: list[_Account] = []
    for i in range(cfg.n_accounts):
        country = countries[int(rng.choice(len(countries), p=country_p / country_p.sum()))]
        currency = COUNTRY_CURRENCY[country]
        segment = segments[int(rng.choice(len(segments), p=segment_p / segment_p.sum()))]
        rate_mean, median_ref, _ = SEGMENT_PROFILE[segment]
        rate = float(rng.lognormal(np.log(rate_mean), 0.6))
        median_amount = (
            float(rng.lognormal(np.log(median_ref), 0.5)) / CURRENCY_TO_REFERENCE[currency]
        )
        # Account age: most accounts are years old; a few opened during the window.
        if rng.random() < 0.06:
            open_offset = timedelta(days=float(rng.uniform(0, cfg.days * 0.8)))
        else:
            open_offset = -timedelta(days=float(rng.exponential(900.0)) + 10.0)
        night_owl = rng.random() < 0.08
        preferred_hour = (
            float(rng.normal(1.5, 1.0)) % 24 if night_owl else float(rng.normal(14.0, 2.5)) % 24
        )
        intl = rng.random()
        cross_border_share = 0.40 if intl < 0.05 else (0.15 if intl < 0.30 else 0.02)
        known = [f"CP-SYN-{i:06d}-{j:02d}" for j in range(int(rng.integers(4, 20)))]
        known += list(rng.choice(merchants, size=int(rng.integers(3, 12)), replace=False))
        devices = [f"DEV-SYN-{i:06d}-{j}" for j in range(int(rng.integers(1, 3)))]
        channel_bias = _choice(rng, list(TYPE_WEIGHTS), list(TYPE_WEIGHTS.values()))
        windows = None
        if rng.random() < 0.07:  # bursty account with legitimate quiet periods
            a, b = sorted(rng.uniform(0, 1, size=2))
            windows = [(a, min(1.0, a + 0.12)), (b, min(1.0, b + 0.12))]
        accounts.append(
            _Account(
                account_id=f"ACC-SYN-{i:06d}",
                country=country,
                currency=currency,
                segment=segment,
                open_date=start + open_offset,
                rate=rate,
                median_amount=median_amount,
                preferred_hour=preferred_hour,
                hour_sigma=float(rng.uniform(2.0, 4.0)),
                cross_border_share=cross_border_share,
                counterparties=known,
                devices=devices,
                channel_bias=channel_bias,
                active_windows=windows,
            )
        )
    return accounts


def _timestamp(
    cfg: GeneratorConfig, acct: _Account, rng: Rng, fraction: float | None = None
) -> datetime:
    start = datetime.combine(cfg.start, datetime.min.time(), tzinfo=UTC)
    if fraction is None:
        if acct.active_windows:
            lo, hi = acct.active_windows[int(rng.integers(len(acct.active_windows)))]
            fraction = float(rng.uniform(lo, hi))
        else:
            fraction = float(rng.uniform(0, 1))
    day = int(fraction * cfg.days)
    hour = float(rng.normal(acct.preferred_hour, acct.hour_sigma)) % 24
    ts = start + timedelta(days=day, hours=hour, minutes=float(rng.uniform(0, 60)))
    if ts.weekday() >= 5 and rng.random() < 0.35:  # weekends are quieter
        ts -= timedelta(days=2)
    ts = max(ts, acct.open_date)
    return ts.replace(second=int(rng.integers(0, 60)), microsecond=0)


def _amount(acct: _Account, rng: Rng, multiplier: float = 1.0, sigma: float = 0.9) -> float:
    value = float(rng.lognormal(np.log(acct.median_amount), sigma)) * multiplier
    exponent = 0 if acct.currency is Currency.JPY else 2
    return max(round(value, exponent), 1.0 if exponent == 0 else 0.01)


def _ip_band(rng: Rng, channel: str, risky: bool = False) -> str:
    if channel in NO_IP_CHANNELS:
        return IpRiskBand.UNKNOWN
    return str(_choice(rng, IP_BANDS, RISKY_IP_WEIGHTS if risky else NORMAL_IP_WEIGHTS))


def _normal_row(cfg: GeneratorConfig, acct: _Account, rng: Rng, counter: int) -> dict[str, Any]:
    txn_type: TransactionType = _choice(rng, list(TYPE_WEIGHTS), list(TYPE_WEIGHTS.values()))
    if rng.random() < 0.35:
        txn_type = acct.channel_bias
    channels, weights = CHANNELS_BY_TYPE[txn_type]
    channel: Channel = _choice(rng, channels, weights)
    multiplier = 8.0 if rng.random() < 0.015 else 1.0  # occasional legitimate large payment
    amount = _amount(acct, rng, multiplier)
    counterparty: str | None
    beneficiary: str | None
    merchant_category: str | None = None
    if txn_type is TransactionType.CARD_PURCHASE:
        merchants = [c for c in acct.counterparties if c.startswith("CP-SYN-M")]
        counterparty = (
            str(rng.choice(merchants))
            if merchants and rng.random() < 0.9
            else f"CP-SYN-M{int(rng.integers(0, 400)):05d}"
        )
        merchant_category = str(rng.choice(MERCHANT_CATEGORIES[:10]))
        beneficiary = (
            acct.country
            if rng.random() > acct.cross_border_share
            else str(rng.choice(FOREIGN_COUNTRIES))
        )
    elif txn_type in (TransactionType.WITHDRAWAL, TransactionType.DEPOSIT):
        counterparty = None
        beneficiary = None
    else:
        if rng.random() < 0.92:
            counterparty = str(
                rng.choice([c for c in acct.counterparties if not c.startswith("CP-SYN-M")])
            )
        else:  # legitimate brand-new beneficiary
            counterparty = f"CP-SYN-N{counter:07d}"
        beneficiary = (
            acct.country
            if rng.random() > acct.cross_border_share
            else str(rng.choice(FOREIGN_COUNTRIES))
        )
        if txn_type is TransactionType.DIRECT_DEBIT:
            beneficiary = acct.country
    device: str | None = (
        str(rng.choice(acct.devices)) if rng.random() < 0.95 else f"DEV-SYN-NEW-{counter:07d}"
    )
    if channel in NO_IP_CHANNELS and channel is not Channel.CARD:
        device = None
    country = acct.country if rng.random() < 0.965 else str(rng.choice(FOREIGN_COUNTRIES))
    return {
        "account_id": acct.account_id,
        "transaction_timestamp": _timestamp(cfg, acct, rng),
        "amount": amount,
        "currency": acct.currency.value,
        "transaction_type": txn_type.value,
        "channel": channel.value,
        "country": country,
        "counterparty_id": counterparty,
        "beneficiary_country": beneficiary,
        "merchant_category": merchant_category,
        "device_id_hash": device,
        "ip_risk_band": _ip_band(rng, channel),
        "customer_segment": acct.segment.value,
        "is_fraud": False,
        "fraud_type": FraudType.NONE.value,
    }


# --- fraud patterns ----------------------------------------------------------------------------

PATTERNS: tuple[tuple[str, FraudType, float], ...] = (
    ("rapid_velocity", FraudType.CARD_NOT_PRESENT, 0.14),
    ("high_amount", FraudType.ACCOUNT_TAKEOVER, 0.12),
    ("new_beneficiary_large", FraudType.ACCOUNT_TAKEOVER, 0.12),
    ("cross_border", FraudType.CARD_NOT_PRESENT, 0.10),
    ("dormant_active", FraudType.ACCOUNT_TAKEOVER, 0.08),
    ("many_counterparties", FraudType.MULE_ACTIVITY, 0.10),
    ("structuring", FraudType.STRUCTURING, 0.10),
    ("behaviour_shift", FraudType.ACCOUNT_TAKEOVER, 0.10),
    ("unusual_hour", FraudType.CARD_NOT_PRESENT, 0.06),
    ("rapid_movement", FraudType.RAPID_MOVEMENT, 0.08),
)


def _fraud_base(
    acct: _Account, ts: datetime, rng: Rng, *, counter: int, risky: bool, new_device: float
) -> dict[str, Any]:
    device = (
        f"DEV-SYN-F{counter:07d}" if rng.random() < new_device else str(rng.choice(acct.devices))
    )
    return {
        "account_id": acct.account_id,
        "transaction_timestamp": ts,
        "currency": acct.currency.value,
        "country": acct.country,
        "merchant_category": None,
        "device_id_hash": device,
        "customer_segment": acct.segment.value,
        "is_fraud": True,
        "_risky": risky,
    }


def _finish(row: dict[str, Any], rng: Rng, fraud_type: FraudType) -> dict[str, Any]:
    risky = row.pop("_risky")
    row["ip_risk_band"] = _ip_band(rng, row["channel"], risky=risky)
    if row["channel"] in NO_IP_CHANNELS and row["channel"] != Channel.CARD:
        row["device_id_hash"] = None
    row["fraud_type"] = fraud_type.value
    return row


def _pattern_rows(
    name: str, fraud_type: FraudType, cfg: GeneratorConfig, acct: _Account, rng: Rng, counter: int
) -> list[dict[str, Any]]:
    mild = rng.random() < 0.15  # harder, less extreme instance of the pattern
    rows: list[dict[str, Any]] = []
    base_ts = _timestamp(cfg, acct, rng)
    foreign = str(rng.choice([c for c in FOREIGN_COUNTRIES if c != acct.country]))

    if name == "rapid_velocity":
        n = int(rng.integers(3, 7)) if mild else int(rng.integers(4, 10))
        if rng.random() < 0.4:
            base_ts = base_ts.replace(hour=int(rng.integers(0, 5)))
        offsets = np.sort(rng.uniform(0, 20 if not mild else 50, size=n))
        for k in range(n):
            row = _fraud_base(
                acct,
                base_ts + timedelta(minutes=float(offsets[k])),
                rng,
                counter=counter + k,
                risky=True,
                new_device=0.6,
            )
            row.update(
                transaction_type=TransactionType.CARD_PURCHASE.value
                if rng.random() < 0.7
                else TransactionType.PAYMENT.value,
                channel=Channel.ONLINE.value,
                amount=_amount(acct, rng, float(rng.uniform(0.5, 3.0))),
                counterparty_id=f"CP-SYN-F{counter + k:07d}",
                beneficiary_country=acct.country if rng.random() < 0.6 else foreign,
                merchant_category=str(
                    rng.choice(("DIGITAL_GOODS", "ELECTRONICS", "GAMBLING", "RETAIL"))
                ),
            )
            rows.append(row)
    elif name == "high_amount":
        row = _fraud_base(acct, base_ts, rng, counter=counter, risky=True, new_device=0.6)
        mult = float(rng.uniform(2.5, 4.0)) if mild else float(rng.lognormal(np.log(9.0), 0.5))
        row.update(
            transaction_type=TransactionType.TRANSFER.value,
            channel=str(_choice(rng, (Channel.ONLINE, Channel.MOBILE), (0.6, 0.4))),
            amount=_amount(acct, rng, max(mult, 2.5), sigma=0.3),
            counterparty_id=f"CP-SYN-F{counter:07d}"
            if rng.random() < 0.7
            else str(rng.choice(acct.counterparties)),
            beneficiary_country=acct.country if rng.random() < 0.7 else foreign,
        )
        rows.append(row)
    elif name == "new_beneficiary_large":
        n = 1 if mild else int(rng.integers(1, 3))
        for k in range(n):
            row = _fraud_base(
                acct,
                base_ts + timedelta(minutes=float(rng.uniform(0, 90)) * k),
                rng,
                counter=counter + k,
                risky=True,
                new_device=0.5,
            )
            row.update(
                transaction_type=TransactionType.TRANSFER.value
                if rng.random() < 0.8
                else TransactionType.PAYMENT.value,
                channel=str(
                    _choice(rng, (Channel.ONLINE, Channel.MOBILE, Channel.API), (0.5, 0.4, 0.1))
                ),
                amount=_amount(
                    acct,
                    rng,
                    float(rng.uniform(2.0, 4.0) if mild else rng.uniform(3.0, 12.0)),
                    sigma=0.3,
                ),
                counterparty_id=f"CP-SYN-F{counter + k:07d}",
                beneficiary_country=acct.country if rng.random() < 0.7 else foreign,
            )
            rows.append(row)
    elif name == "cross_border":
        n = int(rng.integers(1, 4))
        for k in range(n):
            row = _fraud_base(
                acct,
                base_ts + timedelta(minutes=float(rng.uniform(5, 240)) * k),
                rng,
                counter=counter + k,
                risky=True,
                new_device=0.4,
            )
            card = rng.random() < 0.5
            row.update(
                transaction_type=TransactionType.CARD_PURCHASE.value
                if card
                else TransactionType.PAYMENT.value,
                channel=Channel.ONLINE.value,
                amount=_amount(
                    acct, rng, float(rng.uniform(1.0, 3.0) if mild else rng.uniform(1.5, 6.0))
                ),
                counterparty_id=f"CP-SYN-F{counter + k:07d}",
                beneficiary_country=foreign,
                merchant_category=str(
                    rng.choice(("TRAVEL", "ELECTRONICS", "DIGITAL_GOODS", "MONEY_TRANSFER"))
                )
                if card
                else None,
            )
            if rng.random() < 0.4:
                row["country"] = foreign
            rows.append(row)
    elif name == "dormant_active":
        n = 1 if mild else int(rng.integers(1, 4))
        for k in range(n):
            row = _fraud_base(
                acct,
                base_ts + timedelta(minutes=float(rng.uniform(5, 60)) * k),
                rng,
                counter=counter + k,
                risky=True,
                new_device=0.7,
            )
            row.update(
                transaction_type=TransactionType.TRANSFER.value,
                channel=str(_choice(rng, (Channel.ONLINE, Channel.MOBILE), (0.6, 0.4))),
                amount=_amount(
                    acct,
                    rng,
                    float(rng.uniform(1.5, 3.0) if mild else rng.uniform(2.0, 10.0)),
                    sigma=0.4,
                ),
                counterparty_id=f"CP-SYN-F{counter + k:07d}",
                beneficiary_country=acct.country if rng.random() < 0.6 else foreign,
                _dormant=True,
            )
            rows.append(row)
    elif name == "many_counterparties":
        n = int(rng.integers(4, 6)) if mild else int(rng.integers(5, 11))
        span_hours = float(rng.uniform(2, 12))
        offsets = np.sort(rng.uniform(0, span_hours * 60, size=n))
        for k in range(n):
            row = _fraud_base(
                acct,
                base_ts + timedelta(minutes=float(offsets[k])),
                rng,
                counter=counter + k,
                risky=False,
                new_device=0.2,
            )
            row.update(
                transaction_type=TransactionType.TRANSFER.value,
                channel=str(
                    _choice(rng, (Channel.MOBILE, Channel.ONLINE, Channel.API), (0.5, 0.35, 0.15))
                ),
                amount=_amount(acct, rng, float(rng.uniform(0.8, 2.5)), sigma=0.3),
                counterparty_id=f"CP-SYN-F{counter + k:07d}",
                beneficiary_country=acct.country if rng.random() < 0.5 else foreign,
            )
            rows.append(row)
    elif name == "structuring":
        n = int(rng.integers(2, 4)) if mild else int(rng.integers(3, 7))
        threshold = float(rng.choice(STRUCTURING_THRESHOLDS)) / CURRENCY_TO_REFERENCE[acct.currency]
        exponent = 0 if acct.currency is Currency.JPY else 2
        cps = [f"CP-SYN-F{counter:07d}", f"CP-SYN-F{counter + 1:07d}"]
        for k in range(n):
            ts = base_ts + timedelta(hours=float(rng.uniform(2, 72)) * k / max(n - 1, 1))
            row = _fraud_base(acct, ts, rng, counter=counter + k, risky=False, new_device=0.1)
            row.update(
                transaction_type=TransactionType.TRANSFER.value
                if rng.random() < 0.7
                else TransactionType.DEPOSIT.value,
                channel=str(
                    _choice(rng, (Channel.ONLINE, Channel.BRANCH, Channel.MOBILE), (0.5, 0.3, 0.2))
                ),
                amount=round(threshold * float(rng.uniform(0.86, 0.985)), exponent),
                counterparty_id=str(rng.choice(cps)),
                beneficiary_country=acct.country,
            )
            if row["transaction_type"] == TransactionType.DEPOSIT.value:
                row["counterparty_id"] = None
                row["beneficiary_country"] = None
            rows.append(row)
    elif name == "behaviour_shift":
        n = int(rng.integers(2, 5))
        base_ts = base_ts.replace(hour=int((acct.preferred_hour + 12 + rng.uniform(-3, 3)) % 24))
        unusual_channel = (
            Channel.API
            if acct.channel_bias in (TransactionType.CARD_PURCHASE, TransactionType.TRANSFER)
            else Channel.ONLINE
        )
        for k in range(n):
            row = _fraud_base(
                acct,
                base_ts + timedelta(minutes=float(rng.uniform(5, 120)) * k),
                rng,
                counter=counter + k,
                risky=True,
                new_device=0.85,
            )
            row.update(
                transaction_type=TransactionType.TRANSFER.value
                if rng.random() < 0.7
                else TransactionType.PAYMENT.value,
                channel=unusual_channel.value,
                amount=_amount(
                    acct, rng, float(rng.uniform(1.2, 2.5) if mild else rng.uniform(1.5, 5.0))
                ),
                counterparty_id=f"CP-SYN-F{counter + k:07d}"
                if rng.random() < 0.6
                else str(rng.choice(acct.counterparties)),
                beneficiary_country=acct.country if rng.random() < 0.7 else foreign,
            )
            rows.append(row)
    elif name == "unusual_hour":
        n = int(rng.integers(1, 4))
        base_ts = base_ts.replace(hour=int(rng.integers(1, 5)), minute=int(rng.integers(0, 60)))
        for k in range(n):
            row = _fraud_base(
                acct,
                base_ts + timedelta(minutes=float(rng.uniform(3, 40)) * k),
                rng,
                counter=counter + k,
                risky=True,
                new_device=0.4,
            )
            row.update(
                transaction_type=TransactionType.CARD_PURCHASE.value,
                channel=Channel.ONLINE.value,
                amount=_amount(
                    acct, rng, float(rng.uniform(1.0, 2.0) if mild else rng.uniform(1.0, 4.0))
                ),
                counterparty_id=f"CP-SYN-F{counter + k:07d}",
                beneficiary_country=acct.country if rng.random() < 0.5 else foreign,
                merchant_category=str(
                    rng.choice(("DIGITAL_GOODS", "GAMBLING", "ELECTRONICS", "CRYPTO_EXCHANGE"))
                ),
            )
            rows.append(row)
    elif name == "rapid_movement":
        inflow_amount = _amount(
            acct, rng, float(rng.uniform(3.0, 6.0) if mild else rng.uniform(5.0, 20.0)), sigma=0.3
        )
        inflow = _fraud_base(acct, base_ts, rng, counter=counter, risky=False, new_device=0.0)
        inflow.update(
            transaction_type=TransactionType.DEPOSIT.value,
            channel=str(
                _choice(rng, (Channel.BATCH, Channel.ONLINE, Channel.BRANCH), (0.5, 0.3, 0.2))
            ),
            amount=inflow_amount,
            counterparty_id=None,
            beneficiary_country=None,
            is_fraud=False,  # the inbound credit itself is not labelled; the rapid outflow is
        )
        rows.append(inflow)
        n = int(rng.integers(2, 5))
        shares = rng.dirichlet(np.ones(n)) * float(rng.uniform(0.88, 0.99))
        offsets = np.sort(rng.uniform(10, 120 if not mild else 360, size=n))
        exponent = 0 if acct.currency is Currency.JPY else 2
        for k in range(n):
            row = _fraud_base(
                acct,
                base_ts + timedelta(minutes=float(offsets[k])),
                rng,
                counter=counter + 1 + k,
                risky=False,
                new_device=0.3,
            )
            row.update(
                transaction_type=TransactionType.TRANSFER.value,
                channel=str(
                    _choice(rng, (Channel.ONLINE, Channel.MOBILE, Channel.API), (0.45, 0.35, 0.2))
                ),
                amount=max(round(inflow_amount * float(shares[k]), exponent), 1.0),
                counterparty_id=f"CP-SYN-F{counter + 1 + k:07d}",
                beneficiary_country=acct.country if rng.random() < 0.6 else foreign,
            )
            rows.append(row)
    else:  # pragma: no cover - guarded by PATTERNS
        raise ValueError(name)

    finished = []
    for row in rows:
        label_type = fraud_type if row["is_fraud"] else FraudType.NONE
        finished.append(_finish(row, rng, label_type))
    return finished


def generate_dataset(cfg: GeneratorConfig) -> tuple[pd.DataFrame, GenerationSummary]:
    """Generate a labelled contract frame ordered by time, plus a summary."""
    rng = np.random.default_rng(cfg.seed)
    accounts = _make_accounts(cfg, rng)
    n_fraud_target = round(cfg.transactions * cfg.fraud_rate)
    n_normal_target = cfg.transactions - n_fraud_target

    # Legitimate activity, allocated to accounts in proportion to their activity rate.
    rates = np.asarray([a.rate for a in accounts])
    allocation = rng.multinomial(int(n_normal_target * 1.03) + 50, rates / rates.sum())
    counter = 0
    rows: list[dict[str, Any]] = []
    for acct, count in zip(accounts, allocation, strict=True):
        for _ in range(int(count)):
            rows.append(_normal_row(cfg, acct, rng, counter))
            counter += 1

    # Fraud-like events until the target number of labelled rows is reached.
    pattern_names = [p[0] for p in PATTERNS]
    pattern_p = np.asarray([p[2] for p in PATTERNS])
    fraud_rows: list[dict[str, Any]] = []
    dormant_removals: list[tuple[str, datetime]] = []
    labelled = 0
    guard = 0
    while labelled < n_fraud_target and guard < n_fraud_target * 10 + 100:
        guard += 1
        idx = int(rng.choice(len(PATTERNS), p=pattern_p / pattern_p.sum()))
        name, fraud_type, _ = PATTERNS[idx]
        acct = accounts[int(rng.integers(len(accounts)))]
        new_rows = _pattern_rows(name, fraud_type, cfg, acct, rng, counter)
        counter += len(new_rows) + 1
        for row in new_rows:
            if row.pop("_dormant", False):
                dormant_removals.append((acct.account_id, row["transaction_timestamp"]))
        fraud_rows.extend(new_rows)
        labelled += sum(1 for r in new_rows if r["is_fraud"])
    _ = pattern_names

    normal = pd.DataFrame(rows)
    # Dormant pattern: remove the account's legitimate activity in the 60 days before the event.
    if dormant_removals:
        drop = np.zeros(len(normal), dtype=bool)
        for account_id, ts in dormant_removals:
            gap = float(rng.uniform(45, 120))
            mask = (
                (normal["account_id"] == account_id)
                & (normal["transaction_timestamp"] >= ts - timedelta(days=gap))
                & (normal["transaction_timestamp"] < ts + timedelta(hours=2))
            )
            drop |= mask.to_numpy()
        normal = normal.loc[~drop]
    fraud = pd.DataFrame(fraud_rows)
    frame = pd.concat([normal, fraud], ignore_index=True)

    # Trim surplus legitimate rows to hit the requested size exactly.
    surplus = len(frame) - cfg.transactions
    if surplus > 0:
        normal_positions = np.flatnonzero(~frame["is_fraud"].to_numpy())
        drop_positions = rng.choice(normal_positions, size=surplus, replace=False)
        frame = frame.drop(index=frame.index[drop_positions])

    frame["transaction_timestamp"] = pd.to_datetime(frame["transaction_timestamp"], utc=True)
    frame = frame.sort_values(
        ["transaction_timestamp", "account_id"], kind="mergesort"
    ).reset_index(drop=True)
    frame.insert(0, "transaction_id", [f"TXN-SYN-{i:08d}" for i in range(1, len(frame) + 1)])
    frame.insert(0, "schema_version", SCHEMA_VERSION)
    open_dates = {a.account_id: a.open_date for a in accounts}
    frame["account_open_date"] = pd.to_datetime(frame["account_id"].map(open_dates), utc=True)
    frame = compute_history_features(frame)
    frame = frame.drop(columns=["account_open_date"])
    frame["amount"] = frame["amount"].astype("float64")
    for column in ("is_fraud", "is_weekend", "is_new_beneficiary", "is_new_device"):
        frame[column] = frame[column].astype(bool)

    ordered_columns = [
        "schema_version", "transaction_id", "account_id", "transaction_timestamp", "amount",
        "currency", "transaction_type", "channel", "country", "counterparty_id",
        "beneficiary_country", "is_weekend", "account_age_days", "transactions_1h",
        "transactions_24h", "amount_1h", "amount_24h", "unique_counterparties_24h",
        "days_since_last_transaction", "account_median_amount_30d", "account_mean_amount_30d",
        "inflow_amount_24h", "merchant_category", "device_id_hash", "ip_risk_band",
        "customer_segment", "is_new_beneficiary", "is_new_device", "is_fraud", "fraud_type",
    ]  # fmt: skip
    frame = frame.loc[:, ordered_columns]
    counts = frame.loc[frame["is_fraud"], "fraud_type"].value_counts().to_dict()
    summary = GenerationSummary(
        transactions=len(frame),
        fraud_rows=int(frame["is_fraud"].sum()),
        fraud_rate=float(frame["is_fraud"].mean()),
        accounts=int(frame["account_id"].nunique()),
        start=frame["transaction_timestamp"].iloc[0].isoformat(),
        end=frame["transaction_timestamp"].iloc[-1].isoformat(),
        fraud_type_counts={str(k): int(v) for k, v in sorted(counts.items())},
        seed=cfg.seed,
    )
    return frame, summary
