"""Closed vocabularies shared by the contracts, the generator, the features and the API.

Values deliberately mirror the ``txn-data-pipeline`` curated vocabulary where the concepts
overlap, so that curated output can be adapted with a thin mapping (see
``fraud_ml.adapters.txn_pipeline``). Fraud types are synthetic generator labels only and do
not correspond to any regulatory or industry typology.
"""

from __future__ import annotations

from enum import StrEnum

#: Version of the transaction feature input contract (``fraud_ml.contracts.transaction``).
SCHEMA_VERSION = "fraud-transaction/1"
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({SCHEMA_VERSION})

#: Version of the score response contract (``fraud_ml.contracts.score``).
SCORE_SCHEMA_VERSION = "fraud-score/1"


class Currency(StrEnum):
    GBP = "GBP"
    USD = "USD"
    EUR = "EUR"
    JPY = "JPY"
    CHF = "CHF"
    SGD = "SGD"
    AUD = "AUD"


#: Minor-unit exponent per currency; used to validate amount scale.
CURRENCY_EXPONENT: dict[Currency, int] = {
    Currency.GBP: 2,
    Currency.USD: 2,
    Currency.EUR: 2,
    Currency.JPY: 0,
    Currency.CHF: 2,
    Currency.SGD: 2,
    Currency.AUD: 2,
}


class TransactionType(StrEnum):
    TRANSFER = "TRANSFER"
    PAYMENT = "PAYMENT"
    DIRECT_DEBIT = "DIRECT_DEBIT"
    CARD_PURCHASE = "CARD_PURCHASE"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    REFUND = "REFUND"


class Channel(StrEnum):
    ONLINE = "ONLINE"
    MOBILE = "MOBILE"
    BRANCH = "BRANCH"
    ATM = "ATM"
    API = "API"
    CARD = "CARD"
    BATCH = "BATCH"


class IpRiskBand(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class CustomerSegment(StrEnum):
    RETAIL = "RETAIL"
    PREMIUM = "PREMIUM"
    BUSINESS = "BUSINESS"


class FraudType(StrEnum):
    """Synthetic generator label describing which pattern produced a fraud-like row.

    These are generator conventions for evaluation only; they are not regulatory
    definitions and never enter the feature matrix.
    """

    NONE = "NONE"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
    CARD_NOT_PRESENT = "CARD_NOT_PRESENT"
    MULE_ACTIVITY = "MULE_ACTIVITY"
    RAPID_MOVEMENT = "RAPID_MOVEMENT"
    STRUCTURING = "STRUCTURING"


class RiskBand(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DecisionSupport(StrEnum):
    """Recommended handling for a downstream human or control process.

    There is intentionally no ``BLOCK`` value: the system prioritises review, it does not
    decide on its own.
    """

    PASS = "PASS"  # noqa: S105  # nosec B105 - decision label, not a credential
    REVIEW = "REVIEW"
    PRIORITY_REVIEW = "PRIORITY_REVIEW"


class ReasonCode(StrEnum):
    """Human-readable risk indicators attached to a score.

    Reason codes are rule-based indicators derived from the input features. They describe
    *which risk signals are present*; they are not causal explanations of the model output.
    """

    HIGH_AMOUNT_VS_HISTORY = "HIGH_AMOUNT_VS_HISTORY"
    HIGH_VELOCITY_1H = "HIGH_VELOCITY_1H"
    HIGH_VELOCITY_24H = "HIGH_VELOCITY_24H"
    NEW_BENEFICIARY = "NEW_BENEFICIARY"
    CROSS_BORDER = "CROSS_BORDER"
    UNUSUAL_HOUR = "UNUSUAL_HOUR"
    DORMANT_ACCOUNT_REACTIVATED = "DORMANT_ACCOUNT_REACTIVATED"
    MANY_COUNTERPARTIES_24H = "MANY_COUNTERPARTIES_24H"
    NEAR_THRESHOLD_AMOUNT = "NEAR_THRESHOLD_AMOUNT"
    NEW_ACCOUNT = "NEW_ACCOUNT"
    NEW_DEVICE = "NEW_DEVICE"
    HIGH_RISK_IP = "HIGH_RISK_IP"
    RAPID_OUTFLOW_AFTER_INFLOW = "RAPID_OUTFLOW_AFTER_INFLOW"


class DriftStatus(StrEnum):
    OK = "OK"
    WARNING = "WARNING"
    DRIFT = "DRIFT"
