# Data contract

## Input contract: `fraud-transaction/1`

Defined by `fraud_ml.contracts.transaction.TransactionFeatures` and enforced twice: by
Pydantic for API requests and by `fraud_ml.data.validation.validate_frame` for bulk files.
Both use the same enums, bounds and cross-field rules.

A record is **one transaction plus the point-in-time context known before it was
processed**. Every history field must be computed from strictly earlier transactions of the
same account; the reference implementation is `fraud_ml.features.velocity.compute_history_features`.

| Field | Type | Required | Rules |
|---|---|---|---|
| `schema_version` | string | no (default `fraud-transaction/1`) | must be a supported version |
| `transaction_id` | string 1-64 | yes | non-blank; synthetic form `TXN-SYN-…` |
| `account_id` | string 1-64 | yes | non-blank; `ACC-SYN-…` |
| `transaction_timestamp` | ISO-8601 datetime **with offset** | yes | converted to UTC; naive timestamps rejected; not before 2000-01-01; at most 1 day in the future |
| `amount` | Decimal | yes | > 0, ≤ 100,000,000, at most the currency's minor-unit places (JPY: 0, others: 2) |
| `currency` | enum | yes | GBP, USD, EUR, JPY, CHF, SGD, AUD |
| `transaction_type` | enum | yes | TRANSFER, PAYMENT, DIRECT_DEBIT, CARD_PURCHASE, DEPOSIT, WITHDRAWAL, REFUND |
| `channel` | enum | yes | ONLINE, MOBILE, BRANCH, ATM, API, CARD, BATCH |
| `country` | ISO 3166-1 alpha-2 | yes | upper-cased |
| `counterparty_id` | string or null | no | `CP-SYN-…` |
| `beneficiary_country` | alpha-2 or null | no | |
| `is_weekend` | bool | yes | recomputed from the timestamp for features |
| `account_age_days` | int | yes | 0 … 36,500 |
| `transactions_1h` | int | yes | 0 … 10,000; ≤ `transactions_24h` |
| `transactions_24h` | int | yes | 0 … 10,000 |
| `amount_1h` | Decimal | yes | ≥ 0; ≤ `amount_24h` |
| `amount_24h` | Decimal | yes | ≥ 0 |
| `unique_counterparties_24h` | int | yes | ≤ `transactions_24h` |
| `days_since_last_transaction` | float or null | no | null only for an account's first transaction (required when `transactions_24h` > 0) |
| `account_median_amount_30d` | Decimal or null | no | null without 30-day history |
| `account_mean_amount_30d` | Decimal or null | no | |
| `inflow_amount_24h` | Decimal | no (default 0) | sum of DEPOSIT/REFUND in the prior 24h |
| `merchant_category` | string or null | no | open vocabulary |
| `device_id_hash` | string or null | no | identifier only, never a feature |
| `ip_risk_band` | enum | no (default UNKNOWN) | LOW, MEDIUM, HIGH, UNKNOWN |
| `customer_segment` | enum or null | no | RETAIL, PREMIUM, BUSINESS |
| `is_new_beneficiary` | bool or null | no | counterparty unseen on the account before |
| `is_new_device` | bool or null | no | device unseen on the account before |

Unknown fields are ignored. Blank identifiers are rejected. Money is `Decimal` at the
contract boundary and `float64` inside data frames; `fraud_ml.contracts.money.to_decimal`
restores the exact value (property-tested).

### Label contract (training and evaluation files only)

| Field | Type | Rules |
|---|---|---|
| `is_fraud` | bool | synthetic ground truth |
| `fraud_type` | enum | NONE, ACCOUNT_TAKEOVER, CARD_NOT_PRESENT, MULE_ACTIVITY, RAPID_MOVEMENT, STRUCTURING; NONE iff `is_fraud` is false |

These are **generator conventions** for evaluation. They do not represent regulatory or
industry fraud definitions and never enter the feature matrix (`FORBIDDEN_COLUMNS`).

### Unknown categories

* Closed vocabularies (`currency`, `transaction_type`, `channel`, `ip_risk_band`,
  `customer_segment`) are contract-enforced: an unknown value is rejected with
  `UNSUPPORTED_CATEGORY` before the model sees it. Adding a value is a contract change.
* Open vocabularies (`country`, `beneficiary_country`, `merchant_category`) are encoded with
  an explicit *infrequent* bucket: values unseen in training or seen fewer than 20 times
  share one one-hot column. A brand-new country never crashes scoring.

## Output contract: `fraud-score/1`

```json
{
  "schema_version": "fraud-score/1",
  "transaction_id": "TXN-SYN-00049848",
  "model_version": "fraud-risk/1.0.0",
  "risk_probability": 0.450862,
  "risk_band": "CRITICAL",
  "decision_support": "PRIORITY_REVIEW",
  "reason_codes": ["NEW_BENEFICIARY"],
  "threshold": 0.065,
  "scored_at": "2026-10-13T14:24:34.638825Z"
}
```

Bands and decisions are defined in [THRESHOLDING.md](THRESHOLDING.md); reason codes in
[EXPLAINABILITY.md](EXPLAINABILITY.md). Batch output (Parquet) carries the same columns.

## Relationship to `txn-data-pipeline` (conceptual)

```
corebank-ledger  ->  synthetic financial activity
txn-data-pipeline  ->  validated, curated transaction Parquet (financial-transaction/1)
fraud-detection-ml  ->  risk score + explanation + decision support
```

The curated layer of [txn-data-pipeline](https://github.com/Adetlo/txn-data-pipeline) carries
the raw attributes (`transaction_id`, `account_id`, `transaction_timestamp`, `amount`,
`currency`, `transaction_type`, `channel`, `country`, `counterparty_id`,
`beneficiary_country`, `merchant_category`, `status`) but not the point-in-time context.
`fraud_ml.adapters.txn_pipeline.adapt_curated_frame`:

1. keeps POSTED rows only (pending, reversed and cancelled transactions are out of scope);
2. maps the pipeline's vocabulary onto this contract (`FEE` -> `PAYMENT`, `INTEREST` ->
   `DEPOSIT`; the rest is shared);
3. derives every history field with `compute_history_features`;
4. approximates `account_age_days` from the first observed transaction unless an
   `account_open_date` column is supplied (documented approximation);
5. returns a frame that passes `validate_frame`.

The adapter is optional and tested against a curated-shaped fixture. This repository never
imports or requires the pipeline at runtime (ADR 0006).

## Versioning

`schema_version` is carried on every input row, in every artifact's metadata and in every
score. A model trained on `fraud-transaction/1` refuses to load in a build that no longer
supports that version (`MODEL_VERSION_MISMATCH`), and a request carrying an unsupported
version is rejected (`SCHEMA_VERSION_MISMATCH`). A `fraud-transaction/2` would be introduced
as a second Pydantic model plus an adapter, with the version added to
`SUPPORTED_SCHEMA_VERSIONS`.
