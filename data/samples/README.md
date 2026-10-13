# Sample data

Small synthetic fixtures committed for quick starts and the CLI defaults. Larger datasets are
generated locally (`fraud-ml generate-data`) and git-ignored.

| File | Rows | Content |
|---|---:|---|
| `scoring.parquet` | 1,000 | the latest 1,000 transactions of the seed-42 demo dataset **without labels**; default input of `fraud-ml score-file` and `fraud-ml drift-check --current` |
| `transactions_labelled_sample.parquet` | 3,000 | the earliest 3,000 rows of the same dataset **with labels** (`is_fraud`, `fraud_type`); enough for `fraud-ml evaluate` smoke runs, too small for meaningful training |

Both follow `fraud-transaction/1` (see `docs/DATA_CONTRACT.md`). All identifiers are
synthetic (`TXN-SYN-…`, `ACC-SYN-…`, `CP-SYN-…`, `DEV-SYN-…`).
