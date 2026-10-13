# Examples

| File | Purpose |
|---|---|
| `transaction.json` | A higher-risk synthetic transaction (new beneficiary, cross-border, night, new device, high-risk IP). Used by the Docker smoke test and `fraud-ml score`. |
| `transaction_low_risk.json` | An ordinary card purchase on a known merchant. |
| `score_request.sh` | curl calls against a running API. |

```bash
fraud-ml score --input examples/transaction.json
fraud-ml score --input examples/transaction_low_risk.json
```
