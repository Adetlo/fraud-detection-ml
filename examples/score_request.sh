#!/usr/bin/env bash
# Score the example transactions against a running API (fraud-ml serve / docker compose up).
set -euo pipefail
BASE="${FRAUD_ML_API:-http://127.0.0.1:8000}"
DIR="$(cd "$(dirname "$0")" && pwd)"

curl -fsS "$BASE/health/ready"; echo
curl -fsS "$BASE/api/v1/model"; echo
curl -fsS -X POST "$BASE/api/v1/score" -H 'Content-Type: application/json' --data @"$DIR/transaction.json"; echo
curl -fsS -X POST "$BASE/api/v1/score" -H 'Content-Type: application/json' --data @"$DIR/transaction_low_risk.json"; echo
curl -sS -X POST "$BASE/api/v1/score" -H 'Content-Type: application/json' -d '{"transaction_id": "TXN-SYN-BAD"}'; echo
