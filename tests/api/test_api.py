from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud_ml.api.dependencies import CORRELATION_HEADER
from fraud_ml.config import Settings
from fraud_ml.main import create_app
from tests.conftest import transaction_payload


@pytest.fixture(scope="module")
def client(settings: Settings) -> TestClient:
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def no_model_client(tmp_path: Path, settings: Settings) -> TestClient:
    with TestClient(create_app(settings.model_copy(update={"model_dir": tmp_path}))) as client:
        yield client


def test_health_live(client: TestClient) -> None:
    body = client.get("/health/live").json()
    assert body["status"] == "ok" and body["service"] == "fraud-detection-ml"


def test_health_ready_with_model(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["model_version"] == "fraud-risk/1.0.0"


def test_not_ready_without_model(no_model_client: TestClient) -> None:
    ready = no_model_client.get("/health/ready")
    assert ready.status_code == 503 and ready.json()["model_loaded"] is False
    scored = no_model_client.post("/api/v1/score", json=transaction_payload())
    assert scored.status_code == 503
    body = scored.json()["error"]
    assert body["code"] == "MODEL_NOT_READY" and body["correlation_id"]
    assert "No valid fraud risk model" in body["message"]
    assert no_model_client.get("/api/v1/model").status_code == 503


def test_model_info_has_no_paths_or_secrets(client: TestClient, model_root: Path) -> None:
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "fraud-risk/1.0.0"
    assert body["schema_version"] == "fraud-transaction/1"
    assert body["feature_count"] > 20
    assert 0 <= body["policy"]["threshold"] <= 1
    assert str(model_root) not in response.text and "joblib" not in response.text


def test_score_endpoint(client: TestClient) -> None:
    response = client.post("/api/v1/score", json=transaction_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "TXN-SYN-TEST-0001"
    assert 0 <= body["risk_probability"] <= 1
    assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert body["decision_support"] in {"PASS", "REVIEW", "PRIORITY_REVIEW"}
    assert body["model_version"] == "fraud-risk/1.0.0"
    assert body["schema_version"] == "fraud-score/1"
    assert isinstance(body["reason_codes"], list)
    assert response.headers[CORRELATION_HEADER]


def test_score_risky_transaction_has_reason_codes(client: TestClient) -> None:
    risky = transaction_payload(
        amount="4800.00", account_median_amount_30d="60", is_new_beneficiary=True, beneficiary_country="NG",
        transaction_timestamp="2025-06-30T02:40:00Z", is_new_device=True, ip_risk_band="HIGH",
        transactions_1h=3, transactions_24h=5, unique_counterparties_24h=4,
    )  # fmt: skip
    body = client.post("/api/v1/score", json=risky).json()
    assert {"HIGH_AMOUNT_VS_HISTORY", "NEW_BENEFICIARY", "CROSS_BORDER"} <= set(
        body["reason_codes"]
    )
    assert len(body["reason_codes"]) <= 5


def test_invalid_input_returns_structured_error(client: TestClient) -> None:
    response = client.post("/api/v1/score", json=transaction_payload(amount="-1"))
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "INVALID_INPUT" and error["correlation_id"]
    assert error["details"][0]["loc"][-1] == "amount"
    assert "-1" not in str(error["details"])  # inputs are not echoed back


def test_unsupported_category(client: TestClient) -> None:
    response = client.post("/api/v1/score", json=transaction_payload(channel="PIGEON"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_CATEGORY"


def test_schema_version_mismatch(client: TestClient) -> None:
    response = client.post(
        "/api/v1/score", json=transaction_payload(schema_version="fraud-transaction/9")
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "SCHEMA_VERSION_MISMATCH"


def test_unseen_open_vocabulary_values_are_scored(client: TestClient) -> None:
    response = client.post(
        "/api/v1/score", json=transaction_payload(country="ZZ", merchant_category="BRAND_NEW_MCC")
    )
    assert response.status_code == 200


def test_batch_endpoint(client: TestClient) -> None:
    rows = [transaction_payload(transaction_id=f"TXN-SYN-B{i}") for i in range(5)]
    response = client.post("/api/v1/score/batch", json={"transactions": rows})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 5 and len(body["scores"]) == 5
    assert [s["transaction_id"] for s in body["scores"]] == [r["transaction_id"] for r in rows]


def test_batch_rejects_whole_request_when_one_row_invalid(client: TestClient) -> None:
    rows = [transaction_payload(), transaction_payload(amount="0")]
    response = client.post("/api/v1/score/batch", json={"transactions": rows})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"


def test_batch_too_large(client: TestClient, settings: Settings) -> None:
    rows = [transaction_payload() for _ in range(settings.api_max_batch_size + 1)]
    response = client.post("/api/v1/score/batch", json={"transactions": rows})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "BATCH_TOO_LARGE"
    assert client.post("/api/v1/score/batch", json={"transactions": []}).status_code == 422


def test_request_too_large(client: TestClient, settings: Settings) -> None:
    response = client.post(
        "/api/v1/score",
        content=b"{}",
        headers={
            "content-type": "application/json",
            "content-length": str(settings.api_max_request_bytes + 1),
        },
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


def test_correlation_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health/live", headers={CORRELATION_HEADER: "my-trace-123"})
    assert response.headers[CORRELATION_HEADER] == "my-trace-123"
    error = client.post(
        "/api/v1/score", json={}, headers={CORRELATION_HEADER: "bad-input-1"}
    ).json()["error"]
    assert error["correlation_id"] == "bad-input-1"


def test_unexpected_errors_do_not_leak_traces(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fraud_ml.scoring.scorer import Scorer

    def boom(self: Scorer, *_: object, **__: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(Scorer, "score_transactions", boom)
    response = client.post("/api/v1/score", json=transaction_payload())
    assert response.status_code == 500
    body = response.json()["error"]
    assert (
        body["code"] == "INTERNAL_ERROR"
        and "secret" not in response.text
        and "Traceback" not in response.text
    )


def test_scoring_failure_is_structured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from fraud_ml.exceptions import ScoringError
    from fraud_ml.scoring.scorer import Scorer

    def fail(self: Scorer, *_: object, **__: object) -> None:
        raise ScoringError("model inference failed: KeyError")

    monkeypatch.setattr(Scorer, "score_transactions", fail)
    response = client.post("/api/v1/score", json=transaction_payload())
    assert response.status_code == 500 and response.json()["error"]["code"] == "SCORING_FAILED"
    assert "KeyError" not in response.text


def test_openapi_documents_error_envelope(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "ErrorResponse" in schema["components"]["schemas"]
    assert "/api/v1/score/batch" in schema["paths"]
