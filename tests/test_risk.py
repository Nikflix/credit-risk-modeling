import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression

from risklab.api import create_app
from risklab.common import load_bundle, save_bundle
from risklab.data import RAW_FEATURES, features, split_indices
from risklab.evaluation import choose_threshold, operating_point
from risklab.monitoring import drift_report, make_reference
from risklab.service import CreditRequest, RiskService


@pytest.fixture
def request_data():
    return {
        "limit_balance": 100000,
        "repayment_status": [0] * 6,
        "bill_amounts": [30000] * 6,
        "payment_amounts": [4000] * 6,
    }


def test_duplicate_profiles_do_not_cross_splits():
    rng = np.random.default_rng(9)
    base = pd.DataFrame(rng.integers(1, 50, size=(200, len(RAW_FEATURES))), columns=RAW_FEATURES)
    frame = pd.concat([base, base.iloc[:40]], ignore_index=True)
    splits, groups = split_indices(frame)
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(frame)))
    names = list(splits)
    for i, name in enumerate(names):
        for other in names[i + 1 :]:
            assert set(groups[splits[name]]).isdisjoint(groups[splits[other]])


def test_feature_engineering_excludes_target_and_demographics(request_data):
    frame = CreditRequest(**request_data).frame()
    expected = features(frame)
    frame["SEX"], frame["AGE"], frame["default payment next month"] = 1, 99, 1
    pd.testing.assert_frame_equal(expected, features(frame))
    assert expected.utilization_latest.iloc[0] == 0.3
    frame["BILL_AMT1"] = 0
    assert np.isfinite(features(frame).to_numpy()).all()


@pytest.mark.parametrize(
    "change",
    [
        {"limit_balance": 0},
        {"limit_balance": float("nan")},
        {"repayment_status": [0] * 5},
        {"repayment_status": [1.5] * 6},
        {"payment_amounts": [-1] * 6},
        {"unknown": 1},
    ],
)
def test_invalid_requests_are_rejected(request_data, change):
    with pytest.raises(ValidationError):
        CreditRequest(**(request_data | change))


def test_threshold_uses_costs_and_handles_no_alerts():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])
    choice = choose_threshold(y, p)
    assert choice["cost_per_1000"] == 0
    assert operating_point(y, p, 1)["fn"] == 2


def test_drift_includes_outside_training_range():
    frame = pd.DataFrame({"balance": np.arange(100)})
    reference = make_reference(frame)
    assert drift_report(frame, reference)[0]["psi"] == pytest.approx(0)
    shifted = drift_report(pd.DataFrame({"balance": [10000] * 100}), reference)[0]
    assert shifted["investigate"]
    with pytest.raises(ValueError):
        drift_report(pd.DataFrame({"balance": [np.nan]}), reference)


def test_model_integrity_and_real_api_scoring(tmp_path, request_data):
    rng = np.random.default_rng(12)
    rows = []
    for _ in range(50):
        example = request_data | {"limit_balance": float(rng.integers(10000, 300000))}
        rows.append(CreditRequest(**example).frame())
    x = features(pd.concat(rows, ignore_index=True))
    model = LogisticRegression(max_iter=1000).fit(x, np.arange(50) % 2)
    save_bundle(tmp_path, {"model": model, "threshold": 0.4, "name": "test_logistic"})
    service = RiskService(tmp_path)
    with TestClient(create_app(service)) as client:
        assert client.get("/ready").status_code == 200
        result = client.post("/predict", json=request_data)
        assert result.status_code == 200
        assert 0 <= result.json()["default_probability"] <= 1
        assert result.json()["action"] in {"review", "monitor"}
        assert (
            client.post("/predict", json=request_data | {"payment_amounts": [-1] * 6}).status_code
            == 422
        )
    with (tmp_path / "model.joblib").open("ab") as target:
        target.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


def test_missing_model_fails_readiness(monkeypatch, tmp_path, request_data):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=request_data).status_code == 503


@pytest.mark.skipif(
    not Path("artifacts/model.joblib").exists(),
    reason="Run training for the full-model integration check",
)
def test_trained_artifact_matches_report(request_data):
    service = RiskService()
    report = json.loads(Path("reports/metrics.json").read_text())
    with TestClient(create_app(service)) as client:
        result = client.post("/predict", json=request_data).json()
    assert result["model"] == report["selected_model"]
    assert result["review_threshold"] == report["operating_point"]["threshold"]
    assert result["model_version"] == report["artifact"]["sha256"][:12]
