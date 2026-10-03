from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_credit_artifact_scores_demo_rows():
    model = joblib.load(ROOT / "artifacts" / "credit_tier_model.joblib")
    meta = json.loads((ROOT / "artifacts" / "credit_metadata.json").read_text())
    demo = pd.read_csv(ROOT / "data" / "sample" / "credit_applicants_demo.csv").head(2)
    X = demo[meta["feature_columns"]].replace(-99999, pd.NA)
    pred = model.predict(X)
    assert len(pred) == 2
    assert all(int(x) in range(4) for x in pred)


def test_eps_artifact_forecasts_reference_banks():
    model = joblib.load(ROOT / "artifacts" / "eps_forecast_model.joblib")
    meta = json.loads((ROOT / "artifacts" / "eps_metadata.json").read_text())
    ref = pd.read_csv(ROOT / "artifacts" / "bank_latest_reference.csv").head(3)
    ref = ref.rename(columns={"Basic EPS (Rs.)": "Current EPS (Rs.)"})
    X = ref[meta["features"]]
    pred = model.predict(X)
    assert len(pred) == 3
    assert all(float(x) == float(x) for x in pred)


def test_decision_engine():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.models.decision_engine import combine_context
    result = combine_context("P4", -10.0)
    assert result["financial_outlook"] == "negative"
    assert "Enhanced credit review" in result["scenario_posture"]


def test_eps_api_accepts_current_eps_separately():
    import sys
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from api.main import app
    meta = json.loads((ROOT / "artifacts" / "eps_metadata.json").read_text())
    ref = pd.read_csv(ROOT / "artifacts" / "bank_latest_reference.csv").head(1)
    row = ref.iloc[0]
    features = {c: float(row[c]) for c in meta["features"] if c != "Current EPS (Rs.)"}
    client = TestClient(app)
    response = client.post("/eps/forecast", json={
        "features": features,
        "current_eps": float(row["Basic EPS (Rs.)"]),
    })
    assert response.status_code == 200
    payload = response.json()
    assert "forecast_eps" in payload
    assert payload["current_eps"] == float(row["Basic EPS (Rs.)"])


def test_credit_api_scores_demo_row():
    import sys
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from api.main import app
    meta = json.loads((ROOT / "artifacts" / "credit_metadata.json").read_text())
    demo = pd.read_csv(ROOT / "data" / "sample" / "credit_applicants_demo.csv").head(1)
    row = demo.iloc[0]
    features = {
        c: None if pd.isna(row[c]) else (row[c].item() if hasattr(row[c], "item") else row[c])
        for c in meta["feature_columns"]
    }
    client = TestClient(app)
    response = client.post("/credit/score", json={"features": features})
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_tier"] in meta["classes"]
    assert set(payload["probabilities"]) == set(meta["classes"])
