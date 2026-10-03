from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional, Union
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
credit_model = joblib.load(ART / "credit_tier_model.joblib")
eps_model = joblib.load(ART / "eps_forecast_model.joblib")
credit_meta = json.loads((ART / "credit_metadata.json").read_text())
eps_meta = json.loads((ART / "eps_metadata.json").read_text())

app = FastAPI(title="Banking Risk & Financial Intelligence API", version="1.0.0")

class CreditRequest(BaseModel):
    # Missing predictor values are allowed because the trained pipeline imputes them.
    features: Dict[str, Optional[Union[str, float, int]]] = Field(
        ..., description="Credit model feature dictionary; null is allowed for missing values"
    )

class EPSRequest(BaseModel):
    features: Dict[str, float]
    current_eps: float

class CombinedRequest(BaseModel):
    credit_tier: str
    eps_growth_pct: float

@app.get("/health")
def health():
    return {"status": "ok", "models": {"credit": True, "eps": True}}

@app.post("/credit/score")
def credit_score(req: CreditRequest):
    missing = [c for c in credit_meta["feature_columns"] if c not in req.features]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    X = pd.DataFrame([{c: req.features[c] for c in credit_meta["feature_columns"]}])
    idx = int(credit_model.predict(X)[0])
    probs = credit_model.predict_proba(X)[0].tolist()
    return {"predicted_tier": credit_meta["classes"][idx], "probabilities": dict(zip(credit_meta["classes"], probs))}

@app.post("/eps/forecast")
def eps_forecast(req: EPSRequest):
    required_input_features = [c for c in eps_meta["features"] if c != "Current EPS (Rs.)"]
    missing = [c for c in required_input_features if c not in req.features]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})
    row = {c: req.features[c] for c in required_input_features}
    row["Current EPS (Rs.)"] = req.current_eps
    X = pd.DataFrame([row], columns=eps_meta["features"])
    forecast = float(eps_model.predict(X)[0])
    growth = ((forecast / req.current_eps) - 1) * 100 if abs(req.current_eps) > 1e-9 else None
    return {"forecast_eps": forecast, "current_eps": req.current_eps, "forecast_growth_pct": growth, "model": eps_meta["selected_model"]}

@app.post("/combined/context")
def combined_context(req: CombinedRequest):
    if req.credit_tier not in {"P1", "P2", "P3", "P4"}:
        raise HTTPException(status_code=422, detail="credit_tier must be one of P1, P2, P3, P4")
    if req.credit_tier in {"P3", "P4"}:
        posture = "Enhanced credit review"
    else:
        posture = "Standard credit review"
    outlook = "positive" if req.eps_growth_pct > 5 else "negative" if req.eps_growth_pct < -5 else "stable"
    if outlook == "negative":
        posture += " + closer portfolio monitoring"
    return {"credit_tier": req.credit_tier, "financial_outlook": outlook, "scenario_posture": posture, "disclaimer": "Context rule only; not an automated underwriting policy."}
