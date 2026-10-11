"""Small HTTP API so downstream systems can pull signals.

    uvicorn api:app --reload        # docs at http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel

from src.pipeline import ROOT, build_signals
from src.rebalancer import latest_weights, load_constituents, rebalance_history
from src.risk_engine import analyze_text
from src.stress_test import stress_portfolio

app = FastAPI(title="RiskPulse API", version="1.0",
              description="Structured risk signals, index weights and stress tests from text.")

_signals = build_signals()
_constituents = load_constituents()
_portfolio = pd.read_csv(ROOT / "data" / "synthetic_portfolio.csv")


class TextIn(BaseModel):
    text: str
    source: str = "api"
    company: str = ""


class StressIn(BaseModel):
    event_type: str
    impact_score: int
    threshold: int = 7


@app.get("/health")
def health():
    return {"status": "ok", "signals_loaded": len(_signals)}


@app.post("/analyze")
def analyze(body: TextIn):
    return analyze_text(body.text, body.source, body.company)


@app.get("/signals")
def signals(min_impact: int = Query(1, ge=1, le=10), event_type: str | None = None, limit: int = 100):
    df = _signals[_signals["impact_score"] >= min_impact]
    if event_type:
        df = df[df["event_type"] == event_type]
    return df.head(limit).to_dict("records")


@app.get("/index/weights")
def index_weights():
    history, _ = rebalance_history(_signals, _constituents)
    return latest_weights(history, _constituents).round(4).to_dict("records")


@app.post("/stress")
def stress(body: StressIn):
    detail, summary = stress_portfolio(_portfolio, body.event_type, body.impact_score, body.threshold)
    return {"summary": summary, "assets": detail.round(4).to_dict("records")}
