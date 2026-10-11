from fastapi.testclient import TestClient

from api import app
from src.evaluate import evaluate
from src.pipeline import build_signals

client = TestClient(app)


def test_pipeline_uses_two_sources_and_full_schema():
    sig = build_signals()
    assert set(sig["source"]) >= {"news", "social"}
    assert sig["sentiment_score"].between(-1, 1).all()
    assert sig["impact_score"].between(1, 10).all()


def test_api_endpoints():
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/analyze", json={"text": "Northstar Bank faces a fraud investigation"})
    assert r.status_code == 200 and r.json()["event_type"] == "Legal/Regulatory"
    assert len(client.get("/index/weights").json()) == 15
    s = client.post("/stress", json={"event_type": "Credit Event", "impact_score": 9}).json()
    assert s["summary"]["triggered"]
    assert all(x["impact_score"] >= 7 for x in client.get("/signals?min_impact=7").json())


def test_engine_beats_majority_baseline():
    res = evaluate()
    assert res["sentiment"]["accuracy"] > res["sentiment"]["majority_baseline_accuracy"]
    assert res["event_type"]["accuracy"] > res["event_type"]["majority_baseline_accuracy"]
