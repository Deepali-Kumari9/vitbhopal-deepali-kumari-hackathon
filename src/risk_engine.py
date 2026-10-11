"""NLP risk engine: turns raw text into structured risk signals.

Each text item produces:
    sentiment_score  float in [-1, 1]
    event_type       categorical label
    impact_score     int in [1, 10]
plus the company/ticker it refers to and the terms that triggered the scores,
so every number can be traced back to the words behind it.

The scorer is a lexicon + rules baseline (negation, intensifiers, word-boundary
matching). It is deliberately simple and explainable; see README for limitations.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- lexicons
POSITIVE_TERMS = {
    "beat": 0.8, "beats": 0.8, "growth": 0.6, "surge": 0.9, "surges": 0.9,
    "soar": 0.9, "soars": 0.9, "jump": 0.6, "jumps": 0.6, "gain": 0.5, "gains": 0.5,
    "profit": 0.7, "profits": 0.7, "upgrade": 0.7, "upgraded": 0.7, "record": 0.5,
    "strong": 0.5, "rally": 0.8, "rallies": 0.8, "approval": 0.6, "approved": 0.6,
    "partnership": 0.4, "launch": 0.3, "launches": 0.3, "recovery": 0.5,
    "outperform": 0.7, "dividend": 0.4, "expansion": 0.4, "raises": 0.4,
    "boost": 0.5, "boosts": 0.5, "promising": 0.5, "positive": 0.5, "upbeat": 0.6,
    "optimistic": 0.6, "bullish": 0.7, "wins": 0.5, "win": 0.5, "milestone": 0.4,
}
NEGATIVE_TERMS = {
    "fall": -0.6, "falls": -0.6, "fell": -0.6, "drop": -0.6, "drops": -0.6,
    "plunge": -0.9, "plunges": -0.9, "slump": -0.7, "slumps": -0.7, "tumble": -0.7,
    "tumbles": -0.7, "crash": -1.0, "crashes": -1.0, "loss": -0.7, "losses": -0.7,
    "fraud": -1.0, "lawsuit": -0.7, "probe": -0.6, "investigation": -0.5,
    "downgrade": -0.7, "downgraded": -0.7, "default": -1.0, "bankruptcy": -1.0,
    "layoffs": -0.6, "misses": -0.6, "missed": -0.6, "weak": -0.5, "decline": -0.6,
    "declines": -0.6, "sanctions": -0.7, "war": -0.8, "conflict": -0.6,
    "inflation": -0.3, "recession": -0.8, "recall": -0.6, "recalls": -0.6,
    "breach": -0.8, "cuts": -0.4, "cut": -0.4, "delay": -0.4, "delayed": -0.4,
    "warns": -0.5, "warning": -0.5, "concerns": -0.4, "worried": -0.5, "worries": -0.5,
    "worrying": -0.5, "fears": -0.5, "risk": -0.3, "risks": -0.3, "bearish": -0.7,
    "outage": -0.6, "ransomware": -0.8, "fined": -0.6, "fine": -0.3, "shortfall": -0.6,
    "bad": -0.5,
}
NEGATORS = {"no", "not", "never", "without", "denies", "denied", "avoids", "avoided",
            "unlikely", "fails", "failed", "isn't", "wasn't", "doesn't", "didn't", "won't"}
INTENSIFIERS = {"sharp": 1.3, "sharply": 1.3, "significant": 1.2, "major": 1.2,
                "massive": 1.4, "severe": 1.4, "steep": 1.3, "huge": 1.3, "slight": 0.6,
                "modest": 0.7, "marginal": 0.6}
NEGATION_WINDOW = 3

EVENT_KEYWORDS = {
    "Geopolitical": ["war", "conflict", "sanctions", "tariff", "tariffs", "invasion",
                     "geopolitical", "embargo", "military"],
    "Macroeconomic": ["inflation", "interest rate", "interest rates", "central bank",
                      "recession", "gdp", "unemployment", "fed", "rate hike", "rate cut"],
    "Credit Event": ["default", "bankruptcy", "credit downgrade", "downgrade", "debt",
                     "liquidity", "missed payment", "loan losses", "insolvency"],
    "Merger/Acquisition": ["merger", "acquisition", "acquires", "acquire", "buyout",
                           "takeover", "merge"],
    "Product/Business Update": ["product launch", "launches", "launch", "new product",
                                "approval", "expansion", "partnership", "contract"],
    "Legal/Regulatory": ["lawsuit", "investigation", "probe", "regulator", "fine",
                         "fined", "fraud", "antitrust"],
    "Earnings/Financial Results": ["earnings", "revenue", "profit", "quarterly results",
                                   "forecast", "guidance", "quarterly profit", "sales"],
    "Cybersecurity": ["cyberattack", "data breach", "ransomware", "hack", "hackers",
                      "security breach", "breach"],
}
EVENT_BASE_IMPACT = {
    "Geopolitical": 7, "Macroeconomic": 6, "Credit Event": 8,
    "Merger/Acquisition": 5, "Product/Business Update": 3,
    "Legal/Regulatory": 6, "Earnings/Financial Results": 5,
    "Cybersecurity": 6, "General Market News": 2,
}
# extra points added to the impact score when these appear
SEVERITY_TERMS = {
    "default": 2, "bankruptcy": 2, "fraud": 2, "crash": 2, "war": 2, "breach": 1,
    "sanctions": 1, "lawsuit": 1, "investigation": 1, "recession": 2, "ransomware": 1,
    "plunge": 1, "plunges": 1, "downgrade": 1, "misses": 1, "insolvency": 2,
}
EVENT_TYPES = list(EVENT_BASE_IMPACT)
POS_THRESHOLD, NEG_THRESHOLD = 0.15, -0.15


# ----------------------------------------------------------------- helpers
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def score_sentiment(text: str, return_terms: bool = False):
    """Lexicon sentiment in [-1, 1], with negation and intensifier handling."""
    words = _tokens(text)
    total, hits = 0.0, []
    for i, w in enumerate(words):
        val = POSITIVE_TERMS.get(w, 0.0) + NEGATIVE_TERMS.get(w, 0.0)
        if val == 0.0:
            continue
        window = words[max(0, i - NEGATION_WINDOW):i]
        if any(t in NEGATORS for t in window):
            val *= -0.7
        mult = 1.0
        for t in window[-2:]:
            mult *= INTENSIFIERS.get(t, 1.0)
        total += val * mult
        hits.append(w)
    # tanh keeps the scale bounded without letting text length dilute short headlines
    score = round(math.tanh(total / 1.5), 3)
    return (score, hits) if return_terms else score


def _kw_hits(lowered: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\b", lowered) is not None


def classify_event(text: str) -> str:
    lowered = (text or "").lower()
    scores = {ev: sum(1 for k in kws if _kw_hits(lowered, k)) for ev, kws in EVENT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "General Market News"


def calculate_impact(text: str, event_type: str, sentiment_score: float) -> int:
    """Heuristic 1-10 severity: event base + severity words + strong sentiment."""
    words = set(_tokens(text))
    score = EVENT_BASE_IMPACT.get(event_type, 2)
    score += min(2, sum(pts for term, pts in SEVERITY_TERMS.items() if term in words))
    if abs(sentiment_score) >= 0.65:
        score += 1
    return max(1, min(10, int(score)))


@lru_cache(maxsize=1)
def _company_table() -> pd.DataFrame:
    path = ROOT / "data" / "companies.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["ticker", "company"])


def extract_entity(text: str) -> tuple[str, str]:
    """Find the company/ticker a text is about. Cashtags ($NSTB) win over names."""
    table = _company_table()
    lowered = (text or "").lower()
    for tag in re.findall(r"\$([A-Za-z]{2,5})\b", text or ""):
        row = table[table["ticker"].str.lower() == tag.lower()]
        if not row.empty:
            return row.iloc[0]["company"], row.iloc[0]["ticker"]
    for _, row in table.iterrows():
        if row["company"].lower() in lowered:
            return row["company"], row["ticker"]
    return "Unspecified", ""


def analyze_text(text: str, source: str = "manual", company: str = "",
                 timestamp: str | None = None) -> dict[str, Any]:
    sentiment, hits = score_sentiment(text, return_terms=True)
    event_type = classify_event(text)
    impact = calculate_impact(text, event_type, sentiment)
    name, ticker = extract_entity(text)
    if company and company != "Unspecified":
        name = company
        match = _company_table()[_company_table()["company"].str.lower() == company.lower()]
        ticker = match.iloc[0]["ticker"] if not match.empty else ticker
    label = "Positive" if sentiment > POS_THRESHOLD else "Negative" if sentiment < NEG_THRESHOLD else "Neutral"
    return {
        "timestamp": timestamp or "",
        "source": source,
        "company": name,
        "ticker": ticker,
        "text": text,
        "sentiment_score": sentiment,
        "sentiment_label": label,
        "event_type": event_type,
        "impact_score": impact,
        "high_impact": impact >= 7,
        "matched_terms": ", ".join(hits),
    }


def analyze_records(records) -> list[dict[str, Any]]:
    return [analyze_text(str(r.get("text", "")), str(r.get("source", "unknown")),
                         str(r.get("company", "") or ""), str(r.get("timestamp", "") or ""))
            for r in records]
