"""Module B: event-driven stress test on a synthetic wholesale-banking portfolio.

Shocks are defined per event type and applied by asset class:
  equity      -> flat % move
  bond        -> -(duration * rate shift + spread_duration * spread widening)
  loan        -> -(spread_duration * spread widening) - expected credit loss %
  derivative  -> flat % move (simplified, no Greeks)
Shocks are calibrated for an impact score of 7 and scaled up for more severe events.
"""
from __future__ import annotations

import pandas as pd

# rate_bps: parallel rate shift, spread_bps: credit spread widening
DEFAULT_SHOCKS = {
    "Geopolitical":              {"equity_pct": -0.12, "rate_bps": -25, "spread_bps": 150, "loan_loss_pct": 0.010, "derivative_pct": -0.10},
    "Macroeconomic":             {"equity_pct": -0.08, "rate_bps": 100, "spread_bps": 60,  "loan_loss_pct": 0.005, "derivative_pct": -0.06},
    "Credit Event":              {"equity_pct": -0.09, "rate_bps": 0,   "spread_bps": 250, "loan_loss_pct": 0.030, "derivative_pct": -0.08},
    "Merger/Acquisition":        {"equity_pct": -0.03, "rate_bps": 0,   "spread_bps": 20,  "loan_loss_pct": 0.002, "derivative_pct": -0.02},
    "Product/Business Update":   {"equity_pct": -0.02, "rate_bps": 0,   "spread_bps": 10,  "loan_loss_pct": 0.0,   "derivative_pct": -0.01},
    "Legal/Regulatory":          {"equity_pct": -0.07, "rate_bps": 0,   "spread_bps": 80,  "loan_loss_pct": 0.010, "derivative_pct": -0.05},
    "Earnings/Financial Results": {"equity_pct": -0.06, "rate_bps": 0,  "spread_bps": 30,  "loan_loss_pct": 0.003, "derivative_pct": -0.03},
    "Cybersecurity":             {"equity_pct": -0.06, "rate_bps": 0,   "spread_bps": 40,  "loan_loss_pct": 0.005, "derivative_pct": -0.04},
    "General Market News":       {"equity_pct": -0.03, "rate_bps": 0,   "spread_bps": 25,  "loan_loss_pct": 0.003, "derivative_pct": -0.02},
}
REQUIRED = {"asset", "asset_type", "notional_usd"}


def severity_scale(impact_score: int, calibrated_at: int = 7) -> float:
    return min(1.5, max(1.0, impact_score / calibrated_at))


def asset_shock(row: pd.Series, shock: dict, scale: float) -> float:
    kind = str(row["asset_type"]).lower()
    dur = float(row.get("duration_years", 0) or 0)
    sdur = float(row.get("spread_duration", 0) or 0)
    if kind == "equity":
        pct = shock["equity_pct"]
    elif kind == "bond":
        pct = -(dur * shock["rate_bps"] + sdur * shock["spread_bps"]) / 1e4
    elif kind == "loan":
        pct = -(sdur * shock["spread_bps"]) / 1e4 - shock["loan_loss_pct"]
    elif kind == "derivative":
        pct = shock["derivative_pct"]
    else:
        pct = shock["equity_pct"]
    return pct * scale


def stress_portfolio(portfolio: pd.DataFrame, event_type: str, impact_score: int,
                     trigger_threshold: int = 7, sentiment: float | None = None,
                     require_negative: bool = False, custom_shocks: dict | None = None):
    """Return (asset-level detail, summary dict).

    The test only fires when impact >= threshold and, if require_negative is set,
    the signal's sentiment is negative (a positive-news event should not shock the book).
    """
    missing = REQUIRED - set(portfolio.columns)
    if missing:
        raise ValueError(f"Portfolio missing required columns: {sorted(missing)}")

    detail = portfolio.copy()
    detail["notional_usd"] = detail["notional_usd"].astype(float)
    before = float(detail["notional_usd"].sum())

    reason = None
    if impact_score < trigger_threshold:
        reason = f"impact {impact_score} is below threshold {trigger_threshold}"
    elif require_negative and sentiment is not None and sentiment >= 0:
        reason = "signal sentiment is not negative"

    if reason:
        detail["shock_pct"] = 0.0
        detail["loss_usd"] = 0.0
        detail["stressed_value_usd"] = detail["notional_usd"]
        return detail, {"triggered": False, "reason": reason, "before_usd": before, "after_usd": before,
                        "loss_usd": 0.0, "loss_pct": 0.0, "event_type": event_type,
                        "impact_score": impact_score}

    shocks = custom_shocks or DEFAULT_SHOCKS
    shock = shocks.get(event_type, shocks["General Market News"])
    scale = severity_scale(impact_score)
    detail["shock_pct"] = detail.apply(lambda r: asset_shock(r, shock, scale), axis=1)
    detail["stressed_value_usd"] = detail["notional_usd"] * (1 + detail["shock_pct"])
    detail["loss_usd"] = detail["notional_usd"] - detail["stressed_value_usd"]
    after = float(detail["stressed_value_usd"].sum())
    return detail, {"triggered": True, "reason": None, "before_usd": before, "after_usd": after,
                    "loss_usd": before - after, "loss_pct": (before - after) / before * 100 if before else 0.0,
                    "event_type": event_type, "impact_score": impact_score,
                    "severity_scale": scale, "shock_assumptions": shock}
