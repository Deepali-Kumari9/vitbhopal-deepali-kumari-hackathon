"""Module A: tactical sentiment-driven rebalancing of a mock 15-stock index.

For each stock we keep a running sentiment state that decays over time and is
updated by every new signal (scaled by its impact score). Weights are an
exponential tilt of the base weights, then capped and renormalised.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_constituents(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or ROOT / "data" / "companies.csv")


def cap_and_normalise(w: np.ndarray, min_w: float, max_w: float, iters: int = 50) -> np.ndarray:
    """Normalise to sum 1 while keeping every weight inside [min_w, max_w]."""
    w = w / w.sum()
    for _ in range(iters):
        clipped = np.clip(w, min_w, max_w)
        free = (clipped > min_w) & (clipped < max_w)
        gap = 1.0 - clipped.sum()
        if abs(gap) < 1e-12 or not free.any():
            w = clipped
            break
        clipped[free] += gap * clipped[free] / clipped[free].sum()
        w = clipped
    return w / w.sum()


def rebalance_history(signals: pd.DataFrame, constituents: pd.DataFrame, freq: str = "6h",
                      tilt: float = 1.0, half_life_hours: float = 24.0,
                      min_weight: float = 0.02, max_weight: float = 0.15):
    """Replay signals in time order and return (weight history, turnover per step)."""
    tickers = constituents["ticker"].tolist()
    base = constituents["base_weight"].to_numpy(dtype=float)
    base = base / base.sum()
    state = dict.fromkeys(tickers, 0.0)

    sig = signals[signals["ticker"].isin(tickers)].copy()
    sig["ts"] = pd.to_datetime(sig["timestamp"], errors="coerce")
    sig = sig.dropna(subset=["ts"]).sort_values("ts")
    if sig.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "sentiment_state", "weight"]), pd.DataFrame()

    sig["bucket"] = sig["ts"].dt.floor(freq)
    decay_per_hour = math.log(2) / half_life_hours
    prev_t, prev_w = None, base.copy()
    history, turnover = [], []

    for bucket, chunk in sig.groupby("bucket"):
        if prev_t is not None:
            hours = (bucket - prev_t).total_seconds() / 3600
            for k in state:
                state[k] *= math.exp(-decay_per_hour * hours)
        for _, r in chunk.iterrows():
            strength = 0.5 + r["impact_score"] / 20          # 0.55 .. 1.0
            state[r["ticker"]] = float(np.clip(state[r["ticker"]] + r["sentiment_score"] * strength, -2, 2))
        s = np.array([state[t] for t in tickers])
        w = cap_and_normalise(base * np.exp(tilt * s), min_weight, max_weight)
        for t, st_, wt in zip(tickers, s, w):
            history.append({"timestamp": bucket, "ticker": t, "sentiment_state": round(st_, 4), "weight": wt})
        turnover.append({"timestamp": bucket, "turnover": 0.5 * float(np.abs(w - prev_w).sum())})
        prev_t, prev_w = bucket, w
    return pd.DataFrame(history), pd.DataFrame(turnover)


def latest_weights(history: pd.DataFrame, constituents: pd.DataFrame) -> pd.DataFrame:
    """Base vs current weight per stock."""
    out = constituents[["ticker", "company", "sector", "base_weight"]].copy()
    out["base_weight"] = out["base_weight"] / out["base_weight"].sum()
    if history.empty:
        out["current_weight"] = out["base_weight"]
    else:
        last = history[history["timestamp"] == history["timestamp"].max()].set_index("ticker")["weight"]
        out["current_weight"] = out["ticker"].map(last).fillna(out["base_weight"])
    out["change_pp"] = (out["current_weight"] - out["base_weight"]) * 100
    return out.sort_values("current_weight", ascending=False).reset_index(drop=True)
