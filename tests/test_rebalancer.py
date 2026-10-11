import numpy as np
import pandas as pd

from src.rebalancer import cap_and_normalise, latest_weights, load_constituents, rebalance_history


def make_signals(rows):
    return pd.DataFrame(rows, columns=["timestamp", "ticker", "sentiment_score", "impact_score"])


def test_weights_sum_to_one_and_respect_caps():
    w = cap_and_normalise(np.array([5.0, 1, 1, 1, 1, 1]), 0.05, 0.30)
    assert abs(w.sum() - 1) < 1e-9 and w.max() <= 0.30 + 1e-9 and w.min() >= 0.05 - 1e-9


def test_positive_sentiment_raises_weight_negative_lowers_it():
    c = load_constituents()
    sig = make_signals([("2026-10-05 09:00", "ATCL", 0.9, 8), ("2026-10-05 09:30", "NSTB", -0.9, 8)])
    h, _ = rebalance_history(sig, c)
    lw = latest_weights(h, c).set_index("ticker")
    assert lw.loc["ATCL", "change_pp"] > 0 and lw.loc["NSTB", "change_pp"] < 0
    assert abs(lw["current_weight"].sum() - 1) < 1e-9


def test_sentiment_decays_over_time():
    c = load_constituents()
    sig = make_signals([("2026-10-05 09:00", "ATCL", 0.9, 8), ("2026-10-09 09:00", "NSTB", 0.1, 5)])
    h, _ = rebalance_history(sig, c, half_life_hours=12)
    first = h[h.timestamp == h.timestamp.min()].set_index("ticker")["sentiment_state"]["ATCL"]
    last = h[h.timestamp == h.timestamp.max()].set_index("ticker")["sentiment_state"]["ATCL"]
    assert last < first


def test_no_signals_returns_base_weights():
    c = load_constituents()
    h, _ = rebalance_history(make_signals([]), c)
    lw = latest_weights(h, c)
    assert np.allclose(lw["current_weight"], lw["base_weight"])
