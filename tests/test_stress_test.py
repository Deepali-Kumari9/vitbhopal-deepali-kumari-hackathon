import pandas as pd
import pytest

from src.stress_test import severity_scale, stress_portfolio


def book():
    return pd.DataFrame([
        {"asset": "Equity", "asset_type": "equity", "notional_usd": 1000, "duration_years": 0, "spread_duration": 0},
        {"asset": "Corp bond", "asset_type": "bond", "notional_usd": 1000, "duration_years": 5, "spread_duration": 4},
        {"asset": "Loan", "asset_type": "loan", "notional_usd": 1000, "duration_years": 3, "spread_duration": 3},
        {"asset": "Swap", "asset_type": "derivative", "notional_usd": 1000, "duration_years": 4, "spread_duration": 0},
    ])


def test_high_impact_triggers_and_reduces_value():
    detail, s = stress_portfolio(book(), "Geopolitical", 8)
    assert s["triggered"] and s["loss_usd"] > 0 and len(detail) == 4


def test_below_threshold_no_trigger():
    _, s = stress_portfolio(book(), "Geopolitical", 4, 7)
    assert not s["triggered"] and s["loss_usd"] == 0 and s["after_usd"] == s["before_usd"]


def test_positive_sentiment_does_not_trigger_when_required_negative():
    _, s = stress_portfolio(book(), "Earnings/Financial Results", 8, sentiment=0.8, require_negative=True)
    assert not s["triggered"]


def test_loss_matches_before_minus_after():
    _, s = stress_portfolio(book(), "Credit Event", 9)
    assert s["before_usd"] - s["after_usd"] == pytest.approx(s["loss_usd"])


def test_rate_shock_hits_long_duration_bonds_harder():
    d, _ = stress_portfolio(book(), "Macroeconomic", 7)
    bond = d.set_index("asset").loc["Corp bond", "shock_pct"]
    loan = d.set_index("asset").loc["Loan", "shock_pct"]
    assert bond < loan < 0   # 100bp rate shock * 5y duration dominates


def test_flight_to_quality_helps_rate_sensitive_bonds():
    # geopolitical shock lowers rates, so a pure-duration government bond gains
    gov = pd.DataFrame([{"asset": "Gov", "asset_type": "bond", "notional_usd": 100,
                         "duration_years": 7, "spread_duration": 0}])
    d, _ = stress_portfolio(gov, "Geopolitical", 7)
    assert d.loc[0, "shock_pct"] > 0


def test_severity_scale_bounds():
    assert severity_scale(5) == 1.0 and severity_scale(20) == 1.5 and 1.0 < severity_scale(10) < 1.5


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        stress_portfolio(pd.DataFrame({"asset": ["x"]}), "Geopolitical", 8)
