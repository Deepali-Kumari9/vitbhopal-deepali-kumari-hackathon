from src.risk_engine import (analyze_text, calculate_impact, classify_event,
                             extract_entity, score_sentiment)


def test_sentiment_bounds():
    assert -1 <= score_sentiment("fraud default crash bankruptcy") <= 1
    assert -1 <= score_sentiment("record profit growth surge rally") <= 1


def test_direction():
    assert score_sentiment("record profit growth") > 0
    assert score_sentiment("fraud default crash") < 0
    assert score_sentiment("the meeting was announced") == 0


def test_negation_flips_sign():
    assert score_sentiment("profit growth") > 0
    assert score_sentiment("no profit growth") < 0


def test_intensifier_increases_magnitude():
    assert abs(score_sentiment("sharp decline")) > abs(score_sentiment("decline"))


def test_event_classification():
    assert classify_event("War leads to new sanctions") == "Geopolitical"
    assert classify_event("Company reports quarterly earnings and revenue") == "Earnings/Financial Results"
    assert classify_event("Bank warns of default and liquidity concerns") == "Credit Event"
    assert classify_event("Ransomware attack hits retailer") == "Cybersecurity"


def test_word_boundaries_avoid_false_matches():
    assert classify_event("The hackathon winners were announced") == "General Market News"


def test_entity_extraction():
    assert extract_entity("Northstar Bank faces a probe") == ("Northstar Bank", "NSTB")
    assert extract_entity("$ORMO looks strong today")[1] == "ORMO"
    assert extract_entity("Markets were quiet") == ("Unspecified", "")


def test_signal_schema():
    r = analyze_text("Northstar Bank default risk rises", "test")
    for key in ["sentiment_score", "event_type", "impact_score", "source", "company", "ticker", "text"]:
        assert key in r
    assert 1 <= r["impact_score"] <= 10
    assert r["ticker"] == "NSTB"


def test_impact_bounds_and_ordering():
    assert 1 <= calculate_impact("war default crash", "Geopolitical", -1) <= 10
    assert calculate_impact("default", "Credit Event", -0.9) > calculate_impact("update", "General Market News", 0)
