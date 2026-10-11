from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluate import evaluate
from src.ingestion import clean_text, fetch_rss, load_news, load_social
from src.pipeline import SIGNAL_COLUMNS
from src.rebalancer import latest_weights, load_constituents, rebalance_history
from src.risk_engine import analyze_records, analyze_text
from src.stress_test import DEFAULT_SHOCKS, stress_portfolio

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="RiskPulse AI", page_icon="📈", layout="wide")
st.title("📈 RiskPulse AI")
st.caption("News + social text → structured risk signals → index rebalancing and portfolio stress testing")


@st.cache_data
def load_static():
    return (pd.read_csv(ROOT / "data" / "synthetic_portfolio.csv"), load_constituents())


portfolio, constituents = load_static()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    st.subheader("Stress test")
    threshold = st.slider("Impact threshold", 1, 10, 7,
                          help="A stress test fires when a signal's impact score is at or above this.")
    neg_only = st.checkbox("Only stress on negative sentiment", value=True)
    st.subheader("Index rebalancer")
    tilt = st.slider("Sentiment tilt strength", 0.0, 2.0, 1.0, 0.1)
    half_life = st.slider("Sentiment half-life (hours)", 6, 72, 24, 6)
    max_w = st.slider("Max weight per stock", 0.08, 0.30, 0.15, 0.01)
    min_w = st.slider("Min weight per stock", 0.00, 0.05, 0.02, 0.005)
    freq = st.selectbox("Rebalance frequency", ["1h", "3h", "6h", "12h", "1D"], index=2)
    st.subheader("Sources")
    use_rss = st.checkbox("Include live RSS headlines", value=False,
                          help="Needs internet. The bundled data works offline.")
    st.caption("Bundled news and social data are synthetic. No API keys required.")


# ------------------------------------------------------- build the signals
@st.cache_data(show_spinner=False)
def get_rss():
    return fetch_rss()


frames = [load_news(), load_social()]
if use_rss:
    rss = get_rss()
    if rss.empty:
        st.sidebar.warning("RSS returned nothing; using bundled data only.")
    else:
        frames.append(rss)
raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset="text")
signals_df = pd.DataFrame(analyze_records(raw.to_dict("records")))[SIGNAL_COLUMNS]
signals_df = signals_df.sort_values("timestamp").reset_index(drop=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Risk Monitor", "Index Rebalancer", "Stress Test", "Model Evaluation", "Methodology & API"])

# ------------------------------------------------------------ risk monitor
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Items analysed", len(signals_df))
    c2.metric("Negative signals", int((signals_df["sentiment_label"] == "Negative").sum()))
    c3.metric("High-impact (≥7)", int(signals_df["high_impact"].sum()))
    c4.metric("Sources", signals_df["source"].nunique())

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Try your own text**")
        company = st.text_input("Company (optional)", "")
        text = st.text_area("Text", "Northstar Bank faces a credit downgrade after rising loan losses "
                                    "and liquidity concerns", height=90)
        if st.button("Analyse", type="primary"):
            res = analyze_text(clean_text(text), "manual", company)
            st.json({k: res[k] for k in ["company", "ticker", "sentiment_score", "sentiment_label",
                                         "event_type", "impact_score", "high_impact", "matched_terms"]})
    with right:
        by_event = signals_df.groupby("event_type").agg(count=("text", "size"),
                                                        avg_impact=("impact_score", "mean")).reset_index()
        fig = px.bar(by_event.sort_values("count"), x="count", y="event_type", orientation="h",
                     color="avg_impact", color_continuous_scale="Reds", title="Signals by event type")
        fig.update_layout(height=320, yaxis_title="", xaxis_title="Signals", margin=dict(t=40, b=0))
        st.plotly_chart(fig, width="stretch")

    f1, f2 = st.columns(2)
    src_filter = f1.multiselect("Source", sorted(signals_df["source"].unique()),
                                default=sorted(signals_df["source"].unique()))
    min_imp = f2.slider("Minimum impact", 1, 10, 1)
    view = signals_df[signals_df["source"].isin(src_filter) & (signals_df["impact_score"] >= min_imp)]
    st.dataframe(view[["timestamp", "source", "ticker", "sentiment_score", "event_type",
                       "impact_score", "text"]], width="stretch", hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download CSV", signals_df.to_csv(index=False).encode(),
                       "risk_signals.csv", "text/csv")
    d2.download_button("Download JSON", signals_df.to_json(orient="records", indent=2).encode(),
                       "risk_signals.json", "application/json")

# ------------------------------------------------------------- Module A
with tab2:
    st.subheader("Tactical index rebalancing (Module A)")
    st.caption("Mock 15-stock index of fictional companies. Positive sentiment raises a stock's weight, "
               "negative lowers it, and the effect fades with the half-life you choose.")
    history, turnover = rebalance_history(signals_df, constituents, freq=freq, tilt=tilt,
                                          half_life_hours=half_life, min_weight=min_w, max_weight=max_w)
    if history.empty:
        st.info("No signals mapped to index constituents yet.")
    else:
        lw = latest_weights(history, constituents)
        m1, m2, m3 = st.columns(3)
        m1.metric("Rebalance steps", history["timestamp"].nunique())
        m2.metric("Biggest overweight", f"{lw.iloc[0]['ticker']}  {lw.iloc[0]['change_pp']:+.1f}pp")
        worst = lw.sort_values("change_pp").iloc[0]
        m3.metric("Biggest underweight", f"{worst['ticker']}  {worst['change_pp']:+.1f}pp")

        hist_pct = history.assign(weight_pct=history["weight"] * 100)
        fig = px.line(hist_pct, x="timestamp", y="weight_pct", color="ticker", line_shape="hv",
                      title="Index weights over time (%)")
        fig.update_layout(height=420, yaxis_title="Weight (%)", xaxis_title="")
        st.plotly_chart(fig, width="stretch")

        cA, cB = st.columns([3, 2])
        with cA:
            bar = go.Figure()
            bar.add_bar(x=lw["ticker"], y=lw["base_weight"] * 100, name="Base (equal weight)")
            bar.add_bar(x=lw["ticker"], y=lw["current_weight"] * 100, name="After sentiment tilt")
            bar.update_layout(barmode="group", height=340, title="Base vs current weights (%)",
                              yaxis_title="Weight (%)", margin=dict(t=40))
            st.plotly_chart(bar, width="stretch")
        with cB:
            tv = px.bar(turnover, x="timestamp", y="turnover", title="Turnover per rebalance")
            tv.update_layout(height=340, yaxis_tickformat=".0%", margin=dict(t=40), xaxis_title="")
            st.plotly_chart(tv, width="stretch")
        show = lw.assign(base_weight=(lw["base_weight"] * 100).round(2),
                         current_weight=(lw["current_weight"] * 100).round(2),
                         change_pp=lw["change_pp"].round(2))
        st.dataframe(show, width="stretch", hide_index=True)

# ------------------------------------------------------------- Module B
with tab3:
    st.subheader("Event-driven stress test (Module B)")
    options = [f"{i + 1}. [{r.impact_score}/10] {r.event_type} - {r.text[:75]}" for i, r in signals_df.iterrows()]
    default = int(signals_df["impact_score"].idxmax()) if len(signals_df) else 0
    chosen = st.selectbox("Choose a detected event", options, index=default)
    row = signals_df.iloc[options.index(chosen)]
    st.write(f"**{row['event_type']}** · impact **{row['impact_score']}/10** · "
             f"sentiment **{row['sentiment_score']}** · source: {row['source']}")

    detail, s = stress_portfolio(portfolio, row["event_type"], int(row["impact_score"]), threshold,
                                 sentiment=float(row["sentiment_score"]), require_negative=neg_only)
    if s["triggered"]:
        st.error(f"Stress test triggered (shock scaled ×{s['severity_scale']:.2f} for severity).")
    else:
        st.warning(f"Not triggered: {s['reason']}.")

    a, b, c, d = st.columns(4)
    a.metric("Before", f"${s['before_usd'] / 1e6:,.2f}M")
    b.metric("After", f"${s['after_usd'] / 1e6:,.2f}M", delta=f"{(s['after_usd'] - s['before_usd']) / 1e6:,.2f}M")
    c.metric("Loss", f"${s['loss_usd'] / 1e6:,.2f}M")
    d.metric("Loss %", f"{s['loss_pct']:.2f}%")

    g1, g2 = st.columns(2)
    with g1:
        ba = go.Figure(go.Bar(x=["Before", "After"], y=[s["before_usd"] / 1e6, s["after_usd"] / 1e6],
                              marker_color=["#2a9d8f", "#e63946" if s["triggered"] else "#2a9d8f"]))
        ba.update_layout(title="Portfolio value ($M)", height=340, margin=dict(t=40))
        st.plotly_chart(ba, width="stretch")
    with g2:
        by_type = detail.groupby("asset_type").agg(before=("notional_usd", "sum"),
                                                   loss=("loss_usd", "sum")).reset_index()
        by_type["loss_pct"] = by_type["loss"] / by_type["before"] * 100
        lt = px.bar(by_type, x="asset_type", y="loss_pct", title="Loss by asset class (%)")
        lt.update_layout(height=340, margin=dict(t=40), xaxis_title="", yaxis_title="Loss (%)")
        st.plotly_chart(lt, width="stretch")

    out = detail.assign(shock_pct=(detail["shock_pct"] * 100).round(2))
    st.dataframe(out[["asset", "asset_type", "notional_usd", "shock_pct", "loss_usd", "stressed_value_usd"]],
                 width="stretch", hide_index=True)
    st.download_button("Download stress results", detail.to_csv(index=False).encode(),
                       "stress_test_results.csv", "text/csv")
    with st.expander("Scenario shock assumptions"):
        st.json(DEFAULT_SHOCKS.get(row["event_type"], DEFAULT_SHOCKS["General Market News"]))
        st.caption("Bond: -(duration × rate shift + spread duration × spread widening). "
                   "Loan: -(spread duration × spread widening) - expected credit loss. "
                   "Equity and derivatives: flat shock. Calibrated at impact 7, scaled up to ×1.5.")

# -------------------------------------------------------------- evaluation
with tab4:
    st.subheader("How well does the engine do?")
    res = evaluate()
    st.write(f"Scored against **{res['n_examples']}** hand-labelled headlines (`data/labeled_eval.csv`), "
             "compared with a naive majority-class baseline.")
    e1, e2 = st.columns(2)
    for col, key, title in ((e1, "sentiment", "Sentiment (Pos/Neg/Neutral)"), (e2, "event_type", "Event type")):
        r = res[key]
        col.markdown(f"**{title}**")
        col.metric("Accuracy", f"{r['accuracy']:.1%}", delta=f"{(r['accuracy'] - r['majority_baseline_accuracy']) * 100:+.1f}pp vs baseline")
        col.metric("Macro-F1", f"{r['macro_f1']:.2f}")
    st.markdown("**Misclassified examples**")
    errs = pd.DataFrame(res["errors"])
    st.dataframe(errs[["text", "sentiment_label", "pred_sentiment", "event_type", "pred_event"]],
                 width="stretch", hide_index=True)
    st.caption("Small set, labelled by hand. Treat the numbers as indicative, not a benchmark.")

# ----------------------------------------------------------- methodology
with tab5:
    st.subheader("Methodology")
    st.markdown("""
**Sentiment.** Word lexicon scores summed with negation (3-word window) and intensifier handling, squashed with `tanh` into [-1, 1].
**Event type.** Word-boundary keyword matching over 8 event classes, otherwise *General Market News*.
**Impact (1-10).** Event base severity + severity keywords + a bonus for strong sentiment. This is a heuristic, not trained on market returns.
**Entities.** Cashtags (`$NSTB`) and company names are mapped to tickers in `data/companies.csv`.
**Rebalancer.** Per-stock sentiment state with exponential decay; weights = base × exp(tilt × state), capped and renormalised.
**Stress test.** Rule-based shocks per event type, applied via duration / spread-duration for credit-sensitive assets.

**Data.** All bundled news, social posts, companies and the portfolio are synthetic. RSS is optional.
""")
    st.markdown("**HTTP API** (run `uvicorn api:app`, docs at `/docs`)")
    st.code("curl -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' \\\n"
            "  -d '{\"text\": \"Regulator opens fraud investigation into Harbor Retail\"}'", language="bash")
    st.code(json.dumps(analyze_text("Regulator opens fraud investigation into Harbor Retail", "api"), indent=2),
            language="json")
    st.markdown("**Synthetic portfolio**")
    st.dataframe(portfolio, width="stretch", hide_index=True)
