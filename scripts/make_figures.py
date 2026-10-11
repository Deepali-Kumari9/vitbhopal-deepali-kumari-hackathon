"""Build the architecture diagram and result charts used in docs/ and the deck."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import build_signals
from src.rebalancer import latest_weights, load_constituents, rebalance_history
from src.stress_test import stress_portfolio

DOCS = ROOT / "docs"
(DOCS / "img").mkdir(exist_ok=True)
NAVY, TEAL, RED, GREY, LIGHT = "#0B1F3A", "#2A9D8F", "#E63946", "#8D99AE", "#F1F5F9"


def box(ax, x, y, w, h, text, fc, tc="white", size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=size, weight="bold")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.5))


def architecture():
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=200)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis("off")
    fig.patch.set_facecolor("white")
    # sources
    box(ax, 0.3, 4.4, 2.6, 0.9, "News headlines\n(CSV / live RSS)", NAVY)
    box(ax, 0.3, 2.6, 2.6, 0.9, "Social posts\n(CSV, cashtags)", NAVY)
    box(ax, 0.3, 0.8, 2.6, 0.9, "Manual text\n(dashboard / API)", NAVY)
    # ingestion
    box(ax, 3.6, 2.4, 2.2, 1.4, "Ingestion\nclean + dedupe\nsrc/ingestion.py", "#1D3557", size=9)
    # engine
    ax.add_patch(FancyBboxPatch((6.4, 1.2), 3.2, 3.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=LIGHT, ec=TEAL, lw=2))
    ax.text(8.0, 4.75, "AI / NLP Risk Engine", ha="center", color=NAVY, fontsize=11, weight="bold")
    ax.text(8.0, 4.45, "src/risk_engine.py", ha="center", color=GREY, fontsize=8)
    box(ax, 6.65, 3.5, 2.7, 0.7, "Sentiment  [-1, 1]", TEAL, size=9)
    box(ax, 6.65, 2.65, 2.7, 0.7, "Event classification", TEAL, size=9)
    box(ax, 6.65, 1.8, 2.7, 0.7, "Impact score  [1-10]", TEAL, size=9)
    # signals
    box(ax, 10.1, 2.4, 1.6, 1.4, "Structured\nsignals\nCSV / JSON / API", NAVY, size=9)
    # modules
    box(ax, 12.1, 3.9, 1.7, 1.3, "Module A\nIndex\nrebalancer", RED, size=9)
    box(ax, 12.1, 0.6, 1.7, 1.3, "Module B\nStress\ntest", RED, size=9)
    box(ax, 12.1, 2.45, 1.7, 1.0, "Streamlit\ndashboard", "#457B9D", size=9)
    for y in (4.85, 3.05, 1.25):
        arrow(ax, 2.9, y, 3.6, 3.1)
    arrow(ax, 5.8, 3.1, 6.4, 3.1); arrow(ax, 9.6, 3.1, 10.1, 3.1)
    arrow(ax, 11.7, 3.3, 12.1, 4.5); arrow(ax, 11.7, 2.85, 12.1, 1.6); arrow(ax, 11.7, 3.1, 12.1, 2.95)
    ax.text(7.0, 0.35, "Sentiment drives Module A weights  |  Event type + impact drive Module B shocks",
            ha="center", color=GREY, fontsize=9)
    fig.savefig(DOCS / "architecture.png", bbox_inches="tight"); plt.close(fig)


def results():
    sig = build_signals(); c = load_constituents()
    h, _ = rebalance_history(sig, c)
    # weights over time
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    pv = h.pivot(index="timestamp", columns="ticker", values="weight") * 100
    lw = latest_weights(h, c)
    top = lw.head(3)["ticker"].tolist() + lw.tail(3)["ticker"].tolist()
    for t in pv.columns:
        ax.step(pv.index, pv[t], where="post", color="#CBD5E1", lw=0.9)
    for t in top:
        ax.step(pv.index, pv[t], where="post", lw=2.2, label=t, color=TEAL if t in top[:3] else RED, alpha=1 if t in (top[0], top[-1]) else 0.55)
        if t in (top[0], top[-1]):
            ax.text(pv.index[-1], pv[t].iloc[-1], f" {t}", fontsize=8, va="center")
    ax.axhline(100 / 15, color=GREY, ls="--", lw=1)
    ax.set_ylabel("Weight (%)"); ax.set_title("Mock index weights drift with sentiment", fontsize=11, loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.savefig(DOCS / "img" / "weights.png", bbox_inches="tight"); plt.close(fig)
    # stress
    p = pd.read_csv(ROOT / "data" / "synthetic_portfolio.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    evs = ["Geopolitical", "Credit Event", "Macroeconomic", "Legal/Regulatory", "Cybersecurity"]
    losses = [stress_portfolio(p, e, 8)[1]["loss_pct"] for e in evs]
    ax.barh(evs[::-1], losses[::-1], color=RED)
    for i, v in enumerate(losses[::-1]): ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(losses) * 1.2); ax.set_xlabel("Portfolio loss (%) at impact score 8")
    ax.set_title(f"Stress loss on a ${p.notional_usd.sum() / 1e6:.0f}M synthetic book", fontsize=11, loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.savefig(DOCS / "img" / "stress.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    architecture(); results(); print("figures saved to docs/")
