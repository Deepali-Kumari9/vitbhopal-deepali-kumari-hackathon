"""End-to-end run: ingest -> analyze -> write structured signals to disk.

    python -m src.pipeline                # writes outputs/risk_signals.{csv,json}
    python -m src.pipeline --rss          # also pull live RSS headlines
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .ingestion import load_all
from .risk_engine import analyze_records

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

SIGNAL_COLUMNS = ["timestamp", "source", "company", "ticker", "sentiment_score",
                  "sentiment_label", "event_type", "impact_score", "high_impact",
                  "matched_terms", "text"]


def build_signals(include_rss: bool = False) -> pd.DataFrame:
    raw = load_all(include_rss)
    signals = pd.DataFrame(analyze_records(raw.to_dict("records")))
    return signals[SIGNAL_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def write_signals(signals: pd.DataFrame, out_dir: Path = OUT) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / "risk_signals.csv", out_dir / "risk_signals.json"
    signals.to_csv(csv_path, index=False)
    signals.to_json(json_path, orient="records", indent=2)
    return csv_path, json_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the RiskPulse text -> signal pipeline")
    ap.add_argument("--rss", action="store_true", help="include live public RSS headlines")
    args = ap.parse_args()
    sig = build_signals(args.rss)
    csv_path, json_path = write_signals(sig)
    print(f"{len(sig)} signals written to {csv_path} and {json_path}")
    print(sig["event_type"].value_counts().to_string())
