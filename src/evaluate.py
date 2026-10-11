"""Score the engine against the small hand-labelled set in data/labeled_eval.csv.

    python -m src.evaluate

Reports accuracy and macro-F1 for sentiment and event type, next to a naive
majority-class baseline so the numbers have something to be compared against.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .risk_engine import analyze_text

ROOT = Path(__file__).resolve().parent.parent


def macro_f1(y_true, y_pred) -> float:
    f1s = []
    for c in sorted(set(y_true)):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def accuracy(y_true, y_pred) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def evaluate(path: Path | None = None) -> dict:
    df = pd.read_csv(path or ROOT / "data" / "labeled_eval.csv")
    preds = [analyze_text(t) for t in df["text"]]
    out = {"n_examples": len(df)}
    for name, col, key in (("sentiment", "sentiment_label", "sentiment_label"),
                           ("event_type", "event_type", "event_type")):
        y_true = df[col].tolist()
        y_pred = [p[key] for p in preds]
        majority = Counter(y_true).most_common(1)[0][0]
        out[name] = {
            "accuracy": round(accuracy(y_true, y_pred), 3),
            "macro_f1": round(macro_f1(y_true, y_pred), 3),
            "majority_baseline_accuracy": round(accuracy(y_true, [majority] * len(y_true)), 3),
            "confusion": pd.crosstab(pd.Series(y_true, name="actual"),
                                     pd.Series(y_pred, name="predicted")).to_dict(),
        }
    df["pred_sentiment"] = [p["sentiment_label"] for p in preds]
    df["pred_event"] = [p["event_type"] for p in preds]
    out["errors"] = df[(df.sentiment_label != df.pred_sentiment) | (df.event_type != df.pred_event)] \
        .to_dict("records")
    return out


if __name__ == "__main__":
    res = evaluate()
    print(f"examples: {res['n_examples']}")
    for k in ("sentiment", "event_type"):
        r = res[k]
        print(f"{k:11s} accuracy={r['accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}  "
              f"majority-baseline={r['majority_baseline_accuracy']:.3f}")
    out_path = ROOT / "docs" / "evaluation_results.json"
    out_path.write_text(json.dumps({k: v for k, v in res.items() if k != "errors"}, indent=2))
    print(f"saved {out_path}")
