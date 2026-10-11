"""Loaders for the two text sources: news headlines and social posts (plus optional RSS)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
    "https://www.investing.com/rss/news_25.rss",
]


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_news(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA / "news_headlines.csv")
    return pd.DataFrame({"timestamp": df["timestamp"], "source": "news",
                         "text": df["text"].map(clean_text)})


def load_social(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA / "social_posts.csv")
    return pd.DataFrame({"timestamp": df["timestamp"], "source": "social",
                         "text": df["text"].map(clean_text)})


def fetch_rss(limit_per_feed: int = 15) -> pd.DataFrame:
    """Pull live public headlines. Returns an empty frame if feeds are unreachable."""
    import feedparser
    rows = []
    for url in RSS_FEEDS:
        try:
            parsed = feedparser.parse(url)
        except Exception:
            continue
        for entry in parsed.entries[:limit_per_feed]:
            title = clean_text(getattr(entry, "title", ""))
            if title:
                rows.append({"timestamp": getattr(entry, "published", ""),
                             "source": "news-rss", "text": title})
    return pd.DataFrame(rows, columns=["timestamp", "source", "text"])


def load_all(include_rss: bool = False) -> pd.DataFrame:
    frames = [load_news(), load_social()]
    if include_rss:
        frames.append(fetch_rss())
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df
