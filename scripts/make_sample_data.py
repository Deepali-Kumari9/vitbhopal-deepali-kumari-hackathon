"""Generate the synthetic news + social datasets used by the demo.

Everything here is made up: the companies are fictional (see data/companies.csv)
and the headlines/posts are assembled from templates. Run:

    python scripts/make_sample_data.py
"""
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
random.seed(42)

companies = pd.read_csv(ROOT / "data" / "companies.csv")

NEWS_NEG = [
    "{c} faces a credit downgrade after rising loan losses and liquidity concerns",
    "Regulator opens fraud investigation into {c} accounting practices",
    "{c} warns of sharp decline in quarterly profit",
    "{c} hit by ransomware attack, data breach under investigation",
    "New sanctions and conflict disrupt {c} supply chains",
    "{c} misses revenue forecast as sales fall",
    "{c} faces lawsuit over product recall",
    "{c} delays expansion amid recession fears",
]
NEWS_POS = [
    "{c} reports record quarterly profit and raises its full-year forecast",
    "{c} announces new product launch and strategic partnership",
    "{c} wins regulatory approval, shares rally",
    "{c} agrees to acquire a rival in a strong expansion move",
    "{c} upgraded by analysts after strong revenue growth",
    "{c} beats earnings expectations as sales surge",
]
NEWS_NEU = [
    "{c} schedules annual shareholder meeting",
    "{c} appoints new head of investor relations",
]
SOC_NEG = [
    "${t} looks rough, default talk again. not touching it #markets",
    "worried about ${t} after the probe news. this looks bad",
    "${t} plunges on weak numbers, ugh",
    "heard {c} got hit by a breach?? ${t} could drop hard",
    "${t} lawsuit risk is real, bearish for now",
]
SOC_POS = [
    "${t} earnings beat, strong growth. bullish",
    "{c} launch looks promising, ${t} might rally from here",
    "${t} upgrade today, record profit coming?",
    "love the partnership news for ${t}, positive for the whole sector",
]
SOC_NEU = [
    "anyone following ${t} ahead of the shareholder meeting?",
    "${t} chart is flat today, nothing happening",
]
MACRO_NEWS = [
    "Central bank signals higher interest rates as inflation remains elevated",
    "Fed holds rates steady as GDP growth stays strong",
    "Geopolitical conflict escalates as new tariffs hit imports",
    "Unemployment rises as recession fears grow across major economies",
    "Oil prices jump after sanctions tighten on exporters",
]
MACRO_SOC = [
    "inflation and recession worries are rising, rates may go up again #macro",
    "war and sanctions could cause a sharp drop in energy supply",
]

# probability a company's story is negative (builds a visible drift in the demo)
NEG_BIAS = {"NSTB": 0.8, "HBRT": 0.7, "SKLN": 0.65, "MRDE": 0.6,
            "ORMO": 0.15, "ATCL": 0.2, "CRPH": 0.25, "SLRA": 0.2}

start = datetime(2026, 10, 5, 8, 0)
rows = []


def when():
    return start + timedelta(hours=random.randint(0, 4 * 24 + 10), minutes=random.choice([0, 7, 15, 22, 41, 53]))


for _, co in companies.iterrows():
    p_neg = NEG_BIAS.get(co.ticker, 0.4)
    for kind, neg, pos, neu, n in (("news", NEWS_NEG, NEWS_POS, NEWS_NEU, 3), ("social", SOC_NEG, SOC_POS, SOC_NEU, 4)):
        for _ in range(n):
            r = random.random()
            pool = neg if r < p_neg else pos if r < p_neg + (1 - p_neg) * 0.8 else neu
            text = random.choice(pool).format(c=co.company, t=co.ticker)
            rows.append({"source": kind, "text": text, "ts": when()})

for t in MACRO_NEWS:
    rows.append({"source": "news", "text": t, "ts": when()})
for t in MACRO_SOC:
    rows.append({"source": "social", "text": t, "ts": when()})

df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
df["timestamp"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M")
df["id"] = df.groupby("source").cumcount() + 1
df["id"] = [f"{'N' if s == 'news' else 'S'}{i:03d}" for s, i in zip(df["source"], df["id"])]

news = df[df.source == "news"][["id", "timestamp", "text"]]
news.insert(2, "outlet", "synthetic-newswire")
social = df[df.source == "social"][["id", "timestamp", "text"]]
social.insert(2, "platform", "synthetic-microblog")
news.to_csv(ROOT / "data" / "news_headlines.csv", index=False)
social.to_csv(ROOT / "data" / "social_posts.csv", index=False)
print(f"news: {len(news)} rows, social: {len(social)} rows")
