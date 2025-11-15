from typing import Dict, Any
import os
import pandas as pd

# Always load CSV from project root: data/news.csv
CSV_PATH = "data/news.csv"


def _load_df() -> pd.DataFrame:
    print(f"[stats] Loading CSV from: {os.path.abspath(CSV_PATH)}")
    return pd.read_csv(CSV_PATH)


def stats_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("question", "")
    q = question.lower()

    df = _load_df()

    # -------------------------
    # 1) Positive / Negative Sentiment
    # -------------------------
    if "positive" in q and "negative" in q:
        vc = df["sentiment"].fillna("unknown").str.lower().value_counts()
        pos = int(vc.get("positive", 0))
        neg = int(vc.get("negative", 0))
        other = int(len(df) - pos - neg)

        state["answer"] = (
            "Sentiment statistics in the dataset:\n"
            f"- Positive news: {pos}\n"
            f"- Negative news: {neg}\n"
            f"- Neutral/other: {other}"
        )
        return state

    # -------------------------
    # 2) Count news before June 2025
    # -------------------------
    if "before june 2025" in q:
        date_cols = ["timestamp", "original_timestamp", "created_at"]
        available = [c for c in date_cols if c in df.columns]

        if not available:
            state["answer"] = (
                "I could not find a usable date column "
                "(e.g. 'timestamp', 'original_timestamp', 'created_at')."
            )
            return state

        # Build 1 combined date column
        dates = None
        for col in available:
            parsed = pd.to_datetime(df[col], errors="coerce")
            dates = parsed if dates is None else dates.fillna(parsed)

        cutoff = pd.Timestamp("2025-06-01")
        count = int((dates < cutoff).sum())

        state["answer"] = (
            f"There are {count} news articles dated before 1 June 2025 "
            f"based on the available timestamp columns."
        )
        return state

    # -------------------------
    # Default reply for unknown stats questions
    # -------------------------
    state["answer"] = (
        "I can currently answer only basic statistics such as:\n"
        "- Positive/negative sentiment counts\n"
        "- News before June 2025\n"
        "You may ask something like: 'How many positive and negative news are there?'"
    )
    return state