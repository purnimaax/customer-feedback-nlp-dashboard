from __future__ import annotations

from typing import Iterable

import pandas as pd


TEXT_COLUMN_CANDIDATES = (
    "reviewText",
    "feedback_text",
    "text",
    "tweet_text",
    "post_text",
    "comment",
    "body",
    "content",
)
RATING_COLUMN_CANDIDATES = ("overall", "rating", "score", "stars")
SOURCE_COLUMN_CANDIDATES = ("source", "platform")
DEFAULT_RATING = 3


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def normalize_feedback_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize review, feedback, and social CSV exports to dashboard columns.
    """
    text_column = _first_present(df.columns, TEXT_COLUMN_CANDIDATES)
    if text_column is None:
        expected = ", ".join(TEXT_COLUMN_CANDIDATES)
        raise ValueError(f"Input CSV must contain one text column: {expected}")

    normalized = df.copy()
    normalized["reviewText"] = normalized[text_column].fillna("").astype(str)

    rating_column = _first_present(df.columns, RATING_COLUMN_CANDIDATES)
    if rating_column is None:
        normalized["overall"] = DEFAULT_RATING
    else:
        ratings = pd.to_numeric(normalized[rating_column], errors="coerce")
        normalized["overall"] = ratings.fillna(DEFAULT_RATING).clip(1, 5)

    source_column = _first_present(df.columns, SOURCE_COLUMN_CANDIDATES)
    if source_column is None:
        normalized["source"] = "csv"
    else:
        normalized["source"] = normalized[source_column].fillna("csv").astype(str)

    normalized["source_text_column"] = text_column
    return normalized
