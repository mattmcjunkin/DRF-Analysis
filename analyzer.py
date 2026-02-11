from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "track",
    "race_number",
    "horse",
    "surface",
    "distance",
    "field_size",
    "timeform_speed",
    "drf_speed",
    "timeform_pace",
    "drf_pace",
    "pedigree_surface_fit",
    "pedigree_distance_fit",
}

REQUIRED_HISTORY_COLUMNS = {
    "date",
    "track",
    "race_number",
    "surface",
    "distance",
    "field_size",
    "winner_speed",
    "winner_pace",
}


@dataclass
class ModelWeights:
    speed: float = 0.35
    pace: float = 0.2
    surface_fit: float = 0.15
    distance_fit: float = 0.15
    field_size_pressure: float = 0.1
    trend_bias: float = 0.05


def normalize_series(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


def validate_card_columns(df: pd.DataFrame) -> List[str]:
    return sorted(REQUIRED_COLUMNS.difference(df.columns))


def validate_history_columns(df: pd.DataFrame) -> List[str]:
    return sorted(REQUIRED_HISTORY_COLUMNS.difference(df.columns))


def prepare_card(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")

    numeric_columns = [
        "distance",
        "field_size",
        "timeform_speed",
        "drf_speed",
        "timeform_pace",
        "drf_pace",
        "pedigree_surface_fit",
        "pedigree_distance_fit",
    ]
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working = working.dropna(subset=["date", "track", "race_number", "horse", "surface"])

    working["combined_speed"] = working[["timeform_speed", "drf_speed"]].mean(axis=1)
    working["combined_pace"] = working[["timeform_pace", "drf_pace"]].mean(axis=1)
    working["surface"] = working["surface"].astype(str).str.lower().str.strip()
    working["track"] = working["track"].astype(str).str.upper().str.strip()

    return working


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    history = df.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")

    numeric_columns = ["distance", "field_size", "winner_speed", "winner_pace"]
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    history["surface"] = history["surface"].astype(str).str.lower().str.strip()
    history["track"] = history["track"].astype(str).str.upper().str.strip()
    history = history.dropna(
        subset=["date", "track", "race_number", "surface", "distance", "winner_speed", "winner_pace"]
    )

    if "winner_running_style" not in history.columns:
        history["winner_running_style"] = "unknown"
    else:
        history["winner_running_style"] = (
            history["winner_running_style"].fillna("unknown").astype(str).str.lower().str.strip()
        )

    return history


def summarize_trends(history_results: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if history_results.empty:
        return {
            "surface_bias": pd.DataFrame(),
            "distance_buckets": pd.DataFrame(),
            "pace_shape": pd.DataFrame(),
        }

    history = prepare_history(history_results)
    if history.empty:
        return {
            "surface_bias": pd.DataFrame(),
            "distance_buckets": pd.DataFrame(),
            "pace_shape": pd.DataFrame(),
        }

    surface_bias = (
        history.groupby(["track", "surface"], as_index=False)
        .agg(
            avg_win_speed=("winner_speed", "mean"),
            avg_win_pace=("winner_pace", "mean"),
            avg_field_size=("field_size", "mean"),
            races=("race_number", "count"),
        )
        .sort_values(["track", "surface", "races"], ascending=[True, True, False])
    )

    history["distance_bucket"] = pd.cut(
        history["distance"],
        bins=[0, 6, 8, 10, 20],
        labels=["sprint", "extended sprint", "route", "marathon"],
    )

    distance_buckets = (
        history.groupby(["track", "surface", "distance_bucket"], as_index=False)
        .agg(
            races=("race_number", "count"),
            avg_win_speed=("winner_speed", "mean"),
            avg_win_pace=("winner_pace", "mean"),
        )
        .sort_values("races", ascending=False)
    )

    pace_shape = (
        history.groupby(["track", "surface", "winner_running_style"], as_index=False)
        .agg(races=("race_number", "count"))
        .sort_values("races", ascending=False)
    )

    return {
        "surface_bias": surface_bias,
        "distance_buckets": distance_buckets,
        "pace_shape": pace_shape,
    }


def calculate_trend_bias(
    current_cards: pd.DataFrame,
    history_results: Optional[pd.DataFrame],
) -> pd.Series:
    if history_results is None or history_results.empty:
        return pd.Series(np.zeros(len(current_cards)), index=current_cards.index)

    history = prepare_history(history_results)
    if history.empty:
        return pd.Series(np.zeros(len(current_cards)), index=current_cards.index)

    trend_summary = (
        history.groupby(["track", "surface"], as_index=False)
        .agg(hist_speed=("winner_speed", "mean"), hist_pace=("winner_pace", "mean"))
    )

    merged = current_cards.merge(trend_summary, on=["track", "surface"], how="left")
    speed_gap = (merged["combined_speed"] - merged["hist_speed"]).fillna(0)
    pace_gap = (merged["combined_pace"] - merged["hist_pace"]).fillna(0)

    raw_bias = 0.6 * speed_gap + 0.4 * pace_gap
    return normalize_series(raw_bias)


def score_races(
    card_df: pd.DataFrame,
    history_results: Optional[pd.DataFrame] = None,
    weights: Optional[ModelWeights] = None,
) -> pd.DataFrame:
    weights = weights or ModelWeights()
    cards = prepare_card(card_df)

    cards["speed_score"] = normalize_series(cards["combined_speed"])
    cards["pace_score"] = normalize_series(cards["combined_pace"])
    cards["surface_fit_score"] = normalize_series(cards["pedigree_surface_fit"])
    cards["distance_fit_score"] = normalize_series(cards["pedigree_distance_fit"])
    cards["field_size_pressure_score"] = 1 - normalize_series(cards["field_size"])
    cards["trend_bias_score"] = calculate_trend_bias(cards, history_results)

    cards["prediction_score"] = (
        weights.speed * cards["speed_score"]
        + weights.pace * cards["pace_score"]
        + weights.surface_fit * cards["surface_fit_score"]
        + weights.distance_fit * cards["distance_fit_score"]
        + weights.field_size_pressure * cards["field_size_pressure_score"]
        + weights.trend_bias * cards["trend_bias_score"]
    )

    cards["win_probability"] = (
        cards.groupby(["date", "track", "race_number"])["prediction_score"]
        .transform(lambda race_scores: race_scores / race_scores.sum() if race_scores.sum() else 0)
        .round(4)
    )

    cards["predicted_rank"] = cards.groupby(["date", "track", "race_number"])[
        "prediction_score"
    ].rank(ascending=False, method="dense")

    return cards.sort_values(["date", "track", "race_number", "predicted_rank"])


def recommended_columns_text(columns: Iterable[str]) -> str:
    return "\n".join(f"- {col}" for col in sorted(columns))
