from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
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

CACHE_ROOT = Path("data/track_cache")

BRISNET_CARD_ALIASES = {
    "date": ["date", "race_date", "racedate", "dt"],
    "track": ["track", "track_code", "trk"],
    "race_number": ["race_number", "race", "race_num", "raceno", "race#"],
    "horse": ["horse", "horse_name", "name"],
    "surface": ["surface", "surf"],
    "distance": ["distance", "dist", "distance_furlongs"],
    "field_size": ["field_size", "field", "fldsz"],
    "timeform_speed": ["timeform_speed", "tf_speed", "speed_rating", "speed"],
    "drf_speed": ["drf_speed", "bris_speed", "bris_speed", "speed_figure"],
    "timeform_pace": ["timeform_pace", "tf_pace", "pace_rating", "pace"],
    "drf_pace": ["drf_pace", "bris_pace", "pace_figure"],
    "pedigree_surface_fit": ["pedigree_surface_fit", "surface_fit", "ped_surface_fit"],
    "pedigree_distance_fit": ["pedigree_distance_fit", "distance_fit", "ped_distance_fit"],
}

BRISNET_HISTORY_ALIASES = {
    "date": ["date", "race_date", "racedate", "dt"],
    "track": ["track", "track_code", "trk"],
    "race_number": ["race_number", "race", "race_num", "raceno", "race#"],
    "surface": ["surface", "surf"],
    "distance": ["distance", "dist", "distance_furlongs"],
    "field_size": ["field_size", "field", "fldsz"],
    "winner_speed": ["winner_speed", "win_speed", "speed", "speed_figure"],
    "winner_pace": ["winner_pace", "win_pace", "pace", "pace_figure"],
    "winner_running_style": ["winner_running_style", "run_style", "running_style", "style"],
}

BRISNET_POSITIONAL_CARD_COLUMNS = [
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
]

BRISNET_POSITIONAL_HISTORY_COLUMNS = [
    "date",
    "track",
    "race_number",
    "surface",
    "distance",
    "field_size",
    "winner_speed",
    "winner_pace",
    "winner_running_style",
]


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


def _normalize_column_name(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum() or ch == "_")


def _rename_using_aliases(df: pd.DataFrame, aliases: Dict[str, List[str]]) -> pd.DataFrame:
    normalized_to_original = {_normalize_column_name(col): col for col in df.columns}
    renamed = df.copy()
    for canonical, alias_list in aliases.items():
        if canonical in renamed.columns:
            continue
        for alias in alias_list:
            matched_col = normalized_to_original.get(_normalize_column_name(alias))
            if matched_col and matched_col in renamed.columns:
                renamed = renamed.rename(columns={matched_col: canonical})
                break
    return renamed


def _apply_positional_schema(df: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
    if not df.columns.to_series().astype(str).str.startswith("Unnamed").all():
        return df
    if df.shape[1] < len(target_columns):
        return df

    renamed = df.copy()
    renamed.columns = target_columns + [f"extra_{idx}" for idx in range(df.shape[1] - len(target_columns))]
    return renamed


def _read_brisnet_text(file_bytes: bytes) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="ignore")
    for sep in [",", "|", "\t", ";"]:
        try:
            parsed = pd.read_csv(io.StringIO(text), sep=sep)
        except Exception:
            continue
        if parsed.shape[1] > 1:
            return parsed

    try:
        whitespace_df = pd.read_csv(io.StringIO(text), sep=r"\s{2,}", engine="python")
        if whitespace_df.shape[1] > 1:
            return whitespace_df
    except Exception:
        pass

    try:
        fixed = pd.read_fwf(io.StringIO(text))
        if fixed.shape[1] > 1:
            return fixed
    except Exception:
        pass

    return pd.DataFrame()


def load_brisnet_file(file_bytes: bytes, extension: str, history: bool = False) -> pd.DataFrame:
    parsed = _read_brisnet_text(file_bytes)
    if parsed.empty:
        return parsed

    parsed.columns = [str(col).strip() for col in parsed.columns]

    if history:
        parsed = _apply_positional_schema(parsed, BRISNET_POSITIONAL_HISTORY_COLUMNS)
        return _rename_using_aliases(parsed, BRISNET_HISTORY_ALIASES)

    parsed = _apply_positional_schema(parsed, BRISNET_POSITIONAL_CARD_COLUMNS)
    return _rename_using_aliases(parsed, BRISNET_CARD_ALIASES)


def validate_card_columns(df: pd.DataFrame) -> List[str]:
    return sorted(REQUIRED_COLUMNS.difference(df.columns))


def validate_history_columns(df: pd.DataFrame) -> List[str]:
    return sorted(REQUIRED_HISTORY_COLUMNS.difference(df.columns))


def sanitize_track_code(track: str) -> str:
    value = "".join(ch for ch in str(track).upper().strip() if ch.isalnum())
    return value or "UNKNOWN"


def _cache_file_for_track(track: str) -> Path:
    return CACHE_ROOT / f"{sanitize_track_code(track)}.csv"


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
    working["track"] = working["track"].astype(str).apply(sanitize_track_code)

    return working


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    history = df.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")

    numeric_columns = ["distance", "field_size", "winner_speed", "winner_pace"]
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    history["surface"] = history["surface"].astype(str).str.lower().str.strip()
    history["track"] = history["track"].astype(str).apply(sanitize_track_code)
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


def load_track_cache(track: str) -> pd.DataFrame:
    path = _cache_file_for_track(track)
    if not path.exists():
        return pd.DataFrame(columns=sorted(REQUIRED_HISTORY_COLUMNS | {"winner_running_style"}))

    cached = pd.read_csv(path)
    return prepare_history(cached)


def update_track_cache(history_df: pd.DataFrame) -> None:
    prepared = prepare_history(history_df)
    if prepared.empty:
        return

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for track, chunk in prepared.groupby("track"):
        cache_path = _cache_file_for_track(track)
        if cache_path.exists():
            existing = prepare_history(pd.read_csv(cache_path))
            merged = pd.concat([existing, chunk], ignore_index=True)
        else:
            merged = chunk.copy()

        merged = merged.drop_duplicates(
            subset=["date", "track", "race_number", "surface", "distance", "winner_speed", "winner_pace"]
        ).sort_values(["date", "race_number"])

        merged.to_csv(cache_path, index=False)


def history_for_tracks(card_df: pd.DataFrame, uploaded_history: Optional[pd.DataFrame]) -> pd.DataFrame:
    prepared_card = prepare_card(card_df)
    tracks = prepared_card["track"].dropna().unique().tolist()

    cached_frames = [load_track_cache(track) for track in tracks]
    cached = pd.concat(cached_frames, ignore_index=True) if cached_frames else pd.DataFrame()

    if uploaded_history is None or uploaded_history.empty:
        return cached

    prepared_upload = prepare_history(uploaded_history)
    return pd.concat([cached, prepared_upload], ignore_index=True).drop_duplicates()


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
