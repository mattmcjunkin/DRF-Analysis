from pathlib import Path

import pandas as pd

import analyzer
from analyzer import (
    ModelWeights,
    history_for_tracks,
    load_brisnet_file,
    sanitize_track_code,
    score_races,
    summarize_trends,
    update_track_cache,
    validate_history_columns,
)
import pandas as pd

from analyzer import ModelWeights, score_races, summarize_trends, validate_history_columns


def _sample_card() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "track": "SA",
                "race_number": 1,
                "horse": "A",
                "surface": "Dirt",
                "distance": 6,
                "field_size": 8,
                "timeform_speed": 90,
                "drf_speed": 88,
                "timeform_pace": 87,
                "drf_pace": 85,
                "pedigree_surface_fit": 0.7,
                "pedigree_distance_fit": 0.6,
            },
            {
                "date": "2026-01-02",
                "track": "SA",
                "race_number": 1,
                "horse": "B",
                "surface": "Dirt",
                "distance": 6,
                "field_size": 8,
                "timeform_speed": 86,
                "drf_speed": 84,
                "timeform_pace": 82,
                "drf_pace": 83,
                "pedigree_surface_fit": 0.5,
                "pedigree_distance_fit": 0.55,
            },
        ]
    )


def _sample_history(track: str = "SA") -> pd.DataFrame:
def _sample_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-12-31",
                "track": track,
                "race_number": 1,
                "surface": "Dirt",
                "distance": 6,
                "field_size": 8,
                "winner_speed": 88,
                "winner_pace": 85,
                "winner_running_style": "stalker",
            }
        ]
    )


def test_score_races_probability_sums_to_one():
    scored = score_races(_sample_card(), _sample_history(), ModelWeights())
    total = scored["win_probability"].sum()
    assert round(total, 4) == 1.0
    assert scored.iloc[0]["predicted_rank"] == 1.0


def test_validate_history_columns_missing_required_fields():
    missing = validate_history_columns(pd.DataFrame({"date": ["2026-01-01"]}))
    assert "winner_speed" in missing
    assert "track" in missing


def test_summarize_trends_returns_expected_keys():
    trends = summarize_trends(_sample_history())
    assert set(trends.keys()) == {"surface_bias", "distance_buckets", "pace_shape"}
    assert not trends["surface_bias"].empty


def test_sanitize_track_code():
    assert sanitize_track_code(" sa-") == "SA"


def test_track_cache_isolated(tmp_path: Path):
    original_cache_root = analyzer.CACHE_ROOT
    analyzer.CACHE_ROOT = tmp_path / "track_cache"
    try:
        update_track_cache(_sample_history("SA"))
        update_track_cache(_sample_history("BEL"))

        card = _sample_card()
        card["track"] = "SA"
        history = history_for_tracks(card, None)

        assert set(history["track"].unique()) == {"SA"}
        assert (analyzer.CACHE_ROOT / "SA.csv").exists()
        assert (analyzer.CACHE_ROOT / "BEL.csv").exists()
    finally:
        analyzer.CACHE_ROOT = original_cache_root


def test_load_brisnet_card_file_from_pipe_text():
    content = """date|track|race_number|horse|surface|distance|field_size|timeform_speed|drf_speed|timeform_pace|drf_pace|pedigree_surface_fit|pedigree_distance_fit\n2026-01-02|SA|1|A|Dirt|6|8|90|88|87|85|0.7|0.6\n"""
    parsed = load_brisnet_file(content.encode("utf-8"), extension="drf", history=False)
    assert not parsed.empty
    assert set(["horse", "drf_speed", "timeform_speed"]).issubset(parsed.columns)


def test_load_brisnet_history_file_aliases():
    content = """race_date,track_code,race_num,surf,dist,field,win_speed,win_pace,run_style\n2026-01-01,BEL,5,Turf,8.5,10,92,88,closer\n"""
    parsed = load_brisnet_file(content.encode("utf-8"), extension="dr3", history=True)
    assert not parsed.empty
    assert set(["date", "track", "race_number", "winner_speed", "winner_pace"]).issubset(parsed.columns)
