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


def _sample_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-12-31",
                "track": "SA",
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
