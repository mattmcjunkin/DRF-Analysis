from __future__ import annotations

import pandas as pd
import streamlit as st

from analyzer import (
    REQUIRED_COLUMNS,
    REQUIRED_HISTORY_COLUMNS,
    ModelWeights,
    recommended_columns_text,
    score_races,
    summarize_trends,
    validate_card_columns,
    validate_history_columns,
)


def _build_weights() -> ModelWeights:
    with st.sidebar:
        st.header("Model Weights")
        speed_weight = st.slider("Speed", 0.0, 1.0, 0.35, 0.01)
        pace_weight = st.slider("Pace", 0.0, 1.0, 0.2, 0.01)
        surface_weight = st.slider("Surface pedigree fit", 0.0, 1.0, 0.15, 0.01)
        distance_weight = st.slider("Distance pedigree fit", 0.0, 1.0, 0.15, 0.01)
        field_weight = st.slider("Field size pressure", 0.0, 1.0, 0.1, 0.01)
        trend_weight = st.slider("Track trend bias", 0.0, 1.0, 0.05, 0.01)

        normalize = speed_weight + pace_weight + surface_weight + distance_weight + field_weight + trend_weight
        if normalize == 0:
            st.warning("All weights are zero. Predictions will be invalid.")

    return ModelWeights(
        speed=speed_weight / normalize if normalize else 0,
        pace=pace_weight / normalize if normalize else 0,
        surface_fit=surface_weight / normalize if normalize else 0,
        distance_fit=distance_weight / normalize if normalize else 0,
        field_size_pressure=field_weight / normalize if normalize else 0,
        trend_bias=trend_weight / normalize if normalize else 0,
    )


def main() -> None:
    st.set_page_config(page_title="DRF + Timeform Race Analyzer", layout="wide")
    st.title("🏇 DRF Formulator + Timeform US Race Card Analyzer")
    st.caption(
        "Upload today's cards plus historical race results to generate winner predictions and detect track/race-shape trends."
    )

    weights = _build_weights()

    st.subheader("1) Upload current race card")
    st.markdown("Expected columns:\n" + recommended_columns_text(REQUIRED_COLUMNS))
    card_file = st.file_uploader("Race card CSV", type=["csv"], key="card")

    st.subheader("2) Upload historical race results (optional but recommended)")
    st.markdown("Expected columns:\n" + recommended_columns_text(REQUIRED_HISTORY_COLUMNS | {"winner_running_style"}))
    history_file = st.file_uploader("Historical results CSV", type=["csv"], key="history")

    if not card_file:
        st.info("Upload a race card CSV to begin.")
        return

    cards_raw = pd.read_csv(card_file)
    missing_card_cols = validate_card_columns(cards_raw)
    if missing_card_cols:
        st.error(f"Missing required columns in race card: {', '.join(missing_card_cols)}")
        return

    history_df = None
    if history_file:
        history_df = pd.read_csv(history_file)
        missing_history_cols = validate_history_columns(history_df)
        if missing_history_cols:
            st.error(f"Missing required columns in historical results: {', '.join(missing_history_cols)}")
            return

    scored = score_races(cards_raw, history_df, weights)

    st.subheader("Predictions")
    st.dataframe(
        scored[
            [
                "date",
                "track",
                "race_number",
                "horse",
                "surface",
                "distance",
                "field_size",
                "combined_speed",
                "combined_pace",
                "prediction_score",
                "win_probability",
                "predicted_rank",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    contenders = scored[scored["predicted_rank"] == 1][
        ["date", "track", "race_number", "horse", "win_probability", "prediction_score"]
    ]
    st.subheader("Top projected winner per race")
    st.dataframe(contenders, use_container_width=True, hide_index=True)

    csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored race card",
        data=csv,
        file_name="scored_race_card.csv",
        mime="text/csv",
    )

    if history_df is not None and not history_df.empty:
        st.subheader("Trend analysis from historical uploads")
        trends = summarize_trends(history_df)

        tab1, tab2, tab3 = st.tabs(["Surface Bias", "Distance Buckets", "Winning Running Styles"])
        with tab1:
            st.dataframe(trends["surface_bias"], use_container_width=True, hide_index=True)
        with tab2:
            st.dataframe(trends["distance_buckets"], use_container_width=True, hide_index=True)
        with tab3:
            st.dataframe(trends["pace_shape"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
