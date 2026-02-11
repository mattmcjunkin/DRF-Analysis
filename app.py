from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from analyzer import (
    CACHE_ROOT,
    REQUIRED_COLUMNS,
    REQUIRED_HISTORY_COLUMNS,
    ModelWeights,
    history_for_tracks,
    recommended_columns_text,
    score_races,
    summarize_trends,
    update_track_cache,
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


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _parse_text_table(text: str) -> pd.DataFrame:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for delimiter in [",", "|", "\t", ";"]:
        candidates = [line for line in lines if delimiter in line]
        if len(candidates) < 2:
            continue
        csv_like = "\n".join(candidates)
        parsed = pd.read_csv(io.StringIO(csv_like), sep=delimiter)
        if parsed.shape[1] > 1:
            parsed.columns = [str(col).strip() for col in parsed.columns]
            return parsed

    return pd.DataFrame()


def _load_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower().split(".")[-1]
    if suffix == "csv":
        return pd.read_csv(uploaded_file)
    if suffix == "pdf":
        text = _extract_pdf_text(uploaded_file.read())
        return _parse_text_table(text)
    return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="DRF + Timeform Race Analyzer", layout="wide")
    st.title("🏇 DRF Formulator + Timeform US Race Card Analyzer")
    st.caption(
        "Upload today's cards plus historical race results to generate winner predictions and detect track/race-shape trends."
    )

    st.info(
        "PDF support expects text-based PDFs with a delimiter-based table (comma, pipe, tab, semicolon). Scanned-image PDFs require OCR before upload."
    )

    with st.sidebar:
        st.caption(f"Per-track cache directory: `{CACHE_ROOT}`")

    weights = _build_weights()

    st.subheader("1) Upload current race card")
    st.markdown("Expected columns:\n" + recommended_columns_text(REQUIRED_COLUMNS))
    card_file = st.file_uploader("Race card CSV or PDF", type=["csv", "pdf"], key="card")

    st.subheader("2) Upload historical race results (optional but recommended)")
    st.markdown("Expected columns:\n" + recommended_columns_text(REQUIRED_HISTORY_COLUMNS | {"winner_running_style"}))
    history_file = st.file_uploader("Historical results CSV or PDF", type=["csv", "pdf"], key="history")

    if not card_file:
        st.info("Upload a race card CSV/PDF to begin.")
        return

    cards_raw = _load_uploaded_table(card_file)
    if cards_raw.empty:
        st.error("Could not parse race card. For PDF uploads, make sure the table is text-based and delimiter-separated.")
        return

    missing_card_cols = validate_card_columns(cards_raw)
    if missing_card_cols:
        st.error(f"Missing required columns in race card: {', '.join(missing_card_cols)}")
        return

    history_df = None
    if history_file:
        history_df = _load_uploaded_table(history_file)
        if history_df.empty:
            st.error(
                "Could not parse historical results file. For PDF uploads, make sure the table is text-based and delimiter-separated."
            )
            return

        missing_history_cols = validate_history_columns(history_df)
        if missing_history_cols:
            st.error(f"Missing required columns in historical results: {', '.join(missing_history_cols)}")
            return

        update_track_cache(history_df)

    track_specific_history = history_for_tracks(cards_raw, history_df)
    scored = score_races(cards_raw, track_specific_history, weights)

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

    if not track_specific_history.empty:
        st.subheader("Trend analysis from track-specific historical cache + uploads")
        trends = summarize_trends(track_specific_history)

        tab1, tab2, tab3 = st.tabs(["Surface Bias", "Distance Buckets", "Winning Running Styles"])
        with tab1:
            st.dataframe(trends["surface_bias"], use_container_width=True, hide_index=True)
        with tab2:
            st.dataframe(trends["distance_buckets"], use_container_width=True, hide_index=True)
        with tab3:
            st.dataframe(trends["pace_shape"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
