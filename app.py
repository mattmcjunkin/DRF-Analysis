from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from analyzer import (
    CACHE_ROOT,
    REQUIRED_COLUMNS,
    REQUIRED_HISTORY_COLUMNS,
    ModelWeights,
    history_for_tracks,
    load_brisnet_file,
    recommended_columns_text,
    score_races,
    summarize_trends,
    update_track_cache,
    validate_card_columns,
    validate_history_columns,
)

US_TRACKS: Dict[str, str] = {
    "AQU": "Aqueduct",
    "BEL": "Belmont at the Big A",
    "SAR": "Saratoga",
    "BAQ": "Belmont at Aqueduct",
    "CD": "Churchill Downs",
    "KEE": "Keeneland",
    "ELP": "Ellis Park",
    "FG": "Fair Grounds",
    "OP": "Oaklawn Park",
    "SA": "Santa Anita",
    "DMR": "Del Mar",
    "LRC": "Los Alamitos",
    "GG": "Golden Gate",
    "PLN": "Pleasanton",
    "TUP": "Turf Paradise",
    "RUI": "Ruidoso Downs",
    "SUN": "Sunland Park",
    "ALB": "Albuquerque",
    "EMD": "Emerald Downs",
    "MTH": "Monmouth Park",
    "MED": "Meadowlands",
    "PRX": "Parx Racing",
    "PEN": "Penn National",
    "PID": "Presque Isle Downs",
    "DEL": "Delaware Park",
    "LRL": "Laurel Park",
    "PIM": "Pimlico",
    "CNL": "Colonial Downs",
    "TAM": "Tampa Bay Downs",
    "GP": "Gulfstream Park",
    "GPW": "Gulfstream Park West",
    "HIA": "Hialeah",
    "WO": "Woodbine",
    "FE": "Fort Erie",
    "HST": "Hastings",
    "AP": "Arlington Park",
    "HOO": "Horseshoe Indianapolis",
    "IND": "Indiana Grand",
    "BTP": "Belterra Park",
    "MVR": "Mahoning Valley",
    "TDN": "Thistledown",
    "CT": "Charles Town",
    "MNR": "Mountaineer",
    "RP": "Remington Park",
    "WRD": "Will Rogers Downs",
    "EVD": "Evangeline Downs",
    "DED": "Delta Downs",
    "LAD": "Louisiana Downs",
    "RET": "Retama Park",
    "HOU": "Sam Houston",
    "LS": "Lone Star Park",
    "PRM": "Prairie Meadows",
    "CBY": "Canterbury Park",
    "FON": "Fonner Park",
    "ASD": "Assiniboia Downs",
}


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


def _load_uploaded_table(uploaded_file, history: bool = False) -> pd.DataFrame:
    suffix = uploaded_file.name.lower().split(".")[-1]
    if suffix == "csv":
        return pd.read_csv(uploaded_file)
    if suffix in {"drf", "dr2", "dr3", "dr4"}:
        return load_brisnet_file(uploaded_file.read(), extension=suffix, history=history)
    return pd.DataFrame()


def _selected_track_from_sidebar() -> str:
    current_qp = st.query_params
    current_track = str(current_qp.get("track", "ALL")).upper()

    options = ["ALL"] + sorted(US_TRACKS.keys())
    default_index = options.index(current_track) if current_track in options else 0

    with st.sidebar:
        st.header("Track Analyzer")
        selected_track = st.selectbox(
            "Choose US track analyzer",
            options,
            index=default_index,
            format_func=lambda code: "All tracks" if code == "ALL" else f"{code} — {US_TRACKS.get(code, code)}",
        )

        if selected_track == "ALL":
            st.caption("Analyzer link: /?track=ALL")
        else:
            st.markdown(f"[Open analyzer for {selected_track}](?track={selected_track})")

    st.query_params["track"] = selected_track
    return selected_track


def main() -> None:
    st.set_page_config(page_title="Brisnet Race Analyzer", layout="wide")
    st.title("🏇 Brisnet Race Card Analyzer")
    st.caption(
        "Upload Brisnet data files plus historical results to generate winner predictions and detect track/race-shape trends."
    )

    st.info(
        "Supported Brisnet file types: .DRF, .DR2, .DR3, .DR4. CSV uploads are also accepted for normalized/manual datasets."
    )

    with st.sidebar:
        st.caption(f"Per-track cache directory: `{CACHE_ROOT}`")

    selected_track = _selected_track_from_sidebar()
    weights = _build_weights()

    st.subheader("1) Upload current race card")
    st.caption("Brisnet card files are usually .DRF")
    st.markdown("Expected columns:\n" + recommended_columns_text(REQUIRED_COLUMNS))
    card_file = st.file_uploader("Race card file", type=["csv", "drf", "dr2", "dr3", "dr4"], key="card")

    st.subheader("2) Upload historical race results (optional but recommended)")
    st.caption("Brisnet history files can be .DR2/.DR3/.DR4 depending on export type")
    st.markdown("Expected columns:\n" + recommended_columns_text(REQUIRED_HISTORY_COLUMNS | {"winner_running_style"}))
    history_file = st.file_uploader(
        "Historical results file", type=["csv", "drf", "dr2", "dr3", "dr4"], key="history"
    )

    if not card_file:
        st.info("Upload a Brisnet (.DRF/.DR2/.DR3/.DR4) or CSV race card to begin.")
        return

    cards_raw = _load_uploaded_table(card_file, history=False)
    if cards_raw.empty:
        st.error("Could not parse race card. Verify the .DRF/.DR2/.DR3/.DR4 file is text-based and delimiter/fixed-width formatted.")
        return

    missing_card_cols = validate_card_columns(cards_raw)
    if missing_card_cols:
        st.error(f"Missing required columns in race card: {', '.join(missing_card_cols)}")
        return

    history_df = None
    if history_file:
        history_df = _load_uploaded_table(history_file, history=True)
        if history_df.empty:
            st.error("Could not parse historical file. Verify the Brisnet file format and that it contains result/history rows.")
            return

        missing_history_cols = validate_history_columns(history_df)
        if missing_history_cols:
            st.error(f"Missing required columns in historical results: {', '.join(missing_history_cols)}")
            return

        update_track_cache(history_df)

    track_specific_history = history_for_tracks(cards_raw, history_df)
    scored = score_races(cards_raw, track_specific_history, weights)

    if selected_track != "ALL":
        scored = scored[scored["track"] == selected_track]
        track_specific_history = track_specific_history[track_specific_history["track"] == selected_track]

    if scored.empty:
        st.warning("No scored races available for the selected track filter.")
        return

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
