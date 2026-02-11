# DRF + Timeform US Race Predictor

A Streamlit app for blending DRF Formulator-style inputs with Timeform US figures to predict likely winners and inspect race/track trends.

## Features

- Upload a current race card and score each horse using:
  - Combined speed (Timeform + DRF)
  - Combined pace (Timeform + DRF)
  - Pedigree surface fit
  - Pedigree distance fit
  - Field-size pressure
  - Historical trend bias from prior race results
- **Per-track analyzer cache**: historical results are stored separately by track code in `data/track_cache/<TRACK>.csv` so trends never blend across tracks.
- Upload prior-day historical results to detect:
  - Surface bias by track
  - Distance-bucket effects
  - Dominant running styles
- Upload **CSV or PDF** files for card and historical results.
- Tune model weights directly from the sidebar.
- Export scored card predictions as CSV.
- Deploy-ready with both `Procfile` (PaaS) and `Dockerfile` (container deployments).

## Local Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## PDF Support

- PDF ingestion supports **text-based** PDFs where rows are delimiter-based (comma, pipe, tab, semicolon).
- Scanned/image-only PDFs are not directly parseable without OCR pre-processing.

## Deploying

### Option 1: PaaS (Render/Railway/Heroku-style)

This repository includes a `Procfile` that starts Streamlit on the platform-provided `PORT`:

```text
web: sh -c 'streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0'
```

### Option 2: Docker

```bash
docker build -t drf-analysis .
docker run --rm -p 8501:8501 drf-analysis
```

Open http://localhost:8501.

## Required card columns

- `date`
- `track`
- `race_number`
- `horse`
- `surface`
- `distance`
- `field_size`
- `timeform_speed`
- `drf_speed`
- `timeform_pace`
- `drf_pace`
- `pedigree_surface_fit`
- `pedigree_distance_fit`

## Required history columns

- `date`
- `track`
- `race_number`
- `surface`
- `distance`
- `field_size`
- `winner_speed`
- `winner_pace`
- `winner_running_style` (optional; defaults to `unknown` if omitted)

## Notes

This is a decision-support model, not wagering advice. The quality of predictions depends on input quality and calibration of the selected weights.
