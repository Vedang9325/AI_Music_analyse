# SonicVibe Studio

SonicVibe Studio is a finished Streamlit application for music genre prediction and timeline-style audio analysis. Upload one or more WAV/MP3 files and the app predicts the dominant genre, estimates acoustic metrics, and shows how the predicted vibe changes across short time windows.

The app is deployed on Streamlit Community Cloud and the repository includes the trained `RandomForestClassifier` model, feature extraction pipeline, command-line predictor, and optional Docker/process-based deployment files.

## Live App

The project is live on Streamlit Community Cloud. Add the public app URL here once you want it shown in the repository.

## Features

- Predicts music genre from audio using 40 MFCC features.
- Supports batch upload of WAV and MP3 files.
- Shows tempo, RMS energy, and spectral flatness for each track.
- Splits tracks into configurable 1-5 second windows for timeline analysis.
- Displays confidence trends with interactive Plotly charts.
- Exports batch results as a CSV analytics report.
- Includes scripts for feature extraction, model training, and single-file prediction.

## Tech Stack

- Python 3.12
- Streamlit
- Librosa
- scikit-learn
- Plotly
- Pandas and NumPy
- Joblib

## Project Structure

```text
.
|-- app.py                   # Streamlit dashboard and batch audio workflow
|-- extract.py               # Extracts MFCC features from the dataset
|-- train.py                 # Trains and saves the genre classifier
|-- predict.py               # Command-line prediction helper
|-- features.csv             # Extracted training features
|-- music_genre_model.pkl    # Trained model used by the app
|-- requirements.txt         # Python dependencies
|-- packages.txt             # Linux audio packages for Streamlit Cloud
|-- runtime.txt              # Python runtime for Streamlit Cloud
|-- Procfile                 # Process command for compatible hosts
|-- Dockerfile               # Container deployment setup
`-- dataset/                 # Local training dataset
```

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Using the App

1. Start the Streamlit app.
2. Upload one or more `.wav` or `.mp3` files.
3. Choose the timeline window size from the sidebar.
4. Review the batch summary table.
5. Select a track for deeper timeline analysis.
6. Download the generated CSV report if needed.

The app uses `music_genre_model.pkl` from the repository root. If that file is missing, run `python train.py` after generating or confirming `features.csv`.

## Training Workflow

The training pipeline expects audio files arranged by genre:

```text
dataset/
`-- genres_original/
    |-- blues/
    |-- classical/
    |-- country/
    `-- ...
```

Extract MFCC features:

```bash
python extract.py
```

Train and register a model:

```bash
python train.py
```

Training writes the latest app model to:

```text
music_genre_model.pkl
```

It also creates timestamped model and metadata files in:

```text
model_registry/
```

## Command-Line Prediction

To test a single file from Python:

```bash
python predict.py
```

By default, `predict.py` analyzes:

```text
dataset/genres_original/pop/pop.00000.wav
```

Edit the `sample_file` value in `predict.py` to point to another local WAV or MP3 file.

## Streamlit Community Cloud

This project has already been deployed to Streamlit Community Cloud. The deployed app uses `app.py` as the main file and requires these files in the repository root:

- `app.py`
- `music_genre_model.pkl`
- `requirements.txt`
- `packages.txt`
- `runtime.txt`

`packages.txt` installs `ffmpeg` and `libsndfile1`, which help Librosa and SoundFile decode uploaded audio reliably on Linux hosts.

## Docker Deployment

Build the image:

```bash
docker build -t sonicvibe-studio .
```

Run the container:

```bash
docker run --rm -p 8501:8501 sonicvibe-studio
```

Then open:

```text
http://localhost:8501
```

The Docker image copies only the production app files and trained model. The full `dataset/` directory is intentionally excluded from Docker builds.

## Notes

- The bundled model is trained from the generated `features.csv`.
- Audio is loaded for up to 30 seconds per file during analysis.
- Model predictions are best suited for genres represented in the training dataset.
- Large datasets and generated registry files should usually stay out of production deployments unless retraining is required.
