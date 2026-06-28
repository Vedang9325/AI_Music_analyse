# 🔮 SonicVibe Studio — AI Music Genre & Audio Analytics Platform

SonicVibe Studio is a finished Streamlit application for music genre prediction and timeline-style audio analysis. Upload one or more WAV/MP3 files and the app predicts the dominant genre, estimates acoustic metrics, and shows how the predicted vibe changes across short time windows.

The app is deployed on Streamlit Community Cloud and the repository includes the trained `RandomForestClassifier` model, feature extraction pipeline, command-line predictor, and optional Docker/process-based deployment files.

## Live App

The project is live on Streamlit Community Cloud. [https://aimusicanalyse.streamlit.app/](https://sonicvibe-studio.streamlit.app/)

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
- Librosa
- scikit-learn
- RandomForestClassifier
- NumPy
- Pandas
- Joblib

## Deployment

- Streamlit Community Cloud
- Docker
- Procfile-compatible hosting

---

# 🧠 Engineering Challenges Solved

## Audio Feature Extraction Pipeline

Raw audio cannot be directly used by a traditional machine learning classifier, so the system converts songs into meaningful numerical features.

The pipeline was designed to:

- load audio files reliably
- extract 40 MFCC features per track
- normalize training data into a CSV dataset
- keep the feature structure consistent between training and prediction
- support both web-app inference and command-line prediction

---

## Timeline-Based Genre Confidence Tracking

Instead of only predicting one final genre for an entire song, SonicVibe Studio analyzes short audio windows to detect changes across the track.

The system:

- slices audio into configurable time segments
- extracts MFCC features for every segment
- predicts genre probabilities per slice
- identifies the strongest predicted vibe for each interval
- plots confidence changes over the song timeline

---

## Batch Processing Workflow

The app supports multi-file uploads and processes each track independently while keeping results organized for comparison.

Implemented workflow includes:

- batch upload handling
- per-track error isolation
- progress tracking during analysis
- summary table generation
- downloadable CSV report export

---

## Lightweight MLOps Model Registry

Training creates both the active production model and a timestamped model archive.

The registry workflow stores:

- versioned model files
- training timestamps
- model accuracy
- feature count metadata
- root-level model used by the Streamlit app

---

# 🔒 Reliability Features

- Cached model loading for faster Streamlit sessions
- Model existence checks before prediction
- Controlled 30-second audio loading for predictable processing
- Error handling for failed track analysis
- Consistent feature count between training and inference
- Deployment package support for Linux audio dependencies

---

# 📁 Project Structure

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
|-- tests/                   # Test files
`-- dataset/                 # Local training dataset
```

---

# 🚀 Future Improvements

- Deep learning-based genre classification
- Support for longer full-track analysis
- Audio waveform visualization
- Mood and emotion detection
- Playlist-level analytics
- Spotify or YouTube metadata integration
- User authentication and saved analysis history
- Improved model evaluation dashboard
- Expanded test coverage for feature extraction

---

# 🧪 Local Setup

```bash
git clone https://github.com/Vedang9325/AI_Music_analyse

cd AI_Music_analyse

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

Open the local Streamlit URL shown in the terminal:

```text
http://localhost:8501
```

---

# 🏋️ Training Workflow

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

Train and register the model:

```bash
python train.py
```

The latest production model is written to:

```text
music_genre_model.pkl
```

Timestamped model versions and metadata are stored in:

```text
model_registry/
```

---

# 🐳 Docker Deployment

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

---

# 👨‍💻 Author

```text
Vedang Satardekar
B.Tech Computer Engineering
Minor in Cybersecurity
```

---

# 📌 Project Status

```text
Actively developed and deployed.
```
