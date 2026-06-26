# SonicVibe Studio

Streamlit app for music genre prediction and timeline-style audio analysis. It loads the bundled `music_genre_model.pkl` model and accepts WAV/MP3 uploads.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy To Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit app from the repository.
3. Set the main file path to `app.py`.
4. Keep `music_genre_model.pkl`, `requirements.txt`, `packages.txt`, and `runtime.txt` in the repository.

`packages.txt` installs `ffmpeg` and `libsndfile1` so uploaded audio files can be decoded reliably.

## Deploy With Docker

```bash
docker build -t sonicvibe-studio .
docker run --rm -p 8501:8501 sonicvibe-studio
```

Then open `http://localhost:8501`.

## Runtime Files

- `app.py`: Streamlit frontend and prediction workflow.
- `music_genre_model.pkl`: trained genre classifier required by the app.
- `requirements.txt`: Python runtime dependencies.
- `packages.txt`: Linux system packages for hosted Streamlit deployments.
- `Dockerfile`: container deployment path for Render, Fly.io, Railway, VPS, or local Docker.

The full `dataset/` directory is used for training and is intentionally excluded from Docker builds.
