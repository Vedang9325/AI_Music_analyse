import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "music_genre_model.pkl"

# 1. Page Layout Configuration
st.set_page_config(
    page_title="SonicVibe Studio Pro", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MATERIAL DESIGN CSS INJECTION ---
material_css = """
<style>
    /* Import Google's Roboto Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif !important;
    }
    
    /* Material Elevated Cards for Metrics - Updated for Dark Mode Support */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--secondary-background-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.15);
    }
    
    /* Material Primary Buttons */
    .stButton>button {
        border-radius: 24px;
        background-color: #6200EE; /* Material Design Deep Purple */
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 3px 5px -1px rgba(0,0,0,0.2), 0 6px 10px 0 rgba(0,0,0,0.14), 0 1px 18px 0 rgba(0,0,0,0.12);
        transition: background-color 0.3s, box-shadow 0.3s;
    }
    .stButton>button:hover {
        background-color: #3700B3;
        color: white;
        box-shadow: 0 5px 5px -3px rgba(0,0,0,0.2), 0 8px 10px 1px rgba(0,0,0,0.14), 0 3px 14px 2px rgba(0,0,0,0.12);
    }
    
    /* Upload Box Styling - Updated for Dark Mode Support */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #6200EE;
        border-radius: 16px;
        background-color: var(--secondary-background-color);
    }
    
    /* Sidebar Material Surface - Removed hardcoded light background */
    [data-testid="stSidebar"] {
        box-shadow: 2px 0 8px rgba(0,0,0,0.15);
    }
    
    /* Dataframe rounded corners */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
"""
st.markdown(material_css, unsafe_allow_html=True)
# ---------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# App Sub-header Architecture
st.title("🔮 SonicVibe Studio")
st.markdown("#### Advanced Audio Analytics Dashboard")
st.caption("Production-grade signal processing framework extracting time-series structural sequences.")
st.markdown("---")

# 2. Sidebar Integration Panel
with st.sidebar:
    st.markdown("## ⚙️ Core Configuration")
    if MODEL_PATH.exists():
        st.success("AI Model Engine: ONLINE")
    else:
        st.error("AI Model Engine: OFFLINE")
        st.warning("Please execute train.py first.")
    
    st.markdown("---")
    st.markdown("### 🛠️ Extraction Specifications")
    slice_duration = st.slider("Time-Series Window Slice (seconds)", min_value=1, max_value=5, value=3)
    st.info(f"The audio wave will be segmented into sequential {slice_duration}-second intervals.")

# Helper: Extract static perceptual parameters
def extract_perceptual_metrics(y, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    except:
        bpm = 120.0
    
    rms = np.mean(librosa.feature.rms(y=y))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    return round(bpm, 1), round(rms * 100, 2), round(flatness * 100, 2)

# 3. Main Operational Execution
if not MODEL_PATH.exists():
    st.warning("🚨 Master Model Brain missing. Run `train.py` to initialize parameters.")
else:
    model = load_model()
    
    # Batch Processing Interface File Box
    uploaded_files = st.file_uploader(
        "Drop Audio Files Here (Parallel processing enabled)", 
        type=["wav", "mp3"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown("### ⚡ Processing Workspace")
        
        batch_summary_data = []
        file_analytics_map = {}
        progress_bar = st.progress(0)
        
        for index, file in enumerate(uploaded_files):
            with st.spinner(f"Processing structural audio arrays for: {file.name}..."):
                try:
                    y, sr = librosa.load(file, duration=30)
                    bpm, energy, noise_floor = extract_perceptual_metrics(y, sr)
                    
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
                    primary_prediction = model.predict(mfcc_mean)[0].upper()
                    
                    batch_summary_data.append({
                        "File Name": file.name,
                        "Predicted Genre": primary_prediction,
                        "Tempo (BPM)": bpm,
                        "Energy Index (%)": energy,
                        "Flatness Index (%)": noise_floor
                    })

                    total_duration = librosa.get_duration(y=y, sr=sr)
                    intervals = int(total_duration // slice_duration)
                    timeline_records = []

                    for i in range(intervals):
                        start_sample = int(i * slice_duration * sr)
                        end_sample = int((i + 1) * slice_duration * sr)
                        chunk = y[start_sample:end_sample]
                        
                        if len(chunk) > 0:
                            chunk_mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40)
                            chunk_mean = np.mean(chunk_mfcc.T, axis=0).reshape(1, -1)
                            probs = model.predict_proba(chunk_mean)[0]
                            pred_genre = model.classes_[np.argmax(probs)]
                            
                            timeline_records.append({
                                "Timestamp (s)": f"{i * slice_duration}s - {(i + 1) * slice_duration}s",
                                "Predicted Vibe": pred_genre.capitalize(),
                                "Confidence (%)": round(float(np.max(probs)) * 100, 1)
                            })
                    
                    file_analytics_map[file.name] = {
                        "timeline": pd.DataFrame(timeline_records),
                        "audio_buffer": file
                    }
                    
                except Exception as e:
                    st.error(f"Failed to analyze track sequence {file.name}: {e}")
            
            progress_bar.progress((index + 1) / len(uploaded_files))
        
        progress_bar.empty()

        df_summary = pd.DataFrame(batch_summary_data)
        st.dataframe(df_summary, use_container_width=True)

        csv_buffer = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="DOWNLOAD ANALYTICS REPORT",
            data=csv_buffer,
            file_name="sonic_vibe_batch_report.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        st.markdown("### 🔍 Deep-Dive Timeline Analysis")
        selected_track = st.selectbox("Select a processed track to map its temporal progression:", df_summary["File Name"].tolist())
        
        if selected_track:
            track_data = file_analytics_map[selected_track]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Track Deck")
                st.audio(track_data["audio_buffer"], format="audio/wav")
                
                matching_row = df_summary[df_summary["File Name"] == selected_track].iloc[0]
                
                # These metrics will now render as elevated Material cards due to the CSS
                st.metric("Acoustic Tempo", f"{matching_row['Tempo (BPM)']} BPM")
                st.metric("RMS Energy Density", f"{matching_row['Energy Index (%)']}%")
                st.metric("Spectral Noise Flatness", f"{matching_row['Flatness Index (%)']}%")
                
            with col2:
                st.markdown("#### Genre Shifts Over Time")
                timeline_df = track_data["timeline"]
                
                fig = px.line(
                    timeline_df, 
                    x="Timestamp (s)", 
                    y="Confidence (%)", 
                    color="Predicted Vibe",
                    markers=True,
                    labels={"Confidence (%)": "AI Confidence", "Timestamp (s)": "Timeline Segment"}
                )
                fig.update_layout(
                    yaxis_range=[0, 105],
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=10, l=10, r=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
                
    else:
        st.info("💡 Drop MP3/WAV tracks into the drop-zone above to initialize processing.")
