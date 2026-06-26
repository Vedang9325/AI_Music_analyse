import os
import librosa
import numpy as np
import pandas as pd

# Updated to match your exact VS Code folder structure
DATASET_PATH = "dataset/genres_original"
OUTPUT_CSV = "features.csv"

def extract_features():
    data = []
    
    for genre in os.listdir(DATASET_PATH):
        genre_folder = os.path.join(DATASET_PATH, genre)
        if not os.path.isdir(genre_folder):
            continue
            
        print(f"Processing: {genre}")
        for file in os.listdir(genre_folder):
            file_path = os.path.join(genre_folder, file)
            try:
                # Load 30 seconds of audio and extract 40 numerical features
                y, sr = librosa.load(file_path, duration=30)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                
                # Append the features and the target label
                data.append(list(mfcc_mean) + [genre])
            except Exception as e:
                print(f"Skipping corrupt file {file}: {e}")
                
    # Save the structured matrix to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Saved features to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_features()