import sys
import os
import librosa
import numpy as np
import joblib

MODEL_PATH = "music_genre_model.pkl"

def predict_genre(audio_file_path):
    # 1. Check if the file actually exists
    if not os.path.exists(audio_file_path):
        print(f"Error: The file '{audio_file_path}' was not found.")
        return

    # 2. Check if the trained model brain exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model '{MODEL_PATH}' not found. Please run train.py first.")
        return

    print(f"Analyzing audio: {os.path.basename(audio_file_path)}...")

    try:
        # 3. Load the model brain
        model = joblib.load(MODEL_PATH)

        # 4. Extract audio features (must match the exact format used during training)
        y, sr = librosa.load(audio_file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # 5. Make the prediction
        prediction = model.predict(mfcc_mean)[0]
        probabilities = model.predict_proba(mfcc_mean)[0]
        classes = model.classes_

        # 6. Display results
        print("\n=== AI PREDICTION RESULT ===")
        print(f"Predicted Genre: ** {prediction.upper()} **")
        print("============================\n")
        
        print("Confidence Breakdown:")
        for cls, prob in zip(classes, probabilities):
            if prob > 0.01: # Only print genres with more than 1% confidence
                print(f" - {cls.capitalize()}: {prob * 100:.1f}%")

    except Exception as e:
        print(f"An error occurred while processing the audio: {e}")

if __name__ == "__main__":
    # You can change this path to any WAV or MP3 track you want to test!
    sample_file = "dataset/genres_original/pop/pop.00000.wav" 
    predict_genre(sample_file)