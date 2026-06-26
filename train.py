import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configuration
CSV_FILE = "features.csv"
REGISTRY_PATH = "model_registry"
ROOT_MODEL = "music_genre_model.pkl"

# Ensure registry directory exists
if not os.path.exists(REGISTRY_PATH):
    os.makedirs(REGISTRY_PATH)

def train_and_register():
    # 1. Load Data
    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training on {X_train.shape[0]} samples, Testing on {X_test.shape[0]} samples")

    # 3. Train Model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, y_pred))

    # 5. MLOps: Registry Versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"model_{timestamp}.pkl"
    meta_filename = f"metadata_{timestamp}.json"

    # Save to Registry
    joblib.dump(model, os.path.join(REGISTRY_PATH, model_filename))
    
    # Save Metadata
    metadata = {
        "timestamp": timestamp,
        "accuracy": float(acc),
        "model_file": model_filename,
        "features_count": X.shape[1]
    }
    with open(os.path.join(REGISTRY_PATH, meta_filename), "w") as f:
        json.dump(metadata, f, indent=4)

    # Keep Root Link for Frontend
    joblib.dump(model, ROOT_MODEL)
    
    print(f"\nSuccess! Versioned model registered as {model_filename}")
    print(f"Metadata recorded in {meta_filename}")

if __name__ == "__main__":
    train_and_register()