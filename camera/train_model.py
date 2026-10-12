# camera/train_model.py
# Trains the local lightweight ML model for instant (0.05s) ASL translation.
# Uses scikit-learn RandomForestClassifier on 126 hand coordinate features.

import json
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "data", "asl_dataset.json")
MODEL_PATH = os.path.join(BASE_DIR, "asl_model.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "label_map.json")

def train():
    print("========================================")
    print("       VOICELENS MODEL TRAINER          ")
    print("========================================")

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at '{DATASET_PATH}'.")
        print("Run 'python -m camera.collect_data' first to record samples!")
        return

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    X = []
    y = []
    labels = list(dataset.keys())

    print(f"Found {len(labels)} classes in dataset:")
    for label in labels:
        samples = dataset[label]
        print(f"  - {label}: {len(samples)} samples")
        for sample in samples:
            X.append(sample)
            y.append(label)

    if len(X) < 10:
        print("\nNot enough data! Please record at least 20-30 samples per sign.")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"\nTotal training samples: {len(X)}")
    print("Training RandomForest model...")

    # Split for validation if we have enough samples
    if len(X) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nModel Validation Accuracy: {acc * 100:.2f}%")

    # Save model and label mappings
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"[OK] Model saved to: {MODEL_PATH}")
    print(f"[OK] Labels saved to: {LABELS_PATH}")
    print("========================================")
    print("Your local model is ready! hand_tracker.py will now use it automatically.")

if __name__ == "__main__":
    train()
