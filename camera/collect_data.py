# camera/collect_data.py
# Tool to collect training samples for HIGH-FREQUENCY STILL / STATIC ASL signs in 1st-Person POV.
# Records separate batches of 60 frames for Left Hand and 60 frames for Right Hand (up to 240 total).

import os
import sys

# Silence TensorFlow / MediaPipe background terminal logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings("ignore")

import json
import time
import numpy as np

# HIGH-FREQUENCY STILL / STATIC SIGNS (Rock-solid for MediaPipe posture tracking)
SIGNS_LIST = [
    "I LOVE YOU",
    "STOP",
    "OK",
    "YES",
    "NO",
    "PEACE",
    "GOOD",
    "BAD",
    "YOU",
    "ME / I",
    "CALL ME",
    "TIME"
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DATASET_PATH = os.path.join(DATA_DIR, "asl_dataset.json")

def extract_features(results):
    """
    Extracts 126 numbers (63 for Left Hand + 63 for Right Hand),
    normalized relative to the wrist so distance from camera doesn't distort it.
    Returns: (feature_list, active_hand_name)
    """
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)
    active_hand = "None"
    
    if not results.multi_hand_landmarks:
        return None, "None"
        
    for idx, hand_lms in enumerate(results.multi_hand_landmarks):
        label = "Right"
        if results.multi_handedness and idx < len(results.multi_handedness):
            label = results.multi_handedness[idx].classification[0].label
            
        wrist = hand_lms.landmark[0]
        coords = []
        for lm in hand_lms.landmark:
            coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
        coords = np.array(coords)
        if label == "Left":
            left_hand = coords
            active_hand = "Left" if active_hand == "None" else "Both"
        else:
            right_hand = coords
            active_hand = "Right" if active_hand == "None" else "Both"
            
    features = np.concatenate([left_hand, right_hand]).tolist()
    return features, active_hand

def main():
    print("========================================")
    print("    VOICELENS STILL ASL COLLECTOR       ")
    print("========================================")
    print("Choose a static sign to record:\n")
    for idx, sign in enumerate(SIGNS_LIST, 1):
        print(f"  {idx:2d}. {sign}")
    print("========================================")

    selected_sign = None
    while not selected_sign:
        try:
            user_input = input(f"Enter number (1-{len(SIGNS_LIST)}) or sign name: ").strip()
            if not user_input:
                continue
            if user_input.isdigit():
                choice = int(user_input)
                if 1 <= choice <= len(SIGNS_LIST):
                    selected_sign = SIGNS_LIST[choice - 1]
                else:
                    print(f"Please enter a number between 1 and {len(SIGNS_LIST)}.")
            else:
                match = [s for s in SIGNS_LIST if s.lower() == user_input.lower()]
                if match:
                    selected_sign = match[0]
                else:
                    print("Sign not found. Please try again.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    print("\nInitializing camera and hand tracking...")
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    dataset = {}
    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, "r") as f:
                dataset = json.load(f)
        except Exception:
            dataset = {}
            
    if selected_sign not in dataset:
        dataset[selected_sign] = []

    print(f"\n[TARGET SIGN]: '{selected_sign}'")
    print(f"Existing total saved: {len(dataset[selected_sign])}")
    print("\nHow to collect 240 samples (60 Left, 60 Right, etc.):")
    print("  1. Hold your RIGHT hand steady in position and press 'r' (captures 60 frames).")
    print("  2. Hold your LEFT hand steady in position and press 'r' (captures 60 frames).")
    print("  3. Repeat if desired to reach up to 240 samples.")
    print("  4. Press 'q' to save and exit.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording = False
    samples_this_burst = 0
    BURST_SIZE = 60

    # Count how many left/right in dataset
    left_count = 0
    right_count = 0
    for sample in dataset[selected_sign]:
        has_left = any(sample[:63])
        has_right = any(sample[63:])
        if has_left: left_count += 1
        if has_right: right_count += 1

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_active = "No Hand"
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        if recording:
            features, active_hand = extract_features(results)
            if features is not None:
                dataset[selected_sign].append(features)
                samples_this_burst += 1
                if active_hand == "Left":
                    left_count += 1
                elif active_hand == "Right":
                    right_count += 1
                elif active_hand == "Both":
                    left_count += 1
                    right_count += 1

            cv2.putText(frame, f"RECORDING '{selected_sign}': {samples_this_burst}/{BURST_SIZE}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            if samples_this_burst >= BURST_SIZE:
                recording = False
                samples_this_burst = 0
                print(f"\n[BURST COMPLETE] Saved {BURST_SIZE} samples! (Total: {len(dataset[selected_sign])})")
        else:
            # HUD details
            total_samples = len(dataset[selected_sign])
            cv2.putText(frame, f"SIGN: {selected_sign}",
                        (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Total: {total_samples}/240  |  Left: {left_count}  |  Right: {right_count}",
                        (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'r' to record burst of 60 | 'q' to Save & Exit",
                        (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("VoiceLens - Still ASL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording:
            recording = True
            samples_this_burst = 0
            print(f"\nRecording burst of {BURST_SIZE} for '{selected_sign}'...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(DATASET_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved successfully to: {DATASET_PATH}")
    print(f"Total samples for '{selected_sign}': {len(dataset[selected_sign])} (Left: {left_count}, Right: {right_count})")

if __name__ == "__main__":
    main()
