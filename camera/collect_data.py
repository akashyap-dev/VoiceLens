# camera/collect_data.py
# Tool to collect training samples for the High-Frequency ASL signs in 1st-Person POV.
# Suppresses TensorFlow/MediaPipe C++ stderr noise so the menu is clean.

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

# All 15 High-Frequency ASL Signs
SIGNS_LIST = [
    "HELLO",
    "THANK YOU",
    "YES",
    "NO",
    "HELP",
    "PLEASE",
    "MORE",
    "STOP",
    "WATER",
    "EAT / FOOD",
    "SORRY",
    "I / ME",
    "YOU",
    "WANT",
    "FRIEND"
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DATASET_PATH = os.path.join(DATA_DIR, "asl_dataset.json")

def extract_features(results):
    """
    Extracts 126 numbers (63 for Left Hand + 63 for Right Hand),
    normalized relative to the wrist so distance from camera doesn't matter!
    """
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)
    
    if not results.multi_hand_landmarks:
        return None
        
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
        else:
            right_hand = coords
            
    return np.concatenate([left_hand, right_hand]).tolist()

def main():
    print("========================================")
    print("    VOICELENS ASL DATA COLLECTOR        ")
    print("========================================")
    print("Select a sign to record training samples:\n")
    for idx, sign in enumerate(SIGNS_LIST, 1):
        print(f"  {idx:2d}. {sign}")
    print("========================================")

    # Clean input loop with retry if invalid
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
                    print("Sign not found in list. Please try again.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    # Load MediaPipe ONLY after user chooses sign (prevents stdout/stderr flood during input)
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

    # Load existing dataset if present
    dataset = {}
    if os.path.exists(DATASET_PATH):
        try:
            with open(DATASET_PATH, "r") as f:
                dataset = json.load(f)
        except Exception:
            dataset = {}
            
    if selected_sign not in dataset:
        dataset[selected_sign] = []

    print(f"\n[READY] Recording for: '{selected_sign}'")
    print(f"Previously saved samples for this sign: {len(dataset[selected_sign])}")
    print("\nHow to record:")
    print("  1. Position your hands in the camera view (POV perspective).")
    print("  2. Press 'r' to record 30 frames.")
    print("  3. Press 'q' to save and exit.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    recording = False
    samples_this_round = 0
    TARGET_SAMPLES = 30

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand skeletons
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Recording logic
        if recording:
            features = extract_features(results)
            if features is not None:
                dataset[selected_sign].append(features)
                samples_this_round += 1

            cv2.putText(frame, f"RECORDING '{selected_sign}': {samples_this_round}/{TARGET_SAMPLES}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            if samples_this_round >= TARGET_SAMPLES:
                recording = False
                samples_this_round = 0
                print(f"\n✅ Captured {TARGET_SAMPLES} samples! Total for {selected_sign}: {len(dataset[selected_sign])}")
        else:
            cv2.putText(frame, f"Sign: {selected_sign} | Total Saved: {len(dataset[selected_sign])}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'r' to Record 30 samples | 'q' to Save & Exit",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("VoiceLens - ASL Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording:
            recording = True
            samples_this_round = 0
            print(f"\nRecording {TARGET_SAMPLES} samples for '{selected_sign}'...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save to disk
    with open(DATASET_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved successfully to: {DATASET_PATH}")
    print(f"Total samples recorded for {selected_sign}: {len(dataset[selected_sign])}")

if __name__ == "__main__":
    main()
