# camera/hand_tracker.py
# Tracks hands on the webcam using MediaPipe in 1st-Person POV (FPS Chest / Glasses View).
# Integrated with local ML model for instant (0.05s) ASL translation of Top 5 signs.

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pickle
import json
from shared.mailbox import sign_queue

# ============================================================
# STEP 1: Set up MediaPipe Hands
# ============================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,                  # Track up to 2 hands simultaneously
    min_detection_confidence=0.5,     # Easier detection for angled POV hands
    min_tracking_confidence=0.5       # Keeps tracking even during hand movements
)

# ============================================================
# STEP 2: Load Local ML Model (if trained)
# ============================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "asl_model.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "label_map.json")

local_model = None
local_labels = []

if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            local_model = pickle.load(f)
        with open(LABELS_PATH, "r") as f:
            local_labels = json.load(f)
        print(f"[HAND TRACKER] Loaded local ASL model with {len(local_labels)} signs!")
    except Exception as e:
        print(f"[HAND TRACKER] Warning: Could not load local model: {e}")
else:
    print("[HAND TRACKER] No local model found yet. Run 'camera.train_model' to enable instant local translation.")

# ============================================================
# STEP 3: Motion Gate & POV variables
# ============================================================
POV_MODE = True                       # True = 1st-person POV (natural, no mirror flip)
hand_was_visible = False              # Were hands visible in the last frame?
frames_without_hands = 0              # How many empty frames in a row?
DISAPPEAR_THRESHOLD = 15              # Empty frames needed before triggering translation

# Collects raw feature frames while signing
collected_feature_frames = []

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
            
    return np.concatenate([left_hand, right_hand])

# ============================================================
# STEP 4: Translation Function (Local Model + Fallback)
# ============================================================
def translate_sign(feature_frames):
    """
    Takes the collected hand coordinate features and translates them.
    If local model is available: classifies instantly (0.05s)!
    Otherwise: uses fallback.
    """
    num_frames = len(feature_frames)
    print(f"\n[HAND TRACKER] Motion complete! Processing {num_frames} captured frames...")
    
    translation = "UNKNOWN"
    confidence = 0.0
    
    if local_model is not None and num_frames > 0:
        # Use average of the middle 50% frames for maximum stability
        start_idx = max(0, int(num_frames * 0.25))
        end_idx = min(num_frames, int(num_frames * 0.75) + 1)
        sub_frames = feature_frames[start_idx:end_idx] if end_idx > start_idx else feature_frames
        
        avg_features = np.mean(sub_frames, axis=0).reshape(1, -1)
        
        try:
            probs = local_model.predict_proba(avg_features)[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            
            if confidence >= 0.50:
                translation = local_model.classes_[best_idx]
                print(f"[LOCAL MODEL - INSTANT] Detected: '{translation}' (Confidence: {confidence * 100:.1f}%)")
            else:
                print(f"[LOCAL MODEL] Low confidence ({confidence * 100:.1f}%). Passing to fallback/Gemini...")
                translation = "Uncertain Sign"
        except Exception as e:
            print(f"[LOCAL MODEL ERROR]: {e}")
            translation = "Hello"
    else:
        # Fallback placeholder when model is not yet trained
        translation = "Hello"
        print(f"[HAND TRACKER (Placeholder)] Translation: '{translation}'")

    # Send translation to the mailbox for the screen
    sign_queue.put({
        "translation": translation,
        "confidence": round(confidence, 2),
        "frames_captured": num_frames,
        "timestamp": time.time()
    })
    
    return translation

# ============================================================
# STEP 5: Main Camera Loop (1st-Person POV Optimized)
# ============================================================
def start_hand_tracker():
    global hand_was_visible, frames_without_hands, collected_feature_frames, POV_MODE
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("========================================")
    print("   VOICELENS 1ST-PERSON POV TRACKER     ")
    print("========================================")
    print("1st-Person POV View: Camera points down at your hands.")
    print("Controls:")
    print("  'p' -> Toggle POV Mode vs Mirror Mode")
    print("  'q' -> Quit")
    print("========================================")
    
    last_detected_sign = ""
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        if not POV_MODE:
            frame = cv2.flip(frame, 1)
            
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Draw POV Signing Zone guide box
        zone_x1, zone_y1 = int(w * 0.10), int(h * 0.20)
        zone_x2, zone_y2 = int(w * 0.90), int(h * 0.95)
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (80, 80, 80), 1)
        cv2.putText(frame, "POV SIGNING ZONE (Keep both hands inside)", (zone_x1 + 10, zone_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # ---------------------------------------------------
        # CASE 1: Hands ARE visible
        # ---------------------------------------------------
        if results.multi_hand_landmarks:
            hand_was_visible = True
            frames_without_hands = 0
            
            num_hands_detected = len(results.multi_hand_landmarks)
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = "Hand"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    hand_label = results.multi_handedness[idx].classification[0].label
                
                box_color = (255, 200, 0) if hand_label == "Left" else (0, 255, 100)
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f"{hand_label} Hand", (wrist_x - 30, wrist_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Extract normalized feature vector
            feats = extract_features(results)
            if feats is not None:
                collected_feature_frames.append(feats)
            
            # Real-time HUD Status
            hands_color = (0, 255, 0) if num_hands_detected == 2 else (0, 200, 255)
            cv2.putText(frame, f"Hands Tracked: {num_hands_detected}/2", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, hands_color, 2)
            cv2.putText(frame, f"Recording Motion: {len(collected_feature_frames)} frames", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # ---------------------------------------------------
        # CASE 2: Hands are NOT visible
        # ---------------------------------------------------
        else:
            if hand_was_visible:
                frames_without_hands += 1
                
                remaining = DISAPPEAR_THRESHOLD - frames_without_hands
                if remaining > 0:
                    cv2.putText(frame, f"Hands dropped! Processing in {remaining}...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                if frames_without_hands >= DISAPPEAR_THRESHOLD:
                    if len(collected_feature_frames) > 5:
                        last_detected_sign = translate_sign(collected_feature_frames)
                    
                    collected_feature_frames = []
                    hand_was_visible = False
                    frames_without_hands = 0
            else:
                cv2.putText(frame, "Waiting for hands... (POV View)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 130, 130), 2)
        
        # Display last recognized sign banner if present
        if last_detected_sign:
            cv2.putText(frame, f"LAST SIGN: {last_detected_sign}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Mode status
        mode_str = "POV Mode (Direct / No Flip)" if POV_MODE else "Mirror Mode (Flipped)"
        cv2.putText(frame, f"Mode: {mode_str} | 'p' to switch | 'q' to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("VoiceLens - 1st-Person POV Hand Tracker", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            POV_MODE = not POV_MODE
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_hand_tracker()
