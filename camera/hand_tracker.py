# camera/hand_tracker.py
# Hybrid ASL Tracking Engine:
# 1. Local Model evaluates FIRST (instant 0.05s response if confident >= 70%)
# 2. If local model is unsure (< 70%) -> Automatically falls back to Gemini 2.0 Flash!

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import pickle
import json
from dotenv import load_dotenv
from shared.mailbox import sign_queue

# Load .env for Gemini API key
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Client if key exists
gemini_client = None
if GEMINI_API_KEY and GEMINI_API_KEY.strip():
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY.strip())
        print("[HAND TRACKER] Gemini 2.0 Flash fallback initialized!")
    except Exception as e:
        print(f"[HAND TRACKER] Warning initializing Gemini client: {e}")
else:
    print("[HAND TRACKER] Note: No GEMINI_API_KEY in .env. Gemini fallback will prompt for key.")

# ============================================================
# STEP 1: Set up MediaPipe Hands
# ============================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# STEP 2: Load Local ML Model
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
        print(f"[HAND TRACKER] Could not load local model: {e}")

# ============================================================
# STEP 3: Motion Gate & Cache Variables
# ============================================================
POV_MODE = True
hand_was_visible = False
frames_without_hands = 0
DISAPPEAR_THRESHOLD = 15

collected_feature_frames = []
captured_video_frames = []  # Stores raw image frames for Gemini fallback

def extract_features(results):
    """
    Extracts 126 numbers (63 for Left Hand + 63 for Right Hand),
    normalized relative to the wrist.
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
# STEP 4: Gemini 2.0 Flash Fallback Function
# ============================================================
def ask_gemini_fallback(raw_frames):
    """
    Sends 3 sampled visual frames of the sign to Gemini 2.0 Flash.
    Gemini understands dynamic motion and handles any ASL sign!
    """
    global gemini_client
    if not gemini_client:
        return "Uncertain (Set GEMINI_API_KEY in .env for Cloud AI)"

    if len(raw_frames) < 3:
        return "Sign Too Brief"

    print("\n[GEMINI FALLBACK] Local model unsure. Asking Gemini 2.0 Flash...")
    try:
        from google.genai import types

        # Pick 3 key frames: beginning, middle, and end of the gesture
        indices = [
            0,
            len(raw_frames) // 2,
            len(raw_frames) - 1
        ]
        
        image_parts = []
        for idx in indices:
            frame = raw_frames[idx]
            # Resize slightly for fast upload
            small_frame = cv2.resize(frame, (640, 360))
            _, buffer = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_parts.append(
                types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg")
            )

        prompt_text = (
            "You are an expert ASL (American Sign Language) interpreter. "
            "These 3 consecutive frames show a 1st-person POV (chest view) of a user performing an ASL sign. "
            "Identify the sign. Reply ONLY with the English word or phrase in 1 to 3 words. No extra sentences."
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt_text] + image_parts
        )
        
        result_text = response.text.strip().replace('"', '').replace('.', '').upper()
        print(f"[GEMINI FALLBACK] Gemini Translated: '{result_text}'")
        return result_text

    except Exception as e:
        print(f"[GEMINI FALLBACK ERROR]: {e}")
        return "Sign Detected (Gemini Error)"

# ============================================================
# STEP 5: Hybrid Translation Coordinator
# ============================================================
def translate_sign(feature_frames, raw_frames):
    """
    HYBRID CASCADE:
    1. Check Local Model first (0.05s).
    2. If confidence >= 70% -> USE LOCAL MODEL.
    3. If confidence < 70%  -> FALLBACK TO GEMINI 2.0 FLASH!
    """
    num_frames = len(feature_frames)
    print(f"\n[MOTION GATE] Motion ended! Analyzing {num_frames} frames...")
    
    translation = None
    confidence = 0.0
    source = "LOCAL"

    # --- 1. LOCAL MODEL ATTEMPT ---
    if local_model is not None and num_frames > 0:
        start_idx = max(0, int(num_frames * 0.25))
        end_idx = min(num_frames, int(num_frames * 0.75) + 1)
        sub_frames = feature_frames[start_idx:end_idx] if end_idx > start_idx else feature_frames
        avg_features = np.mean(sub_frames, axis=0).reshape(1, -1)
        
        try:
            probs = local_model.predict_proba(avg_features)[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            
            # High confidence threshold for instant local output
            if confidence >= 0.70:
                translation = local_model.classes_[best_idx]
                source = f"LOCAL ({confidence * 100:.0f}%)"
                print(f"[LOCAL MODEL - INSTANT] Match found: '{translation}' ({confidence * 100:.1f}%)")
            else:
                print(f"[LOCAL MODEL] Low confidence ({confidence * 100:.1f}%). Triggering Gemini...")
        except Exception as e:
            print(f"[LOCAL MODEL ERROR]: {e}")

    # --- 2. GEMINI FALLBACK ATTEMPT ---
    if translation is None:
        source = "GEMINI 2.0"
        translation = ask_gemini_fallback(raw_frames)

    # Push to mailbox for the screen
    sign_queue.put({
        "translation": translation,
        "source": source,
        "confidence": round(confidence, 2),
        "frames_captured": num_frames,
        "timestamp": time.time()
    })
    
    return translation, source

# ============================================================
# STEP 6: Main Camera Loop
# ============================================================
def start_hand_tracker():
    global hand_was_visible, frames_without_hands
    global collected_feature_frames, captured_video_frames, POV_MODE
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("========================================")
    print("   VOICELENS HYBRID ASL TRACKER         ")
    print("========================================")
    print("1. Local Model runs in 0.05s if confident.")
    print("2. Gemini 2.0 Flash handles everything else.")
    print("Controls: 'p' = Toggle POV, 'q' = Quit")
    print("========================================")
    
    last_detected_sign = ""
    last_source = ""
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        if not POV_MODE:
            frame = cv2.flip(frame, 1)
            
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Guide Zone
        zone_x1, zone_y1 = int(w * 0.10), int(h * 0.20)
        zone_x2, zone_y2 = int(w * 0.90), int(h * 0.95)
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (80, 80, 80), 1)
        cv2.putText(frame, "POV SIGNING ZONE", (zone_x1 + 10, zone_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # Hands Visible
        if results.multi_hand_landmarks:
            hand_was_visible = True
            frames_without_hands = 0
            
            num_hands = len(results.multi_hand_landmarks)
            
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
            
            feats = extract_features(results)
            if feats is not None:
                collected_feature_frames.append(feats)
                # Keep copy of clean frame for Gemini fallback
                captured_video_frames.append(frame.copy())
            
            # HUD
            hands_color = (0, 255, 0) if num_hands == 2 else (0, 200, 255)
            cv2.putText(frame, f"Hands: {num_hands}/2 | Motion: {len(collected_feature_frames)} frames",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hands_color, 2)
            
        # Hands Not Visible
        else:
            if hand_was_visible:
                frames_without_hands += 1
                remaining = DISAPPEAR_THRESHOLD - frames_without_hands
                
                if remaining > 0:
                    cv2.putText(frame, f"Hands dropped! Processing in {remaining}...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                if frames_without_hands >= DISAPPEAR_THRESHOLD:
                    if len(collected_feature_frames) > 5:
                        last_detected_sign, last_source = translate_sign(
                            collected_feature_frames, captured_video_frames
                        )
                    
                    collected_feature_frames = []
                    captured_video_frames = []
                    hand_was_visible = False
                    frames_without_hands = 0
            else:
                cv2.putText(frame, "Waiting for hands... (POV View)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 130, 130), 2)
        
        # Display Recognized Result Banner
        if last_detected_sign:
            cv2.putText(frame, f"SIGN: {last_detected_sign} [{last_source}]", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Bottom Bar
        mode_str = "POV Mode (Direct)" if POV_MODE else "Mirror Mode"
        cv2.putText(frame, f"Mode: {mode_str} | 'p' = Toggle | 'q' = Quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("VoiceLens - Hybrid Hand Tracker", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            POV_MODE = not POV_MODE
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_hand_tracker()
