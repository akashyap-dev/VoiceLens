# camera/hand_tracker.py
# This file tracks hands on the webcam using MediaPipe.
# It draws a green skeleton over the hand and detects when hands appear/disappear.
# When hands disappear, it sends the collected hand data for translation.

import cv2
import mediapipe as mp
import numpy as np
import time
from shared.mailbox import sign_queue

# ============================================================
# STEP 1: Set up MediaPipe Hands
# ============================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,                  # Track up to 2 hands
    min_detection_confidence=0.7,     # How sure it needs to be to detect a hand
    min_tracking_confidence=0.5       # How sure it needs to be to keep tracking
)

# ============================================================
# STEP 2: Motion Gate variables
# ============================================================
# These track whether hands are currently on screen or not
hand_was_visible = False              # Were hands visible in the last frame?
frames_without_hands = 0              # How many frames in a row with no hands?
DISAPPEAR_THRESHOLD = 15              # How many empty frames before we say "hands gone"

# This list collects hand positions while the person is signing
collected_frames = []

# ============================================================
# STEP 3: The translation function (placeholder for now)
# ============================================================
def translate_sign(hand_data):
    """
    Takes the collected hand coordinate data and translates it.
    RIGHT NOW: This is a placeholder that just prints the data.
    LATER: We will connect this to either a local model or Gemini.
    """
    num_frames = len(hand_data)
    print(f"\n[HAND TRACKER] Captured {num_frames} frames of hand movement!")
    print(f"[HAND TRACKER] This is where we would send data to Gemini or the local model.")
    
    # Placeholder translation
    translation = "Hello"
    
    # Push the translation to the mailbox so the screen can show it
    sign_queue.put({
        "translation": translation,
        "frames_captured": num_frames,
        "timestamp": time.time()
    })
    
    print(f"[HAND TRACKER] Translation: '{translation}' → sent to screen!")
    return translation

# ============================================================
# STEP 4: Main camera loop
# ============================================================
def start_hand_tracker():
    global hand_was_visible, frames_without_hands, collected_frames
    
    cap = cv2.VideoCapture(0)
    print("========================================")
    print("   VOICELENS HAND TRACKER ACTIVE        ")
    print("========================================")
    print("Show your hands to the camera to start.")
    print("Drop your hands to trigger translation.")
    print("Press 'q' to quit.")
    print("========================================")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Flip the frame so it looks like a mirror
        frame = cv2.flip(frame, 1)
        
        # Convert colors for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # ---------------------------------------------------
        # CASE 1: Hands ARE visible on screen
        # ---------------------------------------------------
        if results.multi_hand_landmarks:
            hand_was_visible = True
            frames_without_hands = 0  # Reset the disappear counter
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the green skeleton on the hand
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Collect the 21 landmark coordinates for this frame
                frame_data = []
                for lm in hand_landmarks.landmark:
                    frame_data.append([lm.x, lm.y, lm.z])
                
                collected_frames.append(frame_data)
            
            # Show status on screen
            cv2.putText(frame, "RECORDING HANDS...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {len(collected_frames)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # ---------------------------------------------------
        # CASE 2: Hands are NOT visible
        # ---------------------------------------------------
        else:
            if hand_was_visible:
                frames_without_hands += 1
                
                # Show countdown on screen
                remaining = DISAPPEAR_THRESHOLD - frames_without_hands
                if remaining > 0:
                    cv2.putText(frame, f"Hands gone... sending in {remaining}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # If hands have been gone long enough, trigger translation
                if frames_without_hands >= DISAPPEAR_THRESHOLD:
                    if len(collected_frames) > 5:  # Need at least a few frames
                        translate_sign(collected_frames)
                    
                    # Reset everything for the next sign
                    collected_frames = []
                    hand_was_visible = False
                    frames_without_hands = 0
            else:
                # No hands and we weren't tracking — just show waiting message
                cv2.putText(frame, "Show your hands to start...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Show the camera feed
        cv2.imshow("VoiceLens - Hand Tracker", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# STEP 5: Run it
# ============================================================
if __name__ == "__main__":
    start_hand_tracker()
