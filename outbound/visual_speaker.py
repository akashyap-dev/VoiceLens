import cv2
import mediapipe as mp
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing

# NEW: Import the whiteboard
from shared.queues import live_state
# outbound/visual_speaker.py
import cv2
import mediapipe as mp

# Explicitly force Python to load the solutions module
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing

# We use indices 13 (Upper Lip) and 14 (Lower Lip) to measure mouth opening
UPPER_LIP = 13
LOWER_LIP = 14

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
        # Setup Face Mesh to track multiple faces
    with mp_face_mesh.FaceMesh(
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Convert the BGR image to RGB before processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            h, w, _ = frame.shape
            
            if results.multi_face_landmarks:
                for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                    
                    # 1. Get the coordinates of the upper and lower lip
                    upper_y = face_landmarks.landmark[UPPER_LIP].y * h
                    lower_y = face_landmarks.landmark[LOWER_LIP].y * h
                    
                    # 2. Calculate the distance between them
                    mouth_open_distance = lower_y - upper_y
                    
                    # 3. Determine if speaking based on a threshold
                    # (You might need to tweak this number based on how close you are to the camera)
                    is_speaking = mouth_open_distance > 5.0  
                    
                    # 4. Find the bounding box of the face to draw a rectangle
                    x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # --- YOUR TURN TO DRAW ---
                    # If they are speaking, make the box GREEN and write "Speaker [X]: TALKING"
                    # If they are quiet, make the box RED and write "Speaker [X]: Quiet"
                    
                                        # --- INSIDE YOUR FACE LOOP ---
                    if is_speaking:
                        color = (0, 255, 0) # Green
                        label = f"Speaker {face_id + 1}: TALKING"
                        # NEW: Write it to the whiteboard!
                        live_state["current_visual_speaker"] = f"Speaker {face_id + 1}"
                    else:
                        color = (0, 0, 255) # Red
                        label = f"Speaker {face_id + 1}: Quiet"
                        
                    # Draw Rectangle
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # Draw Text Background & Text
                    cv2.rectangle(frame, (x_min, y_min - 30), (x_max, y_min), color, -1)
                    cv2.putText(frame, label, (x_min + 5, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('Visual Speaker Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main