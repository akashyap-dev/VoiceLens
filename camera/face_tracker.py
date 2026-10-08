# camera/face_tracker.py
# Watches faces on the webcam and detects who is talking by checking mouth movement.
import cv2
import mediapipe as mp
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
from shared.mailbox import live_state

# Landmark numbers for the top and bottom lip
UPPER_LIP = 13
LOWER_LIP = 14

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    # Set up face mesh to track up to 4 faces
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
                
            # Convert colors so MediaPipe can process the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            h, w, _ = frame.shape
            
            if results.multi_face_landmarks:
                for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                    
                    # 1. Get vertical position of upper and lower lips
                    upper_y = face_landmarks.landmark[UPPER_LIP].y * h
                    lower_y = face_landmarks.landmark[LOWER_LIP].y * h
                    
                    # 2. Measure distance between the lips
                    mouth_open_distance = lower_y - upper_y
                    
                    # 3. If lips are more than 5 pixels apart, person is talking
                    is_speaking = mouth_open_distance > 5.0  
                    
                    # 4. Find bounding box coordinates to draw around the face
                    x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # Green box if talking, Red box if quiet
                    if is_speaking:
                        color = (0, 255, 0)
                        label = f"Speaker {face_id + 1}: TALKING"
                        # Save the current speaker to the shared notepad
                        live_state["current_visual_speaker"] = f"Speaker {face_id + 1}"
                    else:
                        color = (0, 0, 255)
                        label = f"Speaker {face_id + 1}: Quiet"
                        
                    # Draw box around face
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # Draw label banner and text above face
                    cv2.rectangle(frame, (x_min, y_min - 30), (x_max, y_min), color, -1)
                    cv2.putText(frame, label, (x_min + 5, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('VoiceLens - Face Tracker', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
