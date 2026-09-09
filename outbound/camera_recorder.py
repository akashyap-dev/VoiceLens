# outbound/camera_recorder.py
"""
Live Camera & Video Recording Engine for Member 2 (The Hands).
Streams live webcam frames and records MP4 clips with visual timestamps & REC badges.
"""
import os
import time
import cv2

RECORDINGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "recordings")
)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

class CameraRecorder:
    def __init__(self, camera_index=0, width=640, height=480, fps=20.0):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.current_recording_path = None
        self.frame_count = 0

    def open(self):
        """Initializes the webcam device."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self.cap.isOpened()

    def start_recording(self, custom_filename=None):
        """Starts recording video frames to an MP4 file."""
        if not self.open():
            print("Cannot start recording: Camera not accessible.")
            return None

        if self.is_recording:
            return self.current_recording_path

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = custom_filename or f"isl_clip_{timestamp}.mp4"
        self.current_recording_path = os.path.join(RECORDINGS_DIR, filename)

        # Use mp4v or XVID codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.current_recording_path, fourcc, self.fps, (self.width, self.height)
        )
        self.is_recording = True
        self.frame_count = 0
        print(f"🔴 Recording started: {self.current_recording_path}")
        return self.current_recording_path

    def stop_recording(self):
        """Stops the current recording and finalizes the MP4 file."""
        if self.is_recording and self.writer:
            self.is_recording = False
            self.writer.release()
            self.writer = None
            print(f"⏹️ Recording saved ({self.frame_count} frames): {self.current_recording_path}")
            saved_path = self.current_recording_path
            self.current_recording_path = None
            return saved_path
        return None

    def read_frame(self):
        """
        Reads a frame from the webcam.
        If recording, writes to file and adds visual REC indicator overlay.
        Returns: (success: bool, frame_bgr: np.ndarray)
        """
        if not self.open():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # If recording, write raw frame to file
        if self.is_recording and self.writer:
            self.writer.write(frame)
            self.frame_count += 1

            # Draw visual recording badge overlay on the displayed copy
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red dot
            cv2.putText(
                frame, "REC", (50, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            elapsed_sec = int(self.frame_count / self.fps)
            time_str = f"{elapsed_sec // 60:02d}:{elapsed_sec % 60:02d}"
            cv2.putText(
                frame, time_str, (105, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        return True, frame

    def close(self):
        """Releases camera and video writer resources."""
        self.stop_recording()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

if __name__ == "__main__":
    cam = CameraRecorder()
    if not cam.open():
        print("Failed to open camera.")
        exit(1)

    print("========================================")
    print("      VOICELENS CAMERA & RECORDER       ")
    print("========================================")
    print("Controls in window:")
    print("  'r' -> Toggle Record On/Off")
    print("  'q' -> Quit")
    print("========================================")

    while True:
        success, frame = cam.read_frame()
        if not success:
            break

        cv2.imshow("VoiceLens Camera Preview", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            if cam.is_recording:
                cam.stop_recording()
            else:
                cam.start_recording()
        elif key == ord("q"):
            break

    cam.close()
    cv2.destroyAllWindows()
