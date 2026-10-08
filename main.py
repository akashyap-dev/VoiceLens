# main.py
# This is the start button for VoiceLens.
# It turns on the microphone AND the camera at the same time.

import warnings
warnings.filterwarnings("ignore")  # Hide ugly warning messages

import threading
from mic.listener import Listener
from camera.face_tracker import main as run_camera

# Step 1: Start the microphone listener in the background
engine = Listener()
audio_thread = threading.Thread(target=engine.start, daemon=True)
audio_thread.start()

# Step 2: Start the camera (this takes over the screen until you press 'q')
run_camera()

# Step 3: When camera closes, stop the microphone too
engine.stop()
