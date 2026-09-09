# run_system.py
import warnings
warnings.filterwarnings("ignore")

import threading
from inbound.stt_engine import STTEngine
from outbound.visual_speaker import main as run_camera

# Start Audio Engine in background
engine = STTEngine()
audio_thread = threading.Thread(target=engine.start, daemon=True)
audio_thread.start()

# Start Camera in foreground
run_camera()

# Clean up STT
engine.stop()