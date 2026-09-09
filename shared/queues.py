# shared/queues.py
import queue

caption_queue = queue.Queue()
tone_queue = queue.Queue()
sign_queue = queue.Queue()

# NEW: The "Whiteboard" for live state
live_state = {
    "current_visual_speaker": "Unknown"
}