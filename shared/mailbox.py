# shared/mailbox.py
# This is the mailbox. Different parts of VoiceLens drop messages here
# and other parts pick them up.
import queue

# Caption messages (what someone said + who said it)
caption_queue = queue.Queue()

# Sign language translations from the camera
sign_queue = queue.Queue()

# Shared notepad that different parts of the app can read and write to
live_state = {
    "current_visual_speaker": "Unknown"
}
