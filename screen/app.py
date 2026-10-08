# screen/app.py
# VoiceLens Smart Glasses Display.
# Shows live speech captions and who is speaking on a dark, futuristic screen.
import sys
import os

# Make sure Python can find all project folders
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import time
from shared.mailbox import caption_queue, sign_queue, live_state
from mic.listener import Listener

# Set up browser tab title and wide layout
st.set_page_config(
    page_title="VoiceLens - Smart Glasses Simulator",
    page_icon="👓",
    layout="wide"
)

# Styling for the dark glasses display
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .caption-box {
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        background-color: #1E222D;
        border-left: 6px solid #4A90D9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .speaker-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .caption-text {
        font-size: 1.25rem;
        font-weight: 500;
        color: #FFFFFF;
        line-height: 1.4;
    }
    .status-bar {
        padding: 10px;
        border-radius: 8px;
        background-color: #1A1F2C;
        border: 1px solid #2D3748;
        font-size: 0.9rem;
        color: #A0AEC0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and subtitle header
col_header, col_status = st.columns([3, 1])
with col_header:
    st.title("🔮 VoiceLens Glasses Simulator")
    st.caption("Hackcessible India 2026 — Live Communication Bridge for the Deaf")

# Keep track of engine status and caption history across page reloads
if "engine_started" not in st.session_state:
    st.session_state.engine_started = False
    st.session_state.captions = []

# Buttons to turn listening on and off
with col_status:
    if not st.session_state.engine_started:
        if st.button("🎙️ START LISTENING", use_container_width=True, type="primary"):
            st.session_state.engine = Listener()
            st.session_state.engine.start()
            st.session_state.engine_started = True
            st.rerun()
    else:
        st.markdown("🟢 **Microphone Listening**")
        if st.button("⏹️ STOP", use_container_width=True):
            st.session_state.engine.stop()
            st.session_state.engine_started = False
            st.rerun()

# Two columns: left shows captions, right shows sign language camera
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("👓 Live Glasses Display (What Vikram Sees)")
    caption_container = st.empty()

with col_right:
    st.subheader("📷 Sign Language Camera (Member 2)")
    st.info("Member 2 will plug hand tracking & ISL sign detection here!")
    st.markdown("""
    ```
    +-----------------------------+
    |                             |
    |   [ WEBCAM STREAM WITH     |
    |     GREEN HAND SKELETON ]   |
    |                             |
    +-----------------------------+
    Detected Sign: [ Hello ]
    Spoken Voice: "Hello, nice to meet you."
    ```
    """)

# Keep checking for new captions and update the screen
if st.session_state.engine_started:
    # Get all new captions from the mailbox
    while not caption_queue.empty():
        new_caption = caption_queue.get()
        st.session_state.captions.insert(0, new_caption)  # Newest at top
        if len(st.session_state.captions) > 15:
            st.session_state.captions.pop()

    # Draw each caption on screen
    with col_left:
        html_content = ""
        for c in st.session_state.captions:
            speaker_color = c.get("speaker_color", "#4A90D9")
            
            html_content += f"""
            <div class="caption-box" style="border-left-color: {speaker_color};">
                <div class="speaker-label" style="color: {speaker_color};">{c['speaker_label']}</div>
                <div class="caption-text">{c['text']}</div>
            </div>
            """
        
        if html_content:
            caption_container.markdown(html_content, unsafe_allow_html=True)
        else:
            caption_container.info("🎙️ Speak into your mic to see captions appear in real-time...")

    time.sleep(0.3)
    st.rerun()
