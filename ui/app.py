# ui/app.py
"""
VoiceLens Live Dashboard (The Glasses Simulator).
Reads from shared/queues.py and displays live color-coded captions,
speaker indicators, and emotional tone bars in real-time.
"""
import sys
import os

# Ensure the root voicelens folder is in Python's search path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import time
from shared.queues import caption_queue, tone_queue, sign_queue, sound_queue
from inbound.stt_engine import STTEngine

st.set_page_config(
    page_title="VoiceLens - Smart Glasses Simulator",
    page_icon="👓",
    layout="wide"
)

# Custom CSS for dark glassmorphism look
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
    .tone-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 8px;
        color: #FFFFFF;
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

# App Header
col_header, col_status = st.columns([3, 1])
with col_header:
    st.title("🔮 VoiceLens Glasses Simulator")
    st.caption("Hackcessible India 2026 — Live Perception-Action Communication Bridge")

# Initialize Session State
if "engine_started" not in st.session_state:
    st.session_state.engine_started = False
    st.session_state.captions = []

# Top Controls
with col_status:
    if not st.session_state.engine_started:
        if st.button("🎙️ START LISTENING", use_container_width=True, type="primary"):
            st.session_state.engine = STTEngine()
            st.session_state.engine.start()
            st.session_state.engine_started = True
            st.rerun()
    else:
        st.markdown("🟢 **Microphone Listening**")
        if st.button("⏹️ STOP ENGINE", use_container_width=True):
            st.session_state.engine.stop()
            st.session_state.engine_started = False
            st.rerun()

# Layout: Left = What Vikram Sees (Inbound), Right = Camera / Signs (Outbound)
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("👓 Live Glasses Display (What Vikram Sees)")
    caption_container = st.empty()

with col_right:
    st.subheader("📷 Outbound View (Member 2's Domain)")
    st.info("Member 2 will plug MediaPipe hand tracking & ISL sign detection right here!")
    st.markdown("""
    ```
    +-----------------------------+
    |                             |
    |   [ WEBCAM STREAM WITH     |
    |     GREEN LANDMARK DOTS ]   |
    |                             |
    +-----------------------------+
    Detected Sign: [ Namaste / Hello ]
    🗣️ Spoken Voice: "Hello, nice to meet you."
    ```
    """)

# Continuous UI Refresh Loop
if st.session_state.engine_started:
    # Pull any new captions from the queue
    while not caption_queue.empty():
        new_caption = caption_queue.get()
        st.session_state.captions.insert(0, new_caption)  # Add newest at top
        if len(st.session_state.captions) > 15:
            st.session_state.captions.pop()

    # Render captions
    with col_left:
        html_content = ""
        for c in st.session_state.captions:
            speaker_color = c.get("speaker_color", "#4A90D9")
            tone = c.get("tone", {})
            emotion = tone.get("emotion", "Calm / Neutral")
            emotion_color = tone.get("emotion_color", "#4CAF50")
            
            html_content += f"""
            <div class="caption-box" style="border-left-color: {speaker_color};">
                <div class="speaker-label" style="color: {speaker_color};">{c['speaker_label']}</div>
                <div class="caption-text">{c['text']}</div>
                <div class="tone-badge" style="background-color: {emotion_color};">
                    Tone: {emotion}
                </div>
            </div>
            """
        
        if html_content:
            caption_container.markdown(html_content, unsafe_allow_html=True)
        else:
            caption_container.info("🎙️ Speak into your mic to see captions appear in real-time...")

    time.sleep(0.3)
    st.rerun()
