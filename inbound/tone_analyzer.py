# inbound/tone_analyzer.py
"""
Tone & Emotion Analyzer for Member 1 (The Ears).
Analyzes raw audio waveform features (RMS energy & frequency dynamics)
to map speech into visual emotional indicators for Vikram.
"""
import numpy as np

def analyze_audio_tone(audio_data):
    """
    Takes a SpeechRecognition AudioData object, extracts acoustic metrics,
    and returns an emotional tone classification with hex color code.
    """
    try:
        # Convert audio to 16kHz, 16-bit mono PCM samples
        raw_bytes = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(samples) == 0:
            return _default_tone()

        # 1. RMS Energy (Volume / Intensity)
        rms = float(np.sqrt(np.mean(samples**2)))

        # 2. Zero Crossing Rate (ZCR - High-frequency / pitch proxy)
        zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2.0)

        # 3. Emotion Decision Matrix
        # Thresholds tuned for typical PC/laptop condenser mics
        if rms > 0.12:
            if zcr > 0.11:
                emotion = "Excited / Animated"
                color = "#9C27B0"  # Purple
            else:
                emotion = "Tense / Urgent"
                color = "#FF5722"  # Deep Orange / Red
        elif rms < 0.035:
            emotion = "Quiet / Hesitant"
            color = "#2196F3"      # Blue
        else:
            if zcr > 0.09:
                emotion = "Engaged / Curious"
                color = "#8BC34A"  # Light Green
            else:
                emotion = "Calm / Neutral"
                color = "#4CAF50"  # Green

        return {
            "energy": round(rms, 3),
            "pitch_freq": round(zcr, 3),
            "emotion": emotion,
            "emotion_color": color
        }

    except Exception as e:
        print(f"[Tone Analyzer Error]: {e}")
        return _default_tone()

def _default_tone():
    return {
        "energy": 0.05,
        "pitch_freq": 0.05,
        "emotion": "Calm / Neutral",
        "emotion_color": "#4CAF50"
    }

if __name__ == "__main__":
    print("Tone analyzer loaded successfully.")
