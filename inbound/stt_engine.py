"""
Continuous Background Speech-to-Text, Acoustic Biometric Diarization & Tone Engine.
Zero reliance on silence delay — uses vocal timbre embeddings to identify speakers!
Now featuring Multi-Modal Diarization (Acoustic + Visual Face Tracking)
"""
import time
import speech_recognition as sr
from shared.queues import caption_queue, tone_queue, live_state
from inbound.tone_analyzer import analyze_audio_tone
from inbound.speaker_tracker import AcousticSpeakerTracker

class STTEngine:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.mic = sr.Microphone()
        # Biometric Acoustic Diarization (Vocal Timbre MFCC Matching)
        self.speaker_tracker = AcousticSpeakerTracker(similarity_threshold=0.92)
        self.stop_fn = None

    def _audio_callback(self, recognizer, audio):
        print("[DEBUG: Audio callback triggered! The mic heard you and is processing...]")
        try:
            text = recognizer.recognize_google(audio)
            if not text.strip():
                return

            now = time.time()

            # 1. ACOUSTIC SPEAKER DIARIZATION (NO SILENCE DELAY)
            # Uses voice frequency fingerprinting to identify exact speaker
            spk_id, spk_label, spk_color, conf = self.speaker_tracker.identify_speaker(audio)

            # 2. EXTRACT ACOUSTIC TONE & EMOTION
            tone_data = analyze_audio_tone(audio)
            tone_data["speaker_id"] = spk_id
            tone_data["timestamp"] = now

            # 3. DROP INTO tone_queue
            tone_queue.put(tone_data)

            # === 4. NEW VISUAL CHECK ===
            # Look at the whiteboard to see whose mouth is currently moving
            camera_speaker = live_state.get("current_visual_speaker", "Unknown")

            # 5. BUILD COMPLETE CAPTION PAYLOAD
            caption_payload = {
                "speaker_id": spk_id,
                "speaker_label": spk_label,       # Acoustic guess
                "visual_speaker": camera_speaker, # Camera guess
                "speaker_color": spk_color,
                "match_confidence": conf,
                "text": text,
                "tone": tone_data,
                "timestamp": now
            }

            # 6. DROP INTO caption_queue (UI reads this)
            caption_queue.put(caption_payload)

            # Terminal print with BOTH biometrics and emotion
            print(f"[Audio: {spk_label} | Camera: {camera_speaker} | {tone_data['emotion']}] \"{text}\"")

        except sr.UnknownValueError:
            print("[DEBUG: Heard noise, but Google STT couldn't understand the words]")
        except sr.RequestError as e:
            print(f"[STT Network Error]: {e}")

    def start(self):
        """Calibrates mic and starts non-blocking background listening thread."""
        print("========================================")
        print("  VOICELENS ACOUSTIC + VISUAL ENGINE    ")
        print("========================================")
        with self.mic as source:
            print("Calibrating background noise... (1s)")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("Acoustic voice fingerprinting active!")
        print("Different voices will be automatically distinguished by vocal timbre.")
        print("Visual face tracking integration ready.")
        print("----------------------------------------")
        
        self.stop_fn = self.recognizer.listen_in_background(
            self.mic, 
            self._audio_callback, 
            phrase_time_limit=8
        )

    def stop(self):
        if self.stop_fn:
            self.stop_fn(wait_for_stop=False)
            print("Stopped listening engine.")

if __name__ == "__main__":
    engine = STTEngine()
    engine.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        engine.stop()