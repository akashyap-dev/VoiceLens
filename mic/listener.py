# mic/listener.py
# This file listens to the microphone and figures out what people are saying.
# It also figures out WHO is speaking using voice fingerprints.
import time
import speech_recognition as sr
from shared.mailbox import caption_queue, live_state
from mic.voice_id import VoiceID

class Listener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.mic = sr.Microphone()
        # Identifies who is speaking by comparing how their voice sounds
        self.voice_id = VoiceID(similarity_threshold=0.92)
        self.stop_fn = None

    def _audio_callback(self, recognizer, audio):
        print("[DEBUG: The mic heard you and is processing...]")
        try:
            text = recognizer.recognize_google(audio)
            if not text.strip():
                return

            now = time.time()

            # 1. Figure out who is talking by matching their voice
            spk_id, spk_label, spk_color, conf = self.voice_id.identify_speaker(audio)

            # 2. Check who the camera thinks is talking (face tracking)
            camera_speaker = live_state.get("current_visual_speaker", "Unknown")

            # 3. Package the caption with speaker info
            caption = {
                "speaker_id": spk_id,
                "speaker_label": spk_label,
                "visual_speaker": camera_speaker,
                "speaker_color": spk_color,
                "match_confidence": conf,
                "text": text,
                "timestamp": now
            }

            # 4. Drop the caption in the mailbox for the screen to display
            caption_queue.put(caption)

            # Print to the terminal so we can see what's happening
            print(f"[Audio: {spk_label} | Camera: {camera_speaker}] \"{text}\"")

        except sr.UnknownValueError:
            print("[DEBUG: Heard noise, but couldn't understand the words]")
        except sr.RequestError as e:
            print(f"[Network Error]: {e}")

    def start(self):
        """Adjusts for background noise and starts listening."""
        print("========================================")
        print("       VOICELENS LISTENER ACTIVE        ")
        print("========================================")
        with self.mic as source:
            print("Calibrating background noise... (1s)")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("Voice identification active!")
        print("Different voices will be automatically told apart.")
        print("----------------------------------------")
        
        self.stop_fn = self.recognizer.listen_in_background(
            self.mic, 
            self._audio_callback, 
            phrase_time_limit=8
        )

    def stop(self):
        if self.stop_fn:
            self.stop_fn(wait_for_stop=False)
            print("Stopped listening.")

if __name__ == "__main__":
    engine = Listener()
    engine.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        engine.stop()
