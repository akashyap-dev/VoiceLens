# mic/voice_id.py
# This file figures out WHO is talking by comparing how their voice sounds.
# It creates a unique voice fingerprint for each person and loads pre-registered names.
import numpy as np
import librosa
import json
import os

SPEAKER_COLORS = ["#4A90D9", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

class VoiceID:
    def __init__(self, similarity_threshold=0.82):
        # Stores voice fingerprints for each speaker id
        self.speaker_profiles = {}
        # Stores the names for each speaker id (e.g. 1: "Rahul")
        self.speaker_names = {}
        self.similarity_threshold = similarity_threshold
        self.next_speaker_id = 1
        
        # Look for the saved voices file
        self.profiles_file = os.path.join(os.path.dirname(__file__), "profiles.json")
        self.load_profiles()

    def load_profiles(self):
        """Loads pre-registered voice profiles from the JSON file."""
        if os.path.exists(self.profiles_file):
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)
                for name, fingerprint in data.items():
                    self.speaker_profiles[self.next_speaker_id] = [np.array(fingerprint)]
                    self.speaker_names[self.next_speaker_id] = name
                    self.next_speaker_id += 1
            print(f"[VOICE ID] Loaded {len(data)} pre-registered voices from profiles.json")

    def extract_voice_fingerprint(self, audio_data):
        """Creates a unique voice fingerprint from the recorded sound."""
        raw_bytes = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(samples) < 1600:
            return None

        mfcc = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=20)
        fingerprint = np.mean(mfcc[1:], axis=1)

        std_val = np.std(fingerprint)
        if std_val > 1e-6:
            fingerprint = (fingerprint - np.mean(fingerprint)) / std_val

        norm = np.linalg.norm(fingerprint)
        if norm > 1e-6:
            fingerprint = fingerprint / norm

        return fingerprint

    def identify_speaker(self, audio_data):
        """
        Figures out who is talking by comparing their voice to saved speakers.
        Returns: (speaker_id, speaker_label, speaker_color, confidence)
        """
        fingerprint = self.extract_voice_fingerprint(audio_data)

        if fingerprint is None:
            name = self.speaker_names.get(1, "Guest 1")
            return 1, name, SPEAKER_COLORS[0], 1.0

        # If no profiles exist at all, register the first guest
        if not self.speaker_profiles:
            self.speaker_profiles[1] = [fingerprint]
            self.speaker_names[1] = "Guest 1"
            self.next_speaker_id = 2
            return 1, "Guest 1", SPEAKER_COLORS[0], 1.0

        best_speaker_id = None
        highest_similarity = -1.0

        # Compare with each known speaker
        for spk_id, saved_fingerprints in self.speaker_profiles.items():
            sims = [float(np.dot(fingerprint, saved)) for saved in saved_fingerprints]
            avg_sim = np.mean(sims)

            if avg_sim > highest_similarity:
                highest_similarity = avg_sim
                best_speaker_id = spk_id

        # If it matches someone (pre-registered or a previous guest)
        if highest_similarity >= self.similarity_threshold:
            # Keep their profile updated with their latest voice
            self.speaker_profiles[best_speaker_id].append(fingerprint)
            if len(self.speaker_profiles[best_speaker_id]) > 5:
                self.speaker_profiles[best_speaker_id].pop(0)

            color = SPEAKER_COLORS[(best_speaker_id - 1) % len(SPEAKER_COLORS)]
            name = self.speaker_names[best_speaker_id]
            print(f"[DEBUG] Matched {name} (similarity: {highest_similarity:.3f})")
            return best_speaker_id, name, color, round(highest_similarity, 2)

        # NEW SPEAKER DETECTED (Not registered, hasn't spoken yet today)
        new_id = self.next_speaker_id
        self.next_speaker_id += 1
        
        self.speaker_profiles[new_id] = [fingerprint]
        guest_name = f"Guest {new_id}"
        self.speaker_names[new_id] = guest_name
        
        color = SPEAKER_COLORS[(new_id - 1) % len(SPEAKER_COLORS)]
        print(f"[DEBUG] New voice detected! Registered as {guest_name}")
        return new_id, guest_name, color, 1.0

if __name__ == "__main__":
    tracker = VoiceID()
    print("Voice ID Engine loaded.")
