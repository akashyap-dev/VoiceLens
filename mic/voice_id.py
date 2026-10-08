# mic/voice_id.py
# This file figures out WHO is talking by comparing how their voice sounds.
# It creates a unique voice fingerprint for each person.
import numpy as np
import librosa

SPEAKER_COLORS = ["#4A90D9", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

class VoiceID:
    def __init__(self, similarity_threshold=0.82):
        # Stores voice fingerprints for each speaker
        self.speaker_profiles = {}
        self.similarity_threshold = similarity_threshold
        self.next_speaker_id = 1

    def extract_voice_fingerprint(self, audio_data):
        """Creates a unique voice fingerprint from the recorded sound."""
        # Convert the audio into a list of sound numbers
        raw_bytes = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(samples) < 1600:  # Need at least 0.1 seconds of audio
            return None

        # Calculate sound features to build a voice fingerprint
        mfcc = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=20)
        # Average the features over time to make a single voice profile.
        # Skip the first value because it only tracks volume, not how the voice sounds!
        fingerprint = np.mean(mfcc[1:], axis=1)

        # Center and scale the numbers so they are easy to compare
        std_val = np.std(fingerprint)
        if std_val > 1e-6:
            fingerprint = (fingerprint - np.mean(fingerprint)) / std_val

        # Scale the fingerprint to a standard length
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

        # Fallback if audio is too short
        if fingerprint is None:
            return 1, "Speaker 1", SPEAKER_COLORS[0], 1.0

        # Register the very first speaker
        if not self.speaker_profiles:
            self.speaker_profiles[1] = [fingerprint]
            return 1, "Speaker 1", SPEAKER_COLORS[0], 1.0

        best_speaker_id = None
        highest_similarity = -1.0

        # Compare with each known speaker to see how similar the voices sound
        for spk_id, saved_fingerprints in self.speaker_profiles.items():
            sims = [float(np.dot(fingerprint, saved)) for saved in saved_fingerprints]
            avg_sim = np.mean(sims)

            if avg_sim > highest_similarity:
                highest_similarity = avg_sim
                best_speaker_id = spk_id

        # If similarity is high enough, we found a match!
        if highest_similarity >= self.similarity_threshold:
            # Save this new sample to make future matching better (keep last 5)
            self.speaker_profiles[best_speaker_id].append(fingerprint)
            if len(self.speaker_profiles[best_speaker_id]) > 5:
                self.speaker_profiles[best_speaker_id].pop(0)

            color = SPEAKER_COLORS[(best_speaker_id - 1) % len(SPEAKER_COLORS)]
            print(f"[DEBUG] Matched Speaker {best_speaker_id} (similarity: {highest_similarity:.3f})")
            return best_speaker_id, f"Speaker {best_speaker_id}", color, round(highest_similarity, 2)

        # Otherwise, a NEW speaker has spoken!
        if len(self.speaker_profiles) < len(SPEAKER_COLORS):
            self.next_speaker_id += 1
            new_id = self.next_speaker_id
            self.speaker_profiles[new_id] = [fingerprint]
            color = SPEAKER_COLORS[(new_id - 1) % len(SPEAKER_COLORS)]
            print(f"[DEBUG] New voice detected! Registered as Speaker {new_id}")
            return new_id, f"Speaker {new_id}", color, 1.0
        else:
            # Max speakers reached, pick the closest match
            color = SPEAKER_COLORS[(best_speaker_id - 1) % len(SPEAKER_COLORS)]
            return best_speaker_id, f"Speaker {best_speaker_id}", color, round(highest_similarity, 2)

if __name__ == "__main__":
    tracker = VoiceID()
    print("Voice ID Engine loaded.")
