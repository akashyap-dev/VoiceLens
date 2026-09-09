# inbound/speaker_tracker.py
"""
Acoustic Speaker Biometrics & Diarization for Member 1.
Identifies distinct speakers based on vocal tract timbre (MFCC voice embeddings)
and cosine similarity — ZERO reliance on silence delay.
"""
import numpy as np
import librosa

SPEAKER_COLORS = ["#4A90D9", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

class AcousticSpeakerTracker:
    def __init__(self, similarity_threshold=0.82):
        # Maps speaker_id -> list of historical voice embedding vectors
        self.speaker_profiles = {}
        self.similarity_threshold = similarity_threshold
        self.next_speaker_id = 1

    def extract_voice_embedding(self, audio_data):
        """Extracts a normalized timbre fingerprint vector from raw PCM audio."""
        raw_bytes = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(samples) < 1600:  # Need at least 0.1s of audio
            return None

        # Extract 20 Mel-Frequency Cepstral Coefficients (acoustic timbre fingerprint)
        mfcc = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=20)
        # Average across time frames to get a static voice signature vector
        # CRITICAL FIX: Skip the 0th coefficient (mfcc[1:]) because it only tracks volume/energy, not vocal timbre!
        embedding = np.mean(mfcc[1:], axis=1)

        # Standardize vector (zero mean, unit variance) for Pearson correlation matching
        std_val = np.std(embedding)
        if std_val > 1e-6:
            embedding = (embedding - np.mean(embedding)) / std_val

        # L2 Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding

    def identify_speaker(self, audio_data):
        """
        Compares incoming voice fingerprint against all registered speakers.
        Returns: (speaker_id, speaker_label, speaker_color, confidence)
        """
        embedding = self.extract_voice_embedding(audio_data)

        # Fallback if audio is too short
        if embedding is None:
            return 1, "Speaker 1", SPEAKER_COLORS[0], 1.0

        # First speaker registration
        if not self.speaker_profiles:
            self.speaker_profiles[1] = [embedding]
            return 1, "Speaker 1", SPEAKER_COLORS[0], 1.0

        best_speaker_id = None
        highest_similarity = -1.0

        # Compare with each known speaker profile using Cosine Similarity
        for spk_id, profile_embeddings in self.speaker_profiles.items():
            # Average similarity against the speaker's known embeddings
            sims = [float(np.dot(embedding, p_emb)) for p_emb in profile_embeddings]
            avg_sim = np.mean(sims)

            if avg_sim > highest_similarity:
                highest_similarity = avg_sim
                best_speaker_id = spk_id

        # If similarity exceeds threshold, it's a known speaker
        if highest_similarity >= self.similarity_threshold:
            # Update their voice profile with the new sample (keep last 5 samples)
            self.speaker_profiles[best_speaker_id].append(embedding)
            if len(self.speaker_profiles[best_speaker_id]) > 5:
                self.speaker_profiles[best_speaker_id].pop(0)

            color = SPEAKER_COLORS[(best_speaker_id - 1) % len(SPEAKER_COLORS)]
            print(f"[DEBUG MFCC] Matched Speaker {best_speaker_id} with similarity {highest_similarity:.3f}")
            return best_speaker_id, f"Speaker {best_speaker_id}", color, round(highest_similarity, 2)

        # Otherwise, a NEW speaker has spoken!
        if len(self.speaker_profiles) < len(SPEAKER_COLORS):
            self.next_speaker_id += 1
            new_id = self.next_speaker_id
            self.speaker_profiles[new_id] = [embedding]
            color = SPEAKER_COLORS[(new_id - 1) % len(SPEAKER_COLORS)]
            print(f"[DEBUG MFCC] Similarity was only {highest_similarity:.3f}. Spawning NEW Speaker {new_id}!")
            return new_id, f"Speaker {new_id}", color, 1.0
        else:
            # Max speakers reached, assign closest match
            color = SPEAKER_COLORS[(best_speaker_id - 1) % len(SPEAKER_COLORS)]
            return best_speaker_id, f"Speaker {best_speaker_id}", color, round(highest_similarity, 2)

if __name__ == "__main__":
    tracker = AcousticSpeakerTracker()
    print("Acoustic Speaker Biometrics Engine loaded.")
