# mic/register.py
# Run this script before the hackathon demo to register your team's voices.
# It saves their voice fingerprint and their real name to a file.

import speech_recognition as sr
import json
import os
import sys

# Make sure Python can find the mic folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mic.voice_id import VoiceID

PROFILES_FILE = os.path.join(os.path.dirname(__file__), "profiles.json")

def register_voice():
    print("========================================")
    print("      VOICELENS VOICE REGISTRATION      ")
    print("========================================")
    
    name = input("Enter the name of the person to register: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    tracker = VoiceID()

    print(f"\n[GET READY] {name}, we will record your voice for 5 seconds.")
    print("Please talk normally when it says 'SPEAK NOW'.")
    input("Press Enter when you are ready to start...")

    try:
        with sr.Microphone() as source:
            print("\nCalibrating background noise (1 second)...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("\n>>> SPEAK NOW! <<< (Talk for 5 seconds)")
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=5)
            print("Recording finished. Processing your voice fingerprint...")

            # Extract the unique voice fingerprint
            fingerprint = tracker.extract_voice_fingerprint(audio)
            
            if fingerprint is None:
                print("\n❌ Error: Audio was too short or silent. Please try again.")
                return

            # Load any existing profiles so we don't delete them
            profiles = {}
            if os.path.exists(PROFILES_FILE):
                with open(PROFILES_FILE, 'r') as f:
                    profiles = json.load(f)

            # Save this person's fingerprint under their name
            profiles[name] = fingerprint.tolist()
            
            with open(PROFILES_FILE, 'w') as f:
                json.dump(profiles, f, indent=4)
                
            print(f"\n✅ SUCCESS! Voice profile for '{name}' has been saved.")
            print(f"When {name} speaks off-camera, the system will now recognize them!")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    register_voice()
