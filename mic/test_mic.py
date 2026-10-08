# mic/test_mic.py
# Quick test to check if your microphone is working.
import speech_recognition as sr

def test_microphone():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    print("========================================")
    print("       VOICELENS MIC TEST               ")
    print("========================================")
    print("Searching for microphone...")

    try:
        with sr.Microphone() as source:
            print("Calibrating background noise... (stay quiet for 1s)")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("\nREADY! Speak clearly into your mic now...")
            print("----------------------------------------")

            audio = recognizer.listen(source, timeout=6, phrase_time_limit=10)
            print("Audio captured! Sending to Google...")
            
            text = recognizer.recognize_google(audio)
            print("\nSUCCESS! What VoiceLens heard:")
            print(f'>> "{text}"')
            print("========================================\n")
            
    except sr.WaitTimeoutError:
        print("\nNo speech detected (you did not speak within 6 seconds).")
    except sr.UnknownValueError:
        print("\nVoice was detected, but could not understand the words.")
    except sr.RequestError as e:
        print(f"\nNetwork error: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    test_microphone()
