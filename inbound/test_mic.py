# inbound/test_mic.py
import speech_recognition as sr

def test_microphone():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    print("========================================")
    print("       VOICELENS AUDIO TEST             ")
    print("========================================")
    print("Searching for microphone...")

    try:
        with sr.Microphone() as source:
            print("Calibrating ambient noise... (stay quiet for 1s)")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("\nREADY! Speak clearly into your mic now...")
            print("----------------------------------------")

            audio = recognizer.listen(source, timeout=6, phrase_time_limit=10)
            print("Audio captured! Sending to speech engine...")
            
            text = recognizer.recognize_google(audio)
            print("\nSUCCESS! What VoiceLens heard:")
            print(f'>> "{text}"')
            print("========================================\n")
            
    except sr.WaitTimeoutError:
        print("\nNo speech detected (you did not speak within 6 seconds).")
    except sr.UnknownValueError:
        print("\nVoice was detected, but could not decipher the words.")
    except sr.RequestError as e:
        print(f"\nNetwork error requesting Google Speech results: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    test_microphone()
