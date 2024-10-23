import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr

class AudioRecorder:
    def __init__(self, samplerate=44100, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self.recording = False
        self.audio_data = None

    def start_recording(self):
        """Starts the recording."""
        self.recording = True
        self.audio_data = sd.rec(int(self.samplerate * 10),  # Record for 10 seconds
                                 samplerate=self.samplerate,
                                 channels=self.channels,
                                 dtype='float64')
        print("Recording... Please speak.")

    def stop_recording(self):
        """Stops the recording and returns the audio data."""
        self.recording = False
        sd.stop()  # Stop the recording
        print("Recording stopped.")

        # Normalize the audio data to the range -1 to 1
        self.audio_data = np.array(self.audio_data, dtype='float64')
        return self.audio_data

    def save_audio(self, filename):
        """Saves the recorded audio to a WAV file."""
        if self.audio_data is not None:
            # Scale the audio data to int16
            audio_int16 = (self.audio_data * 32767).astype(np.int16)
            wav.write(filename, self.samplerate, audio_int16)  # Save as WAV file
            print(f"Audio saved to {filename}")

    def get_audio_as_data(self):
        """Returns the recorded audio as speech_recognition.AudioData."""
        if self.audio_data is not None:
            # Convert numpy array to bytes
            audio_bytes = (self.audio_data * 32767).astype(np.int16).tobytes()
            return sr.AudioData(audio_bytes, self.samplerate, 2)  # 2 bytes for int16
        return None

# Example usage
if __name__ == "__main__":
    recorder = AudioRecorder()

    # Start recording
    recorder.start_recording()

    # Wait for user input to stop recording
    input("Press Enter to stop recording...\n")

    # Stop recording
    recorder.stop_recording()

    # Save the recorded audio
    recorder.save_audio("recorded_audio.wav")

    # Retrieve the audio data as AudioData object
    audio_data = recorder.get_audio_as_data()

    if audio_data:
        print("Audio data recorded successfully.")
    else:
        print("No audio data was recorded.")
    recognizer = sr.Recognizer()
    text = recognizer.recognize_google(audio_data, language="ar-EG")
    print(text)