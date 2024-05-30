import os
import pyaudio
import wave
from faster_whisper import WhisperModel

# Load Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2))

# Audio settings
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Number of audio channels
RATE = 16000              # Sampling rate
CHUNK = 1024              # Buffer size
WAVE_OUTPUT_FILENAME = "output.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Listening...")

try:
    while True:
        frames = []
        
        # Record audio in chunks
        for i in range(0, int(RATE / CHUNK * 3)):  # Record for 5 seconds
            data = stream.read(CHUNK)
            frames.append(data)
        
        # Save the recorded data as a .wav file
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Transcribe the audio file
        segments, info = model.transcribe(WAVE_OUTPUT_FILENAME, beam_size=3)
        speech_text = " ".join([segment.text for segment in segments])
        print(speech_text)

except KeyboardInterrupt:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("\nStopped listening.")


exit()
import os
import pyaudio
import wave
from faster_whisper import WhisperModel
import numpy as np

# Load Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2))

# Audio settings
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Number of audio channels
RATE = 16000              # Sampling rate
CHUNK = 1024              # Buffer size

def speech_to_text(audio_data):
    # print(audio_data)
    segments, info = model.transcribe(audio_data, beam_size=3)
    speech_text = " ".join([segment.text for segment in segments])
    print(speech_text)
    return speech_text

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Listening...")

try:
    while True:
        frames = []
        
        # Record audio in chunks
        for i in range(0, int(RATE / CHUNK * 3)):  # Record for 5 seconds
            data = stream.read(CHUNK)
            frames.append(data)
        
        # Convert audio chunks to a single audio data
        audio_data = b''.join(frames)
        
        # Convert to numpy array for Whisper
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe audio
        speech_to_text(audio_np)

except KeyboardInterrupt:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("\nStopped listening.")

