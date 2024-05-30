from faster_whisper import WhisperModel  
import os  

# Set page title
# Load whisper model
model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2)) 

# Speech to text
def speech_to_text(audio_chunk): 
    print(audio_chunk)
    segments, info = model.transcribe(audio_chunk, beam_size=5) 
    speech_text = " ".join([segment.text for segment in segments]) 
    print(speech_text) 
   
text = speech_to_text('sumit_r.mp3') 

exit()
##################################################################
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

##########################################################################################
##########################################################################################
# import os
# import sys
# import sounddevice as sd
# import numpy as np
# from faster_whisper import WhisperModel

# # Load Whisper model
# model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=int(os.cpu_count() / 2))

# # Audio settings
# SAMPLE_RATE = 16000
# DURATION = 5  # Duration to record in seconds

# def speech_to_text(audio_data):
#     segments, info = model.transcribe(audio_data, beam_size=5)
#     speech_text = " ".join([segment.text for segment in segments])
#     print(speech_text)
#     return speech_text

# def callback(indata, frames, time, status):
#     if status:
#         print(status, file=sys.stderr)
#     # Convert the input data to numpy array
#     audio_data = np.frombuffer(indata, dtype=np.float32)
#     # Transcribe the audio data
#     speech_to_text(audio_data)

# # Open stream
# with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=callback):
#     print("Listening...")
#     try:
#         while True:
#             sd.sleep(int(DURATION * 1000))  # Sleep while the callback function processes the audio
#     except KeyboardInterrupt:
#         print("\nStopped listening.")

