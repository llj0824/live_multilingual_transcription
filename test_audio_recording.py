import sounddevice as sd
import time

# Set the duration and sample rate
duration = 10.0  # seconds
fs = 48000  # Sample rate

# Record audio
print("Recording...")
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
print("Recording finished.")

# Play back the recording
print("Playing back...")
sd.play(myrecording, fs)
sd.wait()  # Wait until file is done playing
print("Playback finished.")