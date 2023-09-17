
import threading
import queue
import random
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from datetime import datetime
from pydub import AudioSegment
import os
import webrtcvad

model_size = "small"  # Specify the size of the Whisper model
init_sample_rate = 44100  # Sample rate of Macbook built-in microphone
init_chunk_duration = 60  # Duration of audio to be buffered in seconds

def sampling(percentage):
    return random.randint(0, 100) < percentage


class AudioProcessor:
  def __init__(self, chunk_duration, sample_rate):
      self.chunk_duration = chunk_duration
      self.sample_rate = sample_rate
      self.chunk_size = self.chunk_duration * self.sample_rate
      self.current_chunk = np.array([], dtype=np.float32)
      self.audio_queue = queue.Queue()
      # Calculate frame size needed for VAD based on frame duration and sample rate
      self.frame_duration = 0.01  # 10 ms
      self.frame_size = int(self.sample_rate * self.frame_duration)

  def callback(self, indata, frames, time, status):
    # Convert the audio data to mono and the appropriate sample width
    audio_data = np.frombuffer(indata, dtype=np.float32)
    # Append the incoming audio data to the current chunk
    self.current_chunk = np.append(self.current_chunk, indata)

    # Only print the statements 5% of the time
    if sampling(percentage=1):
        print("Waiting for audio chunk to hit: " + str(self.chunk_size))
        print("Currently audio chunk size: " + str(len(self.current_chunk)))

    # If the current chunk has reached the desired size
    if len(self.current_chunk) >= self.chunk_size:
        # Put the current chunk in the queue
        self.audio_queue.put(self.current_chunk)
        # And start a new chunk
        self.current_chunk = np.array([], dtype=np.float32)

  def process_audio(self):
      while True:
          # Wait until there's a chunk in the queue
          while self.audio_queue.empty():
              pass

          # Get the chunk from the queue
          audio_data = self.audio_queue.get()

          # Get the current date and time
          now = datetime.now()

          # Format the date and time
          timestamp = now.strftime("%Y%m%d_%H%M")

          # Create the output file name
          output_transcript_file = f"transcript_{timestamp}.txt"
          output_audio_file = f"audio_{timestamp}.mp3"

          # Save audio data to mp3 file
          audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=self.sample_rate, sample_width=audio_data.dtype.itemsize, channels=1)
          
          # If the output file already exists, append the new audio to it
          if os.path.exists(output_audio_file):
              existing_audio = AudioSegment.from_mp3(output_audio_file)
              combined_audio = existing_audio + audio_segment
              combined_audio.export(output_audio_file, format="mp3")
          else:
              audio_segment.export(output_audio_file, format="mp3")

# Usage:
processor = AudioProcessor(chunk_duration=init_chunk_duration, sample_rate=init_sample_rate)
processing_thread = threading.Thread(target=processor.process_audio)
processing_thread.start()

with sd.InputStream(samplerate=init_sample_rate, callback=processor.callback, blocksize=processor.frame_size):
    print("Recording started. Press Ctrl+C to stop the recording.")
    while True:
        pass