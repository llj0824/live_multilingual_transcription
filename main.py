"""
This script uses the Whisper ASR (Automatic Speech Recognition) model to transcribe audio in real-time. 
It records audio from the default input device, processes it in a separate thread, and writes the transcriptions to a text file.

The script uses a queue to buffer the incoming audio data. The recording thread puts incoming audio data into the queue, 
and the processing thread takes data from the queue and processes it.

The Whisper ASR model is initialized with a specific model size and is set to run on the CPU with INT8 precision.

The script also prints out the detected language and its probability.

The transcriptions are written to a text file with a timestamp in its name.
"""
import threading
import queue
import random
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from datetime import datetime
from pydub import AudioSegment
import os

model_size = "small"  # Specify the size of the Whisper model
init_sample_rate = 48000  # Sample rate of Macbook built-in microphone
init_chunk_duration = 5  # Duration of audio to be buffered in seconds

def sampling(percentage):
    return random.randint(0, 100) < percentage


class AudioProcessor:
  def __init__(self, chunk_duration, sample_rate):
      self.chunk_duration = chunk_duration
      self.sample_rate = sample_rate
      self.chunk_size = self.chunk_duration * self.sample_rate
      self.current_chunk = np.array([], dtype=np.int16)
      self.audio_queue = queue.Queue()
      # Run on CPU with INT8
      self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

  def callback(self, indata, frames, time, status):
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
          self.current_chunk = np.array([], dtype=np.int16)

  def process_audio(self):
      while True:
          # Wait until there's a chunk in the queue
          while self.audio_queue.empty():
              pass

          # Get the chunk from the queue
          audio_data = self.audio_queue.get()

          # Process the audio
          print("Processing transcription...")
          segments, info = self.model.transcribe(audio_data, language="zh", task="translate", beam_size=5)
          print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

          # Get the current date and time
          now = datetime.now()

          # Format the date and time
          timestamp = now.strftime("%Y%m%d_%H")

          # Create the output file name
          output_transcript_file = f"transcript_{timestamp}.txt"
          output_audio_file = f"audio_{timestamp}.mp3"

          # Open the output file
          with open(output_transcript_file, "w") as f:
              for segment in segments:
                  # Write the transcription to the file
                  f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))

                  # Also print the transcription
                  print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

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

with sd.InputStream(samplerate=init_sample_rate, callback=processor.callback):
    print("Recording started. Press Ctrl+C to stop the recording.")
    while True:
        pass