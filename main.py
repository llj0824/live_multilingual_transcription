import threading
import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from datetime import datetime

model_size = "large"
# duration of audio to be buffered in seconds
# two buffers - one for recording dialogue, another for processing.
buffer_duration = 10  
sample_rate = 100  # sample rate in Hz

# Run on CPU with INT8
print("Initiating Whisper Model...")
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Create a queue to hold the incoming audio data
audio_queue = queue.Queue()

# This function will be run in a separate thread to process the audio
def process_audio():
    while True:
        # Wait until there's enough audio data in the queue
        while audio_queue.qsize() < buffer_duration * sample_rate:
            print("Waiting for more. Queue size: " + str(audio_queue.qsize()))
            pass

        # Get the audio data from the queue
        audio_data = np.array([], dtype=np.int16)
        while audio_queue.qsize() > 0:
            audio_data = np.append(audio_data, audio_queue.get())

        # Process the audio
        print("Processing transcription...")
        segments, info = model.transcribe(audio_data, language="zh", task="translate", beam_size=5)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        # Get the current date and time
        now = datetime.now()

        # Format the date and time
        timestamp = now.strftime("%Y%m%d_%H")

        # Create the output file name
        output_file = f"audio_{timestamp}.txt"

        # Open the output file
        with open(output_file, "w") as f:
            for segment in segments:
                # Write the transcription to the file
                f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))

                # Also print the transcription
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# This function will be called for each chunk of audio data
def callback(indata, frames, time, status):
    # Put the incoming audio data in the queue
    # Flatten the incoming audio data
    audio_queue.put(indata.flatten())

# Start the processing thread
processing_thread = threading.Thread(target=process_audio)
processing_thread.start()

# Start recording audio
with sd.InputStream(callback=callback):
    print("Recording started. Press Ctrl+C to stop the recording.")
    while True:
        pass