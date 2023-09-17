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
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from datetime import datetime

model_size = "small"  # Specify the size of the Whisper model
sample_rate = 48000  # Sample rate of Macbook built-in microphone

chunk_duration = 10  # Duration of audio to be buffered in seconds
chunk_size = chunk_duration * sample_rate
current_chunk = np.array([], dtype=np.int16)

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

        # # Get the audio data from the queue
        # audio_data = np.array([], dtype=np.int16)
        # while audio_queue.qsize() > 0:
        #     audio_data = np.append(audio_data, audio_queue.get())

        audio_data = np.empty((buffer_duration * sample_rate,), dtype=np.int16)
        for i in range(buffer_duration * sample_rate):
            audio_data[i] = audio_queue.get()

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
    global current_chunk
    # Append the incoming audio data to the current chunk
    current_chunk = np.append(current_chunk, indata)
    
    # If the current chunk has reached the desired size
    if len(current_chunk) >= chunk_size:
        # Put the current chunk in the queue
        audio_queue.put(current_chunk)
        # And start a new chunk
        current_chunk = np.array([], dtype=np.int16)

# Start the processing thread
processing_thread = threading.Thread(target=process_audio)
processing_thread.start()

# Start recording audio
with sd.InputStream(samplerate=sample_rate, callback=callback):
    print("Recording started. Press Ctrl+C to stop the recording.")
    while True:
        pass