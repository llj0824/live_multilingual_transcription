import sounddevice as sd
import threading
import queue
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

# Run on CPU with INT8
model_size = "small"  # Specify the size of the Whisper model
whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

counter = 0
# Set the duration and sample rate
duration = 5.0  # seconds
fs = 44100  # Sample rate

# Create a queue to communicate between threads
q = queue.Queue()

# We need to record and process in the same thread b/c
# in seperate threads it's causing audio issues.
def record_process_thread(processQueue):
    while True:
        print("Recording...")
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        prev_recording = myrecording
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")

        # Signal that recording is done and processing can start
        processQueue.put('continue')

        # Save the recording to a file
        wav.write(f"recording_{counter}.wav", fs, myrecording)

        # Processing
        print("Starting processing.")
        transcribe_audio(audio_data=myrecording)


def transcribe_audio(audio_data):
    # Process the audio
    print("Processing transcription...")
    segments, info = whisper_model.transcribe(
        audio_data, language="zh", task="translate", beam_size=5)
    print("Detected language '%s' with probability %f" %
          (info.language, info.language_probability))

    # Get the current date and time
    now = datetime.now()

    # Format the date and time
    timestamp = now.strftime("%Y%m%d_%H%M")

    # Create the output file name
    output_transcript_file = f"transcript_{timestamp}.txt"
    output_audio_file = f"audio_{timestamp}.mp3"

    # Open the output file
    with open(output_transcript_file, "w") as f:
        for segment in segments:
            # # Write the transcription to the file
            # f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))

            # Also print the transcription
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    # # Save audio data to mp3 file
    # audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=fs, sample_width=audio_data.dtype.itemsize, channels=1)

    # # If the output file already exists, append the new audio to it
    # if os.path.exists(output_audio_file):
    #     existing_audio = AudioSegment.from_mp3(output_audio_file)
    #     combined_audio = existing_audio + audio_segment
    #     combined_audio.export(output_audio_file, format="mp3")
    # else:
    #     audio_segment.export(output_audio_file, format="mp3")


def main_process(queue):
    while True:
        # Wait for a signal to start processing_thread
        msg = q.get()
        process_thread = threading.Thread(
            target=record_process_thread, args=(queue,))
        process_thread.start()


# Signal that processing is done and recording can start again
q.put('start')
main_process(q)
