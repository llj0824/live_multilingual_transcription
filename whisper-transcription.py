from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import threading
import queue
from datetime import datetime
from pyannote.audio import Pipeline


# Model configuration
model_id = "openai/whisper-large-v3-turbo"
device = "cpu"

# Load model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
# Real-time audio settings
sample_rate = 16000
block_duration = 5  # seconds
language = "zh"     # Set the language for transcription


# Ensure this line is correctly indented according to the surrounding code
generate_kwargs = {
    "language": language,
    # TODO see if i can pass in initial prompt into generate_kwargs for AutomaticSpeechRecognitionPipeline using whisper model.
    ## And then use it to do [chinese transcription] followed by [english translation] two lines for each chunk.
}


# Initialize the translation pipeline
translation_pipe = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en",  # Chinese to English translation model
    device=device
)

# Initialize the automatic-speech-recognition pipeline
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    chunk_length_s=30,  # Adjust chunk length as needed
    batch_size=16,       # Adjust batch size based on device capability
    generate_kwargs=generate_kwargs
)

# Initialize the VAD pipeline
vad_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=config['access_token']
)

# Queue to communicate between the audio callback and processing thread
audio_queue = queue.Queue()
# Event to signal the processing thread to stop
stop_event = threading.Event()

def callback(indata, frames, time, status):
    if status:
        print(status)
    # Convert audio data to float32 and enqueue
    audio_queue.put(indata.copy())

def audio_processor():
    audio_buffer = np.empty((0,), dtype=np.float32)
    while not stop_event.is_set():
        try:
            # Accumulate audio data from the queue
            data = audio_queue.get(timeout=1)
            audio_buffer = np.concatenate((audio_buffer, data.flatten()), axis=0)
            # Process when we have at least block_duration seconds of audio
            if len(audio_buffer) >= sample_rate * block_duration:
                # Extract a chunk of audio data
                audio_chunk = audio_buffer[:sample_rate * block_duration]
                # Remove the processed chunk from the buffer
                audio_buffer = audio_buffer[sample_rate * block_duration:]
                
                # Save the audio chunk to a temporary file
                temp_audio_file = "temp_audio.wav"
                sd.write(temp_audio_file, audio_chunk, sample_rate)

                # Apply VAD to the audio chunk
                vad_output = vad_pipeline(temp_audio_file)
                
                for speech in vad_output.get_timeline().support():
                    # Extract active speech segments
                    start_time = int(speech.start * sample_rate)
                    end_time = int(speech.end * sample_rate)
                    active_speech = audio_chunk[start_time:end_time]

                    # Transcribe the active speech in Chinese
                    transcription = asr_pipe(active_speech)['text'].strip()
                    
                    # Translate the transcription to English
                    translation = translation_pipe(transcription)[0]['translation_text']

                    # Get the current timestamp
                    current_time = datetime.now().strftime("%H:%M:%S")
                    
                    # Prepare the output with timestamp prefix
                    timestamped_output = (
                        f"[{current_time}] {transcription}\n"
                        f"[{current_time}] {translation}\n"
                    )
                    print(f"{timestamped_output}\n")
        except queue.Empty:
            continue  # No data received yet

# Start the audio processing thread
processor_thread = threading.Thread(target=audio_processor)
processor_thread.start()

try:
    # Start the audio input stream
    with sd.InputStream(channels=1, samplerate=sample_rate, callback=callback, dtype='float32'):
        print("Real-time transcription running... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)  # Keep the main thread alive
except KeyboardInterrupt:
    print("\nTranscription stopped.")
finally:
    # Signal the processing thread to stop and wait for it to finish
    stop_event.set()
    processor_thread.join()
