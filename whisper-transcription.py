from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import threading
import queue
from datetime import datetime
import webrtcvad
import collections


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

# Initialize VAD
vad = webrtcvad.Vad(2)  # Aggressiveness mode (0-3)

# Frame duration for VAD (must be 10, 20, or 30 ms)
frame_duration = 30  # ms

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

# Queue to communicate between the audio callback and processing thread
audio_queue = queue.Queue()
# Event to signal the processing thread to stop
stop_event = threading.Event()

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * frame_duration_ms / 1000)
    offset = 0
    while offset + n <= len(audio):
        yield audio[offset:offset + n]
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []

    for frame in frames:
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer if vad.is_speech(f.tobytes(), sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer if not vad.is_speech(f.tobytes(), sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join(voiced_frames)

def callback(indata, frames, time, status):
    if status:
        print(status)
    # Convert audio data to float32 and enqueue
    audio_queue.put(indata.copy())

def audio_processor():
    while not stop_event.is_set():
        try:
            # Accumulate audio data from the queue
            data = audio_queue.get(timeout=1)
            # Convert to 16-bit PCM for VAD
            audio = (data.flatten() * 32767).astype(np.int16)
            frames = list(frame_generator(frame_duration, audio, sample_rate))
            segments = vad_collector(sample_rate, frame_duration, 300, vad, frames)
            for segment in segments:
                audio_chunk = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32767.0
                if len(audio_chunk) == 0:
                    continue
                # Transcribe the audio chunk in Chinese
                transcription = asr_pipe(audio_chunk)['text'].strip()
                
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
        print("Real-time transcription running with VAD... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)  # Keep the main thread alive
except KeyboardInterrupt:
    print("\nTranscription stopped.")
finally:
    # Signal the processing thread to stop and wait for it to finish
    stop_event.set()
    processor_thread.join()
