import sounddevice as sd
import librosa
import torch
from transformers import AutoProcessor, AutoModelForSeq2SeqLM
from queue import Queue

# Constants
AUDIO_SAMPLE_RATE = 16000  # Match the expected input sample rate of the model
CHUNK_DURATION = 5         # Duration of each audio chunk in seconds
LANG_PROMPT = "<|audio_bos|><|AUDIO|><|audio_eos|>请把音频翻译成简体中文:"  # Prompt in Chinese to return Simplified Chinese text

# Initialize the processor and model using correct classes
model_name = "Qwen/Qwen2-Audio-7B"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

# A queue to hold audio data chunks for processing
audio_queue = Queue()

# Function to process audio and generate transcription
def process_and_transcribe():
    while True:
        audio_chunk = audio_queue.get()  # Retrieve audio chunks from the queue
        
        # Skip processing if the chunk is empty or None
        if audio_chunk is None:
            break
        
        # Process audio (resample and convert if necessary using librosa)
        audio = librosa.resample(audio_chunk, orig_sr=AUDIO_SAMPLE_RATE, target_sr=processor.feature_extractor.sampling_rate)

        # Prepare inputs for the model (text prompt + audio)
        inputs = processor(text=LANG_PROMPT, audios=audio, return_tensors="pt")
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=256)

        # Truncate the prompt part and extract generated Chinese text
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Output the transcription in Simplified Chinese
        print("Simplified Chinese transcription:", response)

# Function for audio callback to capture live input from the microphone
def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    
    print("audio processing callback...")
    # Send captured audio to the queue for processing
    audio_queue.put(indata.copy())

# Start microphone input stream and processing loop
def main():
    # Open an input stream from the microphone with the desired sample rate
    with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, callback=callback, blocksize=int(AUDIO_SAMPLE_RATE * CHUNK_DURATION)):
        # Continuously process audio and transcribe in real-time
        process_and_transcribe()

if __name__ == "__main__":
    main()