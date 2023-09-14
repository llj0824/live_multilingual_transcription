from faster_whisper import WhisperModel
from datetime import datetime

model_size = "large"
audio_file = "output.mp3"

# Run on CPU with FP32
# model = WhisperModel(model_size, device="cpu", compute_type="float32")

# or run on CPU with INT8
print("Initating Whisper Model...")
model = WhisperModel(model_size, device="cpu", compute_type="int8")

print("Begining Whisper Transcription...")
segments, info = model.transcribe(audio_file, language="zh", task="translate", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# Get the current date and time
now = datetime.now()

# Format the date and time
timestamp = now.strftime("%Y%m%d_%H:%M")

# Create the output file name
output_file = f"{audio_file}_{timestamp}.txt"

# Open the output file
with open(output_file, "w") as f:
    for segment in segments:
        # Write the transcription to the file
        f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))

        # Also print the transcription
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))