import sounddevice as sd
import queue
import threading
import os
from datetime import datetime
from pydub import AudioSegment


def prompt_user_for_input_device():
    devices = sd.query_devices()
    print("Available input devices:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']}")

    device_index = input(
        "Select input device by index (leave empty for default): ")
    return None if device_index == "" else int(device_index)


class AudioProcessor:
    def __init__(self, duration, sample_rate, device):
        self.duration = duration
        self.sample_rate = sample_rate
        self.chunk_size = self.duration * self.sample_rate
        self.audio_queue = queue.Queue()
        self.device = device

    def record_audio(self):
        while True:
            audio_data = sd.rec(int(self.duration * self.sample_rate),
                                samplerate=self.sample_rate, channels=1, device=self.device)
            sd.wait()  # Wait until recording is finished
            # Play the audio data before enqueuing
            print("Playing pre-enqueuing recording")
            sd.play(audio_data, self.sample_rate)
            sd.wait()  # Wait until audio playback is finished
            self.audio_queue.put(audio_data)

    def process_audio(self):
        while True:
            while self.audio_queue.empty():
                pass

            audio_data = self.audio_queue.get()

            # Play the audio data
            # print("Playing dequeued recording")
            # sd.play(audio_data, self.sample_rate)
            # sd.wait()  # Wait until audio playback is finished

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M")
            output_audio_file = f"audio_{timestamp}.wav"

            audio_segment = AudioSegment(audio_data.tobytes(
            ), frame_rate=self.sample_rate, sample_width=audio_data.dtype.itemsize, channels=1)

            if os.path.exists(output_audio_file):
                existing_audio = AudioSegment.from_wav(output_audio_file)
                combined_audio = existing_audio + audio_segment
                combined_audio.export(output_audio_file, format="wav")
            else:
                audio_segment.export(output_audio_file, format="wav")


# Usage:
duration = 10.0  # seconds
sample_rate = 48000  # Sample rate

user_selected_device = prompt_user_for_input_device()

processor = AudioProcessor(
    duration=duration, sample_rate=sample_rate, device=user_selected_device)
recording_thread = threading.Thread(target=processor.record_audio)
processing_thread = threading.Thread(target=processor.process_audio)
recording_thread.start()
processing_thread.start()

print("Recording started. Press Ctrl+C to stop the recording.")
while True:
    pass
