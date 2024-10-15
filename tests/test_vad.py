from pyannote.audio import Pipeline
import json

# Load configuration from JSON file
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

try:
    vad_pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=config['access_token']
    )
    print("VAD Pipeline loaded successfully.")
except Exception as e:
    print(f"Failed to load VAD Pipeline: {e}")
