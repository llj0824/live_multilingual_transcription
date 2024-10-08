# Whisper Transcription Application

## Overview

This application is designed to transcribe real-time Chinese conversations into English captions using a speech recognition model. It captures audio from the environment, processes it in chunks, and outputs the transcriptions.

## Demo Videos

To see the application in action, you can view the following demo videos located in the `demo` directory:

- **Full View Demo:** [live_translation_full_view.mp4](demo/live_translation_full_view.mp4)
- **Screen View Demo:** [live_translation_screenview.mp4](demo/live_translation_screenview.mp4)

## Background

The application listens for conversations and transcribes them into English captions. It is primarily intended to translate Chinese conversations into English.

## TODO

1. Make the application continuously running.
2. Record audio nearby (maybe into chunks) and then print the transcript.

## Setup Guide

### 3.1. Create a New Virtual Environment

It's best to create a fresh virtual environment to avoid conflicts.

- **Navigate to Your Project Directory:**
  ```bash
  cd /path/to/your/project
  ```

- **Create a Virtual Environment:**
  ```bash
  python -m venv venv
  ```
  This command creates a new virtual environment named `venv` in your project directory.

### 3.2. Activate the Virtual Environment

Activate the newly created virtual environment to ensure all package installations are contained within it.

- **For Bash:**
  ```bash
  source venv/bin/activate
  ```

You should see `(venv)` prefixed in your terminal prompt, indicating that the virtual environment is active.

### 3.3. Upgrade pip Within the Virtual Environment

Before installing packages, ensure that pip is up-to-date.

`ffmpeg -i input.mp4 -vn -ab 128k -ar 44100 -y output.mp3`
