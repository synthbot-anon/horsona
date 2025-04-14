
# Audio Input module

Module that provides capabilities to get audio stream from various sources and hook it to the LLM input and error signals

## Setup Instructions

### Setting up and running `faster_whisper_server`

```bash
docker run --gpus=all --publish 8000:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface fedirz/faster-whisper-server:latest-cuda
# or
docker run --publish 8000:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface fedirz/faster-whisper-server:latest-cpu
```

### Setting up the Repository

   Install dependencies:
   ```bash
   poetry install
   ```

### Running samples

   For test audio file transcription (better to use .wav, .pcm for now):
   ```bash
   poetry run python -m audio file <file_to_transcribe>
   ```
   For microphone streaming (doesn't work completely at the moment):
   ```bash
   poetry run python -m audio mic <mic_device_name>
   ```