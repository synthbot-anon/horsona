
import asyncio
import sys

from .audio_provider import Microphone
from .audio_provider import AudioFile
from .whisperstt import WhisperSTT


WHISPER_ENDPOINT = "http://127.0.0.1:8000"
WHISPER_WS_ENDPOINT = "ws://localhost:8000"
WHISPER_MODEL = "Systran/faster-whisper-tiny"
# WHISPER_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"


async def mic_main():
    if len(sys.argv) > 2:
        mic_name = sys.argv[2]
    else:
        raise AttributeError("No microphone name was provided in the script argument")

    audio_provider = Microphone(mic_name)

    whisper = WhisperSTT(WHISPER_ENDPOINT)

    audio_stream = audio_provider.get_audio_stream()
    transcription_stream = whisper.transcribe_audio_stream(audio_stream, base_url=WHISPER_WS_ENDPOINT)

    print("Transcription result: ")
    async for text_chunk in transcription_stream:
        print(text_chunk)


async def file_main():
    if len(sys.argv) > 2:
        file_name = sys.argv[2]
    else:
        raise AttributeError("No file path was provided in the script argument")
    
    audio_provider = AudioFile(file_name)

    whisper = WhisperSTT(WHISPER_ENDPOINT)

    audio_clip = audio_provider.get_audio_clip()
    transcription_result = await whisper.transcribe_audio_clip(audio_clip)

    print(f"Transcription result: {transcription_result.text}")


async def main():
    if len(sys.argv) > 1:
        script_mode = sys.argv[1]
    else:
        raise AttributeError("No source selection provided. Usage: python -m audio [mic|file] [mic name|file path]")

    if script_mode.lower() == "mic":
        await mic_main()
    elif script_mode.lower() == "file":
        await file_main()
    else:
        raise AttributeError("No source selection provided. Usage: python -m audio [mic|file] [mic name|file path]")

asyncio.run(main())
