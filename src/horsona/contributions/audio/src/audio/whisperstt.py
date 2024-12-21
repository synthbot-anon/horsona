import warnings
from operator import itemgetter

import json
import httpx
import websockets
from openai import AsyncOpenAI
from typing import Generator
from typing import Optional
from typing import Dict
from typing import Any

from .audio_provider import AudioProvider
from .audio_provider import AudioClip, AudioChunk
from .audio_provider import AudioStream

from websockets.asyncio.client import connect


class WhisperSTT:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        url = WhisperSTT._format_link_path(self.endpoint)
        self.oai_client = AsyncOpenAI(api_key="---", base_url=url)

    @staticmethod
    def _format_link_path(base_url: str) -> str:
        return f"{base_url}/v1"

    async def transcribe_audio_clip(
        self,
        audio: AudioClip | AudioChunk,
        base_url: Optional[str] = None,
        model: str = "Systran/faster-whisper-tiny",
        language: str = "en",
        response_format: str = "verbose_json",
        temperature: float = 1.0,
    ):
        """
        Transcribing whole block of audio through whisper api.
        
        Args:
            audio (AudioClip | AudioChunk): Audio data with accompanying meta information.
            base_url (str): Url that can override assigned endpoint. Default: None
            model (str): Whisper model name to perform transcription. Defalut: "Systran/faster-whisper-tiny"
            language (str): Language of input speech. Defalut: "en"
            response_format (str): Format used to output transcription results. Defalut: "verbose_json"
            temperature (float): Temperature for sampling. Defalut: 1.0
        
        Returns:
            response: Transcription response from the model server in requested format.
        """

        new_endpoint = base_url or None

        if new_endpoint:
            self.oai_client.base_url = https.URL(new_endpoint)

        params = {
            "language": language,
            "response_format": response_format,
            # "vad_filter": False, # @Incomplete
            "temperature": temperature,
        }

        if audio.sample_rate and audio.sample_rate != 16000:
            warnings.warn("Sample rate of input clip is not equal to 16kHz. Whisper will output garbage!")

        response = await self.oai_client.audio.transcriptions.create(
            model=model,
            file=audio.data,
            **params
        )

        return response

    @staticmethod
    def _format_params_as_get_args(params: Dict[str, Any]) -> str:
        model, language, response_format, temperature = itemgetter("model", "language", "response_format", "temperature")(params)
        return f"model={model}&language={language}&response_format={response_format}&temperature={temperature}"

    @staticmethod
    def _format_ws_link_path_with_args(base_url: str, params: Dict[str, Any]) -> str:
        get_args = WhisperSTT._format_params_as_get_args(params)
        return f"{base_url}/v1/audio/transcriptions?{get_args}"

    async def transcribe_audio_stream(
        self,
        audio: AudioStream,
        base_url: Optional[str] = None,
        model: str = "Systran/faster-whisper-tiny",
        language: str = "en",
        response_format: str = "verbose_json",
        temperature: float = 1.0,
    ):
        """
        Transcribing audio stream through whisper api.
        
        Args:
            audio (AudioStream): Audio data generator that yields samples on each iteration.
            base_url (str): Url that can override assigned endpoint. Default: None
            model (str): Whisper model name to perform transcription. Defalut: "Systran/faster-whisper-tiny"
            language (str): Language of input speech. Defalut: "en"
            response_format (str): Format used to output transcription results. Defalut: "verbose_json"
            temperature (float): Temperature for sampling. Defalut: 1.0
        
        Yields:
            response: Transcription response chunks from the model server in requested format.
        """

        endpoint = base_url or self.endpoint

        params = {
            "model": model,
            "language": language,
            "response_format": response_format,
            "temperature": temperature,
        }

        url = WhisperSTT._format_ws_link_path_with_args(endpoint, params)
        for audio_chunk in audio:
            if audio_chunk.sample_rate and audio_chunk.sample_rate != 16000:
                warnings.warn("Sample rate of input clip is not equal to 16kHz. Whisper will output garbage!")

            # @Incomplete: Maintain connection as long as stream is going
            async with connect(url) as ws_client:
                ws_client.send(audio_chunk.data)
                response = await ws_client.recv()

                yield response
