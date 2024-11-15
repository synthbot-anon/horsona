import asyncio
from collections import defaultdict
from typing import Any, Dict, Optional

import httpx

from horsona.lock.resource_state_lock import ResourceStateLock


class SpeakerLock(ResourceStateLock):
    async def set_state(self, endpoint: str, speaker: str):
        """
        Set the TTS model and reference audio for a given speaker.

        This method ensures that only one model loading operation happens at a time
        by using an event lock (_speaker_model_set). If a model is currently being
        loaded, subsequent calls will wait for it to complete.

        Args:
            endpoint (str): Base URL of the TTS API endpoint
            speaker (str): Name of the speaker to load models for

        The method loads:
        - GPT model from /voices/{speaker}/gpt.ckpt
        - SoVITS model from /voices/{speaker}/sovits.pth
        - Reference audio from /voices/{speaker}/reference.flac
        - Reference transcript from /voices/{speaker}/reference-transcript.txt
        """

        # Set the model
        await asyncio.gather(
            self._set_remote_model(
                endpoint, f"/voices/{speaker}/gpt.ckpt", f"/voices/{speaker}/sovits.pth"
            ),
            self._change_reference(
                endpoint,
                f"/voices/{speaker}/reference.flac",
                f"/voices/{speaker}/reference-transcript.txt",
            ),
        )

    async def _set_remote_model(
        self, base_url: str, gpt_model_path: str, sovits_model_path: str
    ) -> Dict[str, Any]:
        """
        Asynchronously set the GPT and SoVITS model paths for the TTS service.

        Args:
            base_url (str): Base URL of the API
            gpt_model_path (str): Path to the GPT model checkpoint
            sovits_model_path (str): Path to the SoVITS model

        Returns:
            dict: JSON response from the API
        """
        endpoint = f"{base_url}/set_model"
        params = {
            "gpt_model_path": gpt_model_path,
            "sovits_model_path": sovits_model_path,
        }

        headers = {"accept": "application/json"}

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            return response.json()

    async def _change_reference(
        self,
        base_url: str,
        refer_wav_path: str,
        prompt_text: str,
        prompt_language: str = "en",
    ) -> Dict[str, Any]:
        """
        Asynchronously change the reference audio and prompt text for the TTS service.

        Args:
            base_url (str): Base URL of the API
            refer_wav_path (str): Path to the reference audio file
            prompt_text (str): Path to the prompt text file or the prompt text itself
            prompt_language (str): Language of the prompt

        Returns:
            dict: JSON response from the API
        """
        endpoint = f"{base_url}/change_refer"

        params = {
            "refer_wav_path": refer_wav_path,
            "prompt_text": prompt_text,
            "prompt_language": prompt_language,
        }

        headers = {"accept": "application/json"}

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            return response.json()


class GptSovitsTTS:
    _speaker_locks: dict[str, SpeakerLock] = defaultdict(SpeakerLock)

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def generate_speech(self, speaker: str, text: str) -> bytes:
        lock = GptSovitsTTS._speaker_locks[self.endpoint]

        async with lock.acquire(self.endpoint, speaker):
            return await self._generate_speech(text)

    async def _generate_speech(
        self,
        text: str,
        base_url: Optional[str] = None,
        refer_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        prompt_language: str = "en",
        text_language: str = "en",
        cut_punc: str = ".",
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Asynchronously generate speech using the TTS service.

        Args:
            text (str): The text to convert to speech
            base_url (str): Base URL of the API
            refer_wav_path (str): Path to the reference audio file
            prompt_text (str): Optional prompt text path
            prompt_language (str): Language of the prompt
            text_language (str): Language of the input text
            cut_punc (str): Punctuation to use for cutting
            top_k (int): Top K parameter for generation
            top_p (float): Top P parameter for generation
            temperature (float): Temperature parameter for generation
            speed (float): Speech speed multiplier

        Returns:
            dict: JSON response from the API
        """
        endpoint = base_url or self.endpoint

        params = {
            "refer_wav_path": refer_wav_path,
            "prompt_text": prompt_text,
            "prompt_language": prompt_language,
            "text": text,
            "text_language": text_language,
            "cut_punc": cut_punc,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "speed": speed,
        }

        headers = {"accept": "application/json"}

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.get(endpoint, params=params, headers=headers)
            response.raise_for_status()

        # Return the audio data as bytes
        return response.content
