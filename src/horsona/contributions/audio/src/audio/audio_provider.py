from typing import Generator
from typing import Optional
from typing import List
from abc import ABC, abstractmethod

from dataclasses import dataclass
from queue import Queue, Empty
import types

import pyaudio
import audioop


@dataclass
class AudioChunk:
    data: bytes

    # Both can be detected on the server side
    sample_rate: Optional[int] = None
    sample_width: Optional[int] = None


@dataclass
class AudioClip(AudioChunk):
    ...

AudioStream = Generator[AudioChunk, None, None]


class AudioProvider(ABC):

    @abstractmethod
    def get_audio_stream(self) -> AudioStream: ...

    @abstractmethod
    def get_audio_clip(self) -> AudioClip: ...


class Microphone(AudioProvider):
    DEFAULT_RATE = 16000

    def __init__(self, device_name: str, chunk_size: int = 1024, queue_timeout: int = 10):
        self.device_name = device_name
        self.chunk_size = chunk_size
        self.audio_interface = pyaudio.PyAudio()

        self._cvstate = None
        self.buffer_queue = Queue(maxsize=chunk_size)
        self.queue_timeout = queue_timeout

        info = self.audio_interface.get_host_api_info_by_index(0)
        device_count = info.get('deviceCount')

        device_index = None
        for i in range(device_count):
            name = self.audio_interface.get_device_info_by_host_api_device_index(0, i).get('name')
            if self.device_name in name:
                device_index = i
                break

        if not device_index:
            raise RuntimeError(f"Cannot find device with the name: {device_name}")

        self.device_info = self.audio_interface.get_device_info_by_index(device_index)

        grab_sample_callback = types.MethodType(Microphone._grab_sample_callback, self)
        stream = self.audio_interface.open(format=pyaudio.paInt16,
            channels=self.device_info["maxInputChannels"],
            rate=int(self.device_info["defaultSampleRate"]),
            input=True,
            stream_callback=grab_sample_callback,
            frames_per_buffer=self.chunk_size,
            input_device_index=device_index
        )

    def _grab_sample_callback(self, input_data, frame_count, time_info, flags, format = pyaudio.paInt16):
        sample_width = pyaudio.get_sample_size(format)
        converted_data, self._cvstate = audioop.ratecv(
            input_data, sample_width, self.device_info["maxInputChannels"], int(self.DEFAULT_RATE),
            int(self.device_info["defaultSampleRate"]), self._cvstate
        )

        recorded_chunk = AudioChunk(
            data=converted_data,
            sample_rate=self.DEFAULT_RATE,
            sample_width=sample_width
        )

        self.buffer_queue.put(recorded_chunk)

        return input_data, pyaudio.paContinue

    def get_audio_stream(self) -> AudioStream:
        """ 
        Sending data as it becomes available in the buffer queue
        
        Yields:
            Audio chunk, containing binary data with accompanying meta information.
        """
        while True:
            try:
                yield self.buffer_queue.get(block=True)#, timeout=self.queue_timeout)
            except Empty:
                return

    def get_audio_clip(self) -> AudioClip:
        raise NotImplementedError("Unsupported method for getting data from Microphone provider")


class AudioFile(AudioProvider):
    def __init__(self, file_path: str):
        self.file_path = file_path

        with open(file_path, 'rb') as f:
            audio_buffer = f.read()

        self.clip = AudioClip(data=audio_buffer)

    def get_audio_clip(self) -> AudioClip:
        """ 
        Will send binary data of audio clip to server as is,
        without any format parsing and conversion
        
        Returns:
            Complete audio clip, containing binary data with accompanying meta information.
        """
        return self.clip

    def get_audio_stream(self) -> AudioClip:
        raise NotImplementedError("Unsupported method for getting data from AudioFile provider")


class WebSocket(AudioProvider):
    def __init__(self, port: int):
        self.port = port

    def get_audio_stream(self) -> Generator[AudioChunk, None, None]:
        raise NotImplementedError("Not yet implemented")

    def get_audio_clip(self) -> AudioClip:
        raise NotImplementedError("Unsupported method for getting data from WebSocket provider")
