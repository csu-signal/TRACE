from typing import final

import faster_whisper
import os

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import AudioFileInterface, TranscriptionInterface


@final
class WhisperTranscription(BaseFeature[TranscriptionInterface]):
    """
    Transcribe audio files using the Whisper model

    Input interface is `AudioFileInterface`.

    Output interface is `TranscriptionInterface`.
    """

    def __init__(self, audio: BaseFeature[AudioFileInterface]):
        super().__init__(audio)

    def initialize(self):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.model = faster_whisper.WhisperModel("small")

    def get_output(
        self,
        audio: AudioFileInterface,
    ):
        if not audio.is_new():
            return None

        segments, info = self.model.transcribe(str(audio.path), language="en")
        transcription = " ".join(
            segment.text for segment in segments if segment.no_speech_prob < 0.5
        )  # Join segments into a single string

        return TranscriptionInterface(
            speaker_id=audio.speaker_id,
            start_time=audio.start_time,
            end_time=audio.end_time,
            text=transcription,
        )
