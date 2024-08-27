from dataclasses import dataclass
from pathlib import Path
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import (
    AudioFileInterface,
    ColorImageInterface,
    TranscriptionInterface,
)


@dataclass
class _TranscriptionAndAudioInterface(BaseInterface):
    """
    Private interface used to make sure transcriptions and
    audio are synced during ablation testing
    """

    audio_file: Path
    speaker_id: str
    start_time: float
    end_time: float
    text: str


@final
class _TranscriptionAndAudioAblation(BaseFeature[_TranscriptionAndAudioInterface]):
    """
    Note: do not use this feature directly, create with
    `create_transcription_and_audio_ablation_features` instead

    TODO: docstring
    """

    def __init__(
        self, color: BaseFeature[ColorImageInterface], *, csv_path: Path
    ) -> None:
        super().__init__(color)
        self.csv_path = csv_path

    def initialize(self):
        # TODO
        pass

    def get_output(self, color):
        # TODO
        pass


@final
class TranscriptionAblation(BaseFeature[TranscriptionInterface]):
    """
    Note: do not use this feature directly, create with
    `create_transcription_and_audio_ablation_features` instead

    TODO: docstring
    """

    def __init__(
        self, transcription_and_audio_interface: _TranscriptionAndAudioAblation
    ) -> None:
        super().__init__(transcription_and_audio_interface)

    def get_output(
        self, ta: _TranscriptionAndAudioInterface
    ) -> TranscriptionInterface | None:
        if not ta.is_new():
            return None

        return TranscriptionInterface(
            speaker_id=ta.speaker_id,
            start_time=ta.start_time,
            end_time=ta.end_time,
            text=ta.text,
        )


@final
class AudioAblation(BaseFeature[AudioFileInterface]):
    """
    Note: do not use this feature directly, create with
    `create_transcription_and_audio_ablation_features` instead

    TODO: docstring
    """

    def __init__(
        self, transcription_and_audio_interface: _TranscriptionAndAudioAblation
    ) -> None:
        super().__init__(transcription_and_audio_interface)

    def get_output(
        self, ta: _TranscriptionAndAudioInterface
    ) -> AudioFileInterface | None:
        if not ta.is_new():
            return None

        return AudioFileInterface(
            speaker_id=ta.speaker_id,
            start_time=ta.start_time,
            end_time=ta.end_time,
            path=ta.audio_file,
        )


def create_transcription_and_audio_ablation_features(
    color: BaseFeature[ColorImageInterface], *, csv_path: Path
):
    """
    Create features for transcription and audio ablation.

    Arguments:
    `color` -- feature which return color frames, used for frame count
    `csv_path` -- path to WTD transcription csv file

    Returns:
    transcription -- transcription feature which returns TranscriptionInterface
    audio -- audio feature which returns AudioFileInterface
    """
    ta = _TranscriptionAndAudioAblation(color, csv_path=csv_path)
    transcription = TranscriptionAblation(ta)
    audio = AudioAblation(ta)

    return transcription, audio
