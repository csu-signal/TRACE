import csv
import time
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
from mmdemo.utils.frame_time_converter import FrameTimeConverter


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
class _TranscriptionAndAudioGroundTruth(BaseFeature[_TranscriptionAndAudioInterface]):
    """
    Note: do not use this feature directly, create with
    `create_transcription_and_audio_ground_truth_features` instead

    Helper feature for syncing ground truth audio and transcriptions. Creates a queue
    of utterances to send, so if they are more frequent than the frames the audio will
    lag behind the video. This should not happen in almost all cases though. The audio
    times/frames are converted using a FrameTimeConverter.

    Input interface is `ColorImageInterface`. Output interface is `_TranscriptionAndAudioInterface`.

    Keyword arguments:
    `csv_path` -- path to the WTD annotation asrOutput csv file. The audio paths in this
    csv should be relative to the parent directory of this file.
    """

    def __init__(
        self, color: BaseFeature[ColorImageInterface], *, csv_path: Path
    ) -> None:
        super().__init__(color)
        self.csv_path = csv_path

    def initialize(self):
        self.data = _TranscriptionAndAudioGroundTruth.read_csv(self.csv_path)
        self.current_index = 0

        # color frame to demo time lookup
        self.frame_time_converter = FrameTimeConverter()

    def get_output(
        self, color: ColorImageInterface
    ) -> _TranscriptionAndAudioInterface | None:
        if not color.is_new():
            return None

        self.frame_time_converter.add_data(color.frame_count, time.time())

        # no utterance if there is no more data
        if self.current_index >= len(self.data):
            return None

        # get frame and row data
        frame, row_data = self.data[self.current_index]

        # no utterance if we are past the color frame
        if frame > color.frame_count:
            return None

        # we have a valid utterance to return, so increment
        # the current index and return the utterance
        self.current_index += 1
        start_time = self.frame_time_converter.get_time(row_data["start_frame"])
        stop_time = self.frame_time_converter.get_time(row_data["stop_frame"])

        return _TranscriptionAndAudioInterface(
            audio_file=Path(row_data["audio_file"]),
            speaker_id=row_data["speaker_id"],
            start_time=start_time,
            end_time=stop_time,
            text=row_data["text"],
        )

    @staticmethod
    def read_csv(path) -> list[tuple[int, dict[str, str]]]:
        """
        Returns a list[tuple[int, dict[str, str]]] mapping frame counts to
        the csv row as a dict. This is a list because multiple utterances
        could be received on the same frame.
        """
        data_by_frame: list[tuple[int, dict[str, str]]] = []

        with open(path, "r") as f:
            reader = csv.reader(f)
            keys = next(reader)
            for row in reader:
                data = {i: j for i, j in zip(keys, row)}
                data_by_frame.append((int(data["frame_received"]), data))

        # make sure the data is in order of increasing frame received
        data_by_frame.sort(key=lambda x: x[0])

        return data_by_frame


@final
class TranscriptionGroundTruth(BaseFeature[TranscriptionInterface]):
    """
    Note: do not use this feature directly, create with
    `create_transcription_and_audio_ground_truth_features` instead

    Ground truth transcriptions.

    Input interface is `_TranscriptionAndAudioInterface`

    Output interface is `TranscriptionInterface`
    """

    def __init__(
        self, transcription_and_audio_interface: _TranscriptionAndAudioGroundTruth
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
class AudioGroundTruth(BaseFeature[AudioFileInterface]):
    """
    Note: do not use this feature directly, create with
    `create_transcription_and_audio_ground_truth_features` instead

    Ground truth audio files.

    Input interface is `_TranscriptionAndAudioInterface`

    Output interface is `AudioFileInterface`
    """

    def __init__(
        self, transcription_and_audio_interface: _TranscriptionAndAudioGroundTruth
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


def create_transcription_and_audio_ground_truth_features(
    color: BaseFeature[ColorImageInterface], *, csv_path: Path
):
    """
    Create features for transcription and audio ablation.

    Arguments:
    `color` -- feature which return color frames, used for frame count
    `csv_path` -- path to WTD asrOoutput csv file

    Returns:
    transcription -- transcription feature which returns TranscriptionInterface
    audio -- audio feature which returns AudioFileInterface
    """
    ta = _TranscriptionAndAudioGroundTruth(color, csv_path=csv_path)
    transcription = TranscriptionGroundTruth(ta)
    audio = AudioGroundTruth(ta)

    return transcription, audio
