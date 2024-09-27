import time
from pathlib import Path

import numpy as np
import pytest

from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo
from mmdemo.features.wtd_ablation_testing.transcription_feature import (
    create_transcription_and_audio_ground_truth_features,
)
from mmdemo.interfaces import (
    AudioFileInterface,
    ColorImageInterface,
    EmptyInterface,
    TranscriptionInterface,
)
from tests.utils.features import ColorFrameCount


class CollectOutputDelay(BaseFeature[EmptyInterface]):
    """
    Collect output of ground truth features and do a small
    delay so the times do not overlap
    """

    def __init__(self, *args, output_list):
        super().__init__(*args)
        self.output = output_list

    def get_output(self, *args):
        self.output.append(tuple(i if i.is_new() else None for i in args))

        # we can assume frames do not happen at the exact same time
        time.sleep(0.01)

        return EmptyInterface()


@pytest.mark.parametrize(
    "frame_counts,expected_text",
    [
        (
            [0, 11, 21, 31, 41, 51],
            [None, "test one", None, "test two", "test three", "test four"],
        ),
        (
            [0, 11, 12, 13, 14, 21, 31, 32, 41, 51, 52, 53],
            [
                None,
                "test one",
                None,
                None,
                None,
                None,
                "test two",
                None,
                "test three",
                "test four",
                None,
                None,
            ],
        ),
        (
            [51, 52, 53, 54, 55, 56, 57],
            ["test one", "test two", "test three", "test four", None, None, None],
        ),
    ],
)
def test_ground_truth_utterances(frame_counts, expected_text, test_data_dir):
    """
    Make sure that reading ground truth utterances from a file
    works correctly. The features need to be run from a demo
    because there is a hidden feature which helps sync the returned
    ones.

    Arguments:
    `frame_counts` -- image frame counts to evaluate on
    `expected_text` -- either the expected transcription or None
                        if there is no expected transcription
    """
    output: list[tuple[TranscriptionInterface | None, AudioFileInterface | None]] = []

    color_frames = ColorFrameCount(frames=frame_counts)
    transcriptions, audio = create_transcription_and_audio_ground_truth_features(
        color_frames,
        csv_path=test_data_dir / "example_ground_truth_wtd" / "utterances.csv",
        chunk_dir_path=test_data_dir,
    )
    output_feature = CollectOutputDelay(transcriptions, audio, output_list=output)

    Demo(targets=[output_feature]).run()

    for (trans, audio_file), expected in zip(output, expected_text):
        if expected is not None:
            # if there is expected to be text, check that the correct
            # output is returned and that the transcription and audio
            # interfaces have identical info
            assert isinstance(trans, TranscriptionInterface)
            assert isinstance(audio_file, AudioFileInterface)
            assert trans.start_time == audio_file.start_time
            assert trans.end_time == audio_file.end_time
            assert trans.speaker_id == audio_file.speaker_id
            assert audio_file.path == test_data_dir / "activity.wav"
            assert trans.text == expected
        else:
            assert trans is None
            assert audio_file is None
