from pathlib import Path

import numpy as np
import pytest

from mmdemo.base_feature import BaseFeature
from mmdemo.demo import Demo
from mmdemo.features.wtd_ablation_testing.transcription_feature import (
    create_transcription_and_audio_ground_truth_features,
)
from mmdemo.interfaces import ColorImageInterface, EmptyInterface
from tests.utils.features import ColorFrameCount


class CollectOutput(BaseFeature[EmptyInterface]):
    """
    Check output of ground truth features
    """

    def __init__(self, *args, output_list):
        super().__init__(*args)
        self.output = output_list

    def get_output(self, *args):
        self.output.append(tuple(i if i.is_new() else None for i in args))
        return EmptyInterface()


@pytest.mark.parametrize("frame_counts,expected_text", [([0, 1, 2, 3, 4, 5], [])])
def test_ground_truth_utterances(frame_counts, expected_text, test_data_dir):
    """
    Make sure that reading ground truth utterances from a file
    works correctly. The features need to be run from a demo
    because there is a hidden feature which helps sync the returned
    ones.
    """
    output = []

    color_frames = ColorFrameCount(frames=frame_counts)
    transcriptions, audio = create_transcription_and_audio_ground_truth_features(
        color_frames,
        csv_path=test_data_dir / "example_ground_truth_wtd" / "utterances.csv",
        chunk_dir_path=test_data_dir,
    )
    output_feature = CollectOutput(transcriptions, audio, output_list=output)

    Demo(targets=[output_feature]).run()
