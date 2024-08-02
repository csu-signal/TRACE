import csv
import json
import os
from pathlib import Path

import pandas as pd
from testing_profiles import TestDenseParaphrasingProfile

from demo.featureModules import (AsrFeature, DenseParaphrasingFeature,
                                 GestureFeature)


# inputs is [(utterance, list of block names)]
def build_dense_paraphrasing_inputs(inputs, dir):
    os.makedirs(dir, exist_ok=True)
    max_index = 0
    with open(dir / AsrFeature.LOG_FILE, "w", newline="") as asr:
        with open(dir / GestureFeature.LOG_FILE, "w", newline="") as gesture:
            asr_writer = csv.writer(asr)
            gesture_writer = csv.writer(gesture)
            asr_writer.writerow(
                [
                    "utterance_id",
                    "frame_received",
                    "speaker_id",
                    "text",
                    "start_frame",
                    "stop_frame",
                    "audio_file",
                ]
            )
            gesture_writer.writerow(["frame", "blocks", "body_id", "handedness"])

            for i, (utterance, blocks) in enumerate(inputs):
                asr_writer.writerow(
                    [i, 30 * i + 30, "", utterance, 30 * i, 30 * i + 29, ""]
                )
                gesture_writer.writerow([30 * i + 15, json.dumps(blocks), 0, "Left"])
                max_index = i

    return 30 * max_index + 31


def test_dense_paraphrasing():
    # [((utterance, blocks), expected output), ...]
    TEST_CASES = [
        (
            ("This one", ["red"]),
            "red one"
        ),
        (
            ("That block", ["blue"]),
            "blue block"
        ),
        (
            ("This is 10 and that is 20", ["red", "green"]),
            "red is 10 and green is 20"
        ),
        (
            ("These are both 10", ["red", "blue"]),
            "red, blue are both 10"
        ),
        (
            ("Those are 10 and this one is 20", ["red", "blue", "green"]),
            "red, blue, green are 10 and red one is 20",
        ),
        (
            ("This is 10 and this is 20", ["blue", "green"]),
            "blue is 10 and green is 20",
        ),
    ]

    dir = Path("dense_paraphrasing_test_tmp")
    input_dir = dir / "input"
    output_dir = dir / "output"

    max_frames = build_dense_paraphrasing_inputs(
        map(lambda x: x[0], TEST_CASES), input_dir
    )

    TestDenseParaphrasingProfile(input_dir, output_dir, max_frames).run()

    df = pd.read_csv(output_dir / DenseParaphrasingFeature.LOG_FILE)

    # all utterances should have been processed
    assert len(df) == len(TEST_CASES)

    # check dense paraphrasing output
    for actual, expected in zip(df["updated_text"], map(lambda x: x[1], TEST_CASES)):
        assert actual == expected
