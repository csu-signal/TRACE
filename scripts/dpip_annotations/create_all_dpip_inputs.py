"""
A script to generate all inputs for evaluating this demo on
the Weights Task Dataset. Needs to be run on rosch.

Usage: `python create_all_dpip_inputs.py <dir>` 
"""

import os
import sys
from pathlib import Path

from dpip_gesture_helper import create_gesture_input
from dpip_object_helper import create_object_input
from dpip_utterance_helper import create_utterance_input

UTTERANCE_PATH = "G:\\DPIP\\GAMR\\Utterances\\TB_DPIP_Group_03-master.csv"
AUDIO_PATH = "G:\\DPIP\\DPIP_Azure_Recordings\\TB_DPIP_Group_03-audio.wav"
# OBJECT_PATH = (
#     "G:\\DPIP\\ObjectTracking\\Group03\\TB_DPIP_Group_03-master_synchronized_labels.json"
# )
# GAMR_PATH = "G:\\DPIP\\GAMR\\CSV\\Group_03_merge_CM.csv"
# ANNOTATIONS_FILE = "G:\\DPIP\\Pointing\\GroundTruthFrames\\Group{0}.csv"


def create_all_inputs(parent_dir):
    parent_dir = Path(parent_dir)
    for group in [3]:#[1, 2, 4, 5]:
        print("creating inputs for group", group)
        dir = parent_dir / f"group{group}"
        os.makedirs(dir, exist_ok=True)

        create_utterance_input(
            UTTERANCE_PATH.format(group),
            AUDIO_PATH.format(group),
            dir / "utterances.csv",
            dir / "chunks",
        )
        # create_object_input(OBJECT_PATH.format(group), dir / "objects.csv")
        # create_gesture_input(
        #     GAMR_PATH.format(group),
        #     ANNOTATIONS_FILE.format(group),
        #     dir / "gestures.csv",
        # )


if __name__ == "__main__":
    create_all_inputs(sys.argv[1])
