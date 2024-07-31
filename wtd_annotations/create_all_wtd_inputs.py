"""
A script to generate all inputs for evaluating this demo on
the Weights Task Dataset. Needs to be run on rosch.

Usage: `python create_all_wtd_inputs.py <dir>`
"""

import os
import sys
from pathlib import Path

from demo.featureModules import AsrFeature, GestureFeature, ObjectFeature

from postProcessGTConversion_gesture import create_gesture_input
from postProcessGTConversion_objects import create_object_input
from postProcessGTConversion_utterance import create_utterance_input

UTTERANCE_PATH = "F:\\Weights_Task\\Data\\GAMR\\Utterances\\Group_{0:02}.csv"
AUDIO_PATH = "F:\\Weights_Task\\Data\\Group_{0:02}-audio.wav"
OBJECT_PATH = "F:\\Weights_Task\\Data\\6DPose\\Group{0}\\Group_{0:02}-objects_interpolated.json"
GAMR_PATH = "F:\\Weights_Task\\Data\\GAMR\\CSV\\Group_{0:02}_merge_CM.csv"
ANNOTATIONS_FILE = "F:\\Weights_Task\\Data\\Pointing\\GroundTruthFrames\\Group{0}.csv"

def create_all_inputs(parent_dir):
    parent_dir = Path(parent_dir)
    for group in [1,2,4,5]:
        print("creating inputs for group", group)
        dir = parent_dir / f"group{group}"
        os.makedirs(dir, exist_ok=True)

        create_utterance_input(UTTERANCE_PATH.format(group), AUDIO_PATH.format(group), dir / AsrFeature.LOG_FILE, dir / "chunks")
        create_object_input(OBJECT_PATH.format(group), dir / ObjectFeature.LOG_FILE)
        create_gesture_input(GAMR_PATH.format(group), ANNOTATIONS_FILE.format(group), dir / GestureFeature.LOG_FILE)

if __name__ == "__main__":
    create_all_inputs(sys.argv[1])
