import os
import sys
from pathlib import Path

from featureModules import AsrFeature, GestureFeature, ObjectFeature
from wtd_annotations import (create_gesture_input, create_object_input,
                             create_utterance_input)

UTTERANCE_PATH = "F:\\Weights_Task\\Data\\GAMR\\Utterances\\Group_{0:02}.csv"
OBJECT_PATH = "F:\\Weights_Task\\Data\\6DPose\\Group{0}\\Group_{0:02}-objects_interpolated.json"
GAMR_PATH = "F:\\Weights_Task\\Data\\GAMR\\CSV\\Group_{0:02}_merge_CM.csv"
ANNOTATIONS_FILE = "F:\\Weights_Task\\Data\\Pointing\\GroundTruthFrames\\Group{0}.csv"

def create_all_inputs(parent_dir):
    parent_dir = Path(parent_dir)
    for group in [1,2,4,5]:
        print("creating inputs for group", group)
        dir = parent_dir / f"group{group}"
        os.makedirs(dir, exist_ok=True)

        create_utterance_input(UTTERANCE_PATH.format(group), dir / AsrFeature.LOG_FILE)
        create_object_input(OBJECT_PATH.format(group), dir / ObjectFeature.LOG_FILE)
        create_gesture_input(GAMR_PATH.format(group), ANNOTATIONS_FILE.format(group), dir / GestureFeature.LOG_FILE)

if __name__ == "__main__":
    create_all_inputs(sys.argv[1])
