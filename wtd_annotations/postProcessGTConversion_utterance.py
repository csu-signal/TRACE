import argparse
from utils import loadUtteranceFeatures
import os
import csv
import re

def initalizeCsv(path):
    if os.path.exists(path):
        os.remove(path)

    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utterance_id", "frame_received", "speaker_id", "text", "start_frame", "stop_frame", "audio_file"])
    return 

def LogCsv(path, utterance_id, frame_received, speaker_id, text, start_frame, stop_frame, audio_file):
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([utterance_id, frame_received, speaker_id, text, start_frame, stop_frame, audio_file])
    return 


def create_utterance_input(utterancePath, outputFile):
    initalizeCsv(outputFile)

    utteranceFeatures = loadUtteranceFeatures(utterancePath)
    assert utteranceFeatures is not None

    subs = [
            (r"\bten\b", "10"),
            (r"\btwenty\b", "20"),
            (r"\bthirty\b", "30"),
            (r"\bforty\b", "40"),
            (r"\bfifty\b", "50")
        ]

    #for each ground truth utterance log the ASR values
    count = 1
    for u in utteranceFeatures:
        # print(u)
        startFrame = int(float(u[1]) * 30)
        endFrame = int(float(u[2]) * 30)

        text = u[3]
        for i,j in subs:
            text = re.sub(i, j, text, flags=re.IGNORECASE)

        LogCsv(outputFile, count, endFrame, "Group", text, startFrame, endFrame, "")
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--utterancePath', nargs='?', default="F:\\Weights_Task\\Data\\GAMR\\Utterances\\Group_01.csv")
    parser.add_argument('--outputFile', nargs='?', default="F:\\Weights_Task\\Data\\FactPostProcessing\\Utterances\\Group_01.csv")
    args = parser.parse_args()

    create_utterance_input(args.utterancePath, args.outputFile)
