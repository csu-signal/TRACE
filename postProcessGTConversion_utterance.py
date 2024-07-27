import argparse
import json
from utils import *
import csv

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


parser = argparse.ArgumentParser()
parser.add_argument('--utterancePath', nargs='?', default="F:\\Weights_Task\\Data\\GAMR\\Utterances\\Group_01.csv")
parser.add_argument('--outputFile', nargs='?', default="F:\\Weights_Task\\Data\\FactPostProcessing\\Utterances\\Group_01.csv")
args = parser.parse_args()
initalizeCsv(args.outputFile)

utteranceFeatures = loadUtteranceFeatures(args.utterancePath)

#for each ground truth utterance log the ASR values
count = 1
for u in utteranceFeatures:
    print(u)
    startFrame = int(float(u[1]) * 30)
    endFrame = int(float(u[2]) * 30)
    LogCsv(args.outputFile, count, endFrame, "Group", u[3], startFrame, endFrame, "")
    count += 1