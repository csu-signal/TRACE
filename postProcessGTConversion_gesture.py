import argparse
import json
from utils import *
import csv

def initalizeCsv(path):
    if os.path.exists(path):
        os.remove(path)

    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "blocks", "body_id", "handedness"])
    return 

def LogCsv(path, frame, blocks, body_id, handedness):
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([frame, blocks, body_id, handedness])
    return 


parser = argparse.ArgumentParser()
parser.add_argument('--gamrPath', nargs='?', default="F:\\Weights_Task\\Data\\GAMR\\CSV\\Group_05_merge_CM.csv")
#parser.add_argument('--annotationsFile', nargs='?', default="F:\\Weights_Task\\Data\\Pointing\\KeyFrameSelectionOutput\\Group_1-master\\point\\combined.csv")
parser.add_argument('--annotationsFile', nargs='?', default="F:\\Weights_Task\\Data\\Pointing\\GroundTruthFrames\\Group5.csv")
parser.add_argument('--outputFile', nargs='?', default="F:\\Weights_Task\\Data\\FactPostProcessing\\Gesture\\Group5.csv")
args = parser.parse_args()

initalizeCsv(args.outputFile)

pointAnnotations = [args.annotationsFile]
pointFeatures = loadKeyFrameFeatures(pointAnnotations)
gamrFeatures = loadGamrFeatures(args.gamrPath)

#for each ground truth pointing frame find the ground truth gamrs and log the targets
for f in pointFeatures:
    start = f[4]
    end = f[5]
    for frameCount in range(int(start), int(end), 1):
        totalSeconds = frameCount / 30 #30 frames per second
        gamrs = convertGamrValues(totalSeconds, gamrFeatures)

        if(gamrs):
            blocks = []
            for g in gamrs:
                if(g.category == GamrCategory.DEIXIS):
                    for t in g.targets:
                        # ignore unknowns
                        if(t != GamrTarget.UNKNOWN and t != GamrTarget.SCALE and t != GamrTarget.MYSTERY_BLOCK):
                            blocks.append(t.value)

            if(len(blocks) > 0):
                LogCsv(args.outputFile, frameCount, blocks, f[2], f[1])


