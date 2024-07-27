import argparse
import json
from utils import *
import csv

def initalizeCsv(path):
    if os.path.exists(path):
        os.remove(path)

    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "class", "p10", "p11", "p20", "p21"])
    return 

def LogCsv(path, frame, objectClass, p10, p11, p20, p21):
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([frame, objectClass, p10, p11, p20, p21])
    return 


parser = argparse.ArgumentParser()
parser.add_argument('--objectPath', nargs='?', default="F:\\Weights_Task\\Data\\6DPose\\Group5\\Group_05-objects_interpolated.json")
parser.add_argument('--outputFile', nargs='?', default="F:\\Weights_Task\\Data\\FactPostProcessing\\Objects\\Group5.csv")
args = parser.parse_args()
initalizeCsv(args.outputFile)

objectJsonFile = open(args.objectPath)
objectData = json.load(objectJsonFile)

#for each ground truth object frame log the bounding boxes of each of the objects

image_width = 512
image_height = 512
for key in objectData:
    frameIndex = int(key.split("_")[2].split(".")[0])
    objects = objectData[key] 
    for o in objects:
        bounding_box_xy = o[8:10]
        bounding_box_dims = o[10:12]
        right = int(bounding_box_xy[0])
        bottom = int(bounding_box_xy[1])
        left = int(right - bounding_box_dims[0])
        top = int(bottom - bounding_box_dims[1])
        classIndex = int(o[0])
        if(classIndex <= 4):
            LogCsv(args.outputFile, frameIndex, int(o[0]), top, left, bottom, right)


