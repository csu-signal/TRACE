import argparse
import csv
import json
import os


def create_object_input(objectPath, outputFile):
    with open(objectPath, "r") as objectJsonFile:
        objectData = json.load(objectJsonFile)

    assert not os.path.exists(outputFile), "output file already exists"

    with open(outputFile, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "class", "p10", "p11", "p20", "p21"])

        # for each ground truth object frame log the bounding boxes of each of the objects

        image_width = 1920
        image_height = 1080
        for key in objectData:
            frameIndex = int(key.split("_")[2].split(".")[0])
            objects = objectData[key]
            for o in objects:
                bounding_box_xy = o[8:10]
                bounding_box_dims = o[10:12]
                right = int(image_width - bounding_box_xy[0])
                bottom = int(image_height - bounding_box_xy[1])
                left = int(right - bounding_box_dims[0])
                top = int(bottom - bounding_box_dims[1])
                classIndex = int(o[0])
                if classIndex <= 4:
                    writer.writerow([frameIndex, int(o[0]), left, top, right, bottom])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objectPath",
        nargs="?",
        default="E:\\Weights_Task\\Data\\6DPose\\Group5\\Group_05-objects_interpolated.json",
    )
    parser.add_argument(
        "--outputFile",
        nargs="?",
        default="E:\\Weights_Task\\Data\\FactPostProcessing\\Objects\\Group5.csv",
    )
    args = parser.parse_args()

    create_object_input(args.objectPath, args.outputFile)
