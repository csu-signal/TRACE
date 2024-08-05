from mmdemo_azure_kinect import create_output_features

import mmdemo
import mmdemo.features as fs

if __name__ == "__main__":
    color, depth, body_tracking = create_output_features()

    gaze = fs.GazeBodyTracking(body_tracking)

    objects = fs.Blocks(color, model="path to model")

    output_frame = fs.Frame([gaze, objects])

    gui = fs.Gui(output_frame)

    mmdemo.Demo(targets=[gui]).run()
