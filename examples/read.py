import pickle

import cv2 as cv
import numpy as np

if __name__ == "__main__":
    for i in ["frame_01.pkl", "frame_02.pkl", "gesture_01.pkl", "gesture_02.pkl"]:
        with open(i, "rb") as f:
            color, depth, bt, calibration = pickle.load(f)

            im = np.copy(color.frame)
            im = cv.cvtColor(im, cv.COLOR_RGB2BGR)

            cv.putText(im, i, (50, 50), 0, 1, (0, 0, 255))

            cv.imshow("", im)
            cv.waitKey(2000)
