import pickle

import cv2 as cv
import numpy as np

if __name__ == "__main__":
    for i in range(10):
        with open(f"frame{i:05}.pkl", "rb") as f:
            color, depth, bt, calibration = pickle.load(f)

            im = np.copy(color.frame)
            im = cv.cvtColor(im, cv.COLOR_RGB2BGR)

            cv.putText(im, f"frame{i:05}.pkl", (50, 50), 0, 1, (0, 0, 255))

            cv.imshow("", im)
            cv.waitKey(2000)
