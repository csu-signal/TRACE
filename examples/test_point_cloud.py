import pickle

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    for i in ["frame00005.pkl"]:
        with open(i, "rb") as f:
            color, depth, bt, calibration = pickle.load(f)

            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            S = 10
            d = depth.frame[::S, ::S]
            c = color.frame[::S, ::S]

            d = d.reshape(-1, d.shape[-1])
            c = c.reshape(-1, c.shape[-1])

            ax.scatter(d[:, 0], d[:, 2], -d[:, 1], c=c / 255.0, s=1)

            plt.show()
