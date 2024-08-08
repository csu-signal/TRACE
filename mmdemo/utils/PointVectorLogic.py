import cv2
import numpy as np

from mmdemo.utils.SupportUtils import ParseResult


def getDirectionalVector(terminal, initial):
    vectorX = terminal[0] - initial[0]
    vectorY = terminal[1] - initial[1]
    vectorZ = terminal[2] - initial[2]
    return np.array([vectorX, vectorY, vectorZ], dtype=int)


def getDirectionalVector2D(terminal, initial):
    vectorX = terminal[0] - initial[0]
    vectorY = terminal[1] - initial[1]
    return (vectorX, vectorY)


def convertTo3D(cameraMatrix, dist, depth, u, v):
    dv, du = depth.shape

    ## ignore frames with invalid depth info
    if u > du - 1 or v > dv - 1:
        return [], ParseResult.InvalidDepth

    z = depth[v, u]
    # print("X: " + str(u) + " Y: " + str(v))
    # print("Z: " + str(z))
    if z == 0:
        # print("Invalid Depth, Z returned 0")
        return [], ParseResult.InvalidDepth

    f_x = cameraMatrix[0, 0]
    f_y = cameraMatrix[1, 1]
    c_x = cameraMatrix[0, 2]
    c_y = cameraMatrix[1, 2]

    points_undistorted = np.array([])
    points_undistorted = cv2.undistortPoints((u, v), cameraMatrix, dist, P=cameraMatrix)
    points_undistorted = np.squeeze(points_undistorted, axis=1)

    result = []
    for idx in range(points_undistorted.shape[0]):
        # try:

        x = (points_undistorted[idx, 0] - c_x) / f_x * z
        y = (points_undistorted[idx, 1] - c_y) / f_y * z
        result.append(x.astype(float))
        result.append(y.astype(float))
        result.append(z.astype(float))

    # except:
    #     print("An exception occurred")
    return result, ParseResult.Success


def getVectorPoint(terminal, vector):
    return (terminal[0] + vector[0], terminal[1] + vector[1], terminal[2] + vector[2])


def processPoint(landmarks, box, w, h, cameraMatrix, dist, depth):
    try:
        for index, lm in enumerate(landmarks):
            if index == 5:
                bx, by = lm[0], lm[1]
            if index == 8:
                tx, ty = lm[0], lm[1]

        tip3D, tSuccess = convertTo3D(cameraMatrix, dist, depth, tx, ty)
        base3D, bSuccess = convertTo3D(cameraMatrix, dist, depth, bx, by)

        if tSuccess == ParseResult.InvalidDepth or bSuccess == ParseResult.InvalidDepth:
            return (0, 0, 0, 0, 0, 0, 0, ParseResult.InvalidDepth)

        vector3D = getDirectionalVector(tip3D, base3D)
        nextPoint = getVectorPoint(tip3D, vector3D)
        nextPoint = getVectorPoint(nextPoint, vector3D)
        i = 1
        while i < 3:
            nextPoint = getVectorPoint(nextPoint, vector3D)
            i += 1

        # distance = distance3D(base3D, nextPoint)
        return (tx, ty, tip3D, bx, by, base3D, nextPoint, ParseResult.Success)
    except Exception as error:
        print(error)
        return (0, 0, 0, 0, 0, 0, 0, ParseResult.Exception)


def getRadiusPoint(rUp, rDown, vectorPoint):
    up = vectorPoint.copy()
    down = vectorPoint.copy()
    up[0][1] += rUp
    down[0][1] -= rDown
    return up, down
