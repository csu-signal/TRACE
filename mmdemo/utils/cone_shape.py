import math

import cv2
import numpy as np
from mmdemo.utils.point_vector_logic import getDirectionalVector
from mmdemo.utils.threeD_object_loc import distance3D, projectedPoint
from mmdemo.utils.twoD_object_loc import convert2D


class ConeShape:
    def __init__(self, vertex, base, nearRadius, farRadius, cameraMatrix, dist):
        vector = getDirectionalVector(vertex, base)
        self.VectorX = vector[0]
        self.VectorY = vector[1]
        self.VectorZ = vector[2]
        self.VertexX = vertex[0]
        self.VertexY = vertex[1]
        self.VertexZ = vertex[2]
        self.BaseX = base[0]
        self.BaseY = base[1]
        self.BaseZ = base[2]
        # print(("Vertex X: {0:0.2f}").format(self.VertexX))
        # print(("Vertex Y: {0:0.2f}").format(self.VertexY))
        # print(("Vertex Z: {0:0.2f}\n").format(self.VertexZ))
        # print(("Vector X: {0:0.2f}").format(self.VectorX))
        # print(("Vector Y: {0:0.2f}").format(self.VectorY))
        # print(("Vector Z: {0:0.2f}\n").format(self.VectorZ))
        # print(("Base X: {0:0.2f}").format(self.BaseX))
        # print(("Base Y: {0:0.2f}").format(self.BaseY))
        # print(("Base Z: {0:0.2f}\n").format(self.BaseZ))

        self.farRadius = farRadius
        self.nearRadius = nearRadius

        self.vector = vector
        self.vertex = vertex
        self.base = base

        self.cameraMatrix = cameraMatrix
        self.dist = dist

        self.Height = distance3D(vertex, base)
        # print(("Height: {0:0.2f}\n").format(self.Height))

    def conePointsBase(self):
        return (
            [self.BaseX, self.BaseY + self.farRadius, self.BaseZ],
            [self.BaseX, self.BaseY - self.farRadius, self.BaseZ],
            [self.BaseX, self.BaseY, self.BaseZ + self.farRadius],
            [self.BaseX, self.BaseY, self.BaseZ - self.farRadius],
        )

    def conePointsVertex(self):
        return (
            [self.VertexX, self.VertexY + self.nearRadius, self.VertexZ],
            [self.VertexX, self.VertexY - self.nearRadius, self.VertexZ],
            [self.VertexX, self.VertexY, self.VertexZ + self.nearRadius],
            [self.VertexX, self.VertexY, self.VertexZ - self.nearRadius],
        )

    def projectRadiusLines(self, shift, frame, includeY, includeZ, gaze):
        baseUpY, baseDownY, baseUpZ, baseDownZ = self.conePointsBase()
        vertexUpY, vertexDownY, vertexUpZ, vertexDownZ = self.conePointsVertex()

        if gaze:
            yColor = (255, 107, 170)
            ZColor = (107, 255, 138)
        else:
            yColor = (255, 255, 0)
            ZColor = (243, 82, 121)

        if includeY:
            baseUp2DY = convert2D(baseUpY, self.cameraMatrix, self.dist)
            baseDown2DY = convert2D(baseDownY, self.cameraMatrix, self.dist)
            vertexUp2DY = convert2D(vertexUpY, self.cameraMatrix, self.dist)
            vertexDown2DY = convert2D(vertexDownY, self.cameraMatrix, self.dist)

            pointUpY = (int(baseUp2DY[0] * 2**shift), int(baseUp2DY[1] * 2**shift))
            pointDownY = (
                int(baseDown2DY[0] * 2**shift),
                int(baseDown2DY[1] * 2**shift),
            )

            vertexPointUpY = (
                int(vertexUp2DY[0] * 2**shift),
                int(vertexUp2DY[1] * 2**shift),
            )
            vertexPointDownY = (
                int(vertexDown2DY[0] * 2**shift),
                int(vertexDown2DY[1] * 2**shift),
            )

            cv2.line(
                frame, vertexPointUpY, pointUpY, color=yColor, thickness=5, shift=shift
            )
            cv2.line(
                frame,
                vertexPointDownY,
                pointDownY,
                color=yColor,
                thickness=5,
                shift=shift,
            )

        if includeZ:
            vertexUp2DZ = convert2D(vertexUpZ, self.cameraMatrix, self.dist)
            vertexDown2DZ = convert2D(vertexDownZ, self.cameraMatrix, self.dist)
            baseUp2DZ = convert2D(baseUpZ, self.cameraMatrix, self.dist)
            baseDown2DZ = convert2D(baseDownZ, self.cameraMatrix, self.dist)

            pointUpZ = (int(baseUp2DZ[0] * 2**shift), int(baseUp2DZ[1] * 2**shift))
            pointDownZ = (
                int(baseDown2DZ[0] * 2**shift),
                int(baseDown2DZ[1] * 2**shift),
            )

            vertexPointUpZ = (
                int(vertexUp2DZ[0] * 2**shift),
                int(vertexUp2DZ[1] * 2**shift),
            )
            vertexPpointDownZ = (
                int(vertexDown2DZ[0] * 2**shift),
                int(vertexDown2DZ[1] * 2**shift),
            )

            cv2.line(
                frame, vertexPointUpZ, pointUpZ, color=ZColor, thickness=5, shift=shift
            )
            cv2.line(
                frame,
                vertexPpointDownZ,
                pointDownZ,
                color=ZColor,
                thickness=5,
                shift=shift,
            )

    def ContainsPoint(
        self,
        x,
        y,
        z,
        frame,
        includeOverlay=False,
        shift=7,
        gesture=False,
        index=(0, 0, 0),
    ):
        # cone radius relative to the height at perpindicular point on vector
        proj = projectedPoint(self.vertex, self.base, [x, y, z])

        if math.isnan(proj[0]) or math.isnan(proj[1]) or math.isnan(proj[2]):
            return False, -1

        if includeOverlay:
            proj2D = convert2D(proj, self.cameraMatrix, self.dist)
            projShifted = (int(proj2D[0] * 2**shift), int(proj2D[1] * 2**shift))
            cv2.circle(
                frame,
                projShifted,
                radius=15,
                color=(255, 0, 255),
                thickness=15,
                shift=shift,
            )

        dot = np.dot(getDirectionalVector(self.vertex, proj), self.vector)
        if dot < 0:
            return False, -1

        distVertex = distance3D(proj, self.vertex)
        # print(("Distance on Vector: {0:0.2f}").format(distVertex))
        if distVertex > self.Height:
            return False, -1

        coneRadius = self.nearRadius + (self.farRadius - self.nearRadius) * (
            distVertex / self.Height
        )
        # print(("Cone Radius: {0:0.2f}").format(coneRadius))

        # point radius relative to the plane/vector
        pointRadius = distance3D(proj, [x, y, z])
        # print(("Point Radius: {0:0.2f}").format(pointRadius))

        if pointRadius <= coneRadius:
            # print("Target\n")
            if gesture:
                return True, distance3D(index, proj)
            else:
                return True, -1
        # print("\n")
        return False, -1
