import itertools
import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
from enum import Enum
import os
from glob import glob
import cv2
import math 
import numpy as np
import csv
import shutil
import json
from numpy.linalg import norm

#https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
class Handedness(Enum):
    Left = "Right"
    Right = "Left"

class ParseResult(Enum):
    Unknown = 0,
    Success = 1,
    Exception = 2,
    InvalidDepth = 3,
    NoObjects = 4,
    NoGamr = 5

class Joint(Enum):
        PELVIS = 0
        SPINE_NAVEL = 1
        SPINE_CHEST = 2
        NECK = 3
        CLAVICLE_LEFT = 4
        SHOULDER_LEFT = 5
        ELBOW_LEFT = 6
        WRIST_LEFT = 7
        HAND_LEFT = 8
        HANDTIP_LEFT = 9
        THUMB_LEFT = 10
        CLAVICLE_RIGHT = 11
        SHOULDER_RIGHT = 12
        ELBOW_RIGHT = 13
        WRIST_RIGHT = 14
        HAND_RIGHT = 15
        HANDTIP_RIGHT = 16
        THUMB_RIGHT = 17
        HIP_LEFT = 18
        KNEE_LEFT = 19
        ANKLE_LEFT = 20
        FOOT_LEFT = 21
        HIP_RIGHT = 22
        KNEE_RIGHT = 23
        ANKLE_RIGHT = 24
        FOOT_RIGHT = 25
        HEAD = 26
        NOSE = 27
        EYE_LEFT = 28
        EAR_LEFT = 29
        EYE_RIGHT = 30
        EAR_RIGHT = 31

class BodyCategory(Enum):
        HEAD = 0
        RIGHT_ARM = 1
        RIGHT_HAND = 7
        LEFT_ARM = 2
        LEFT_HAND = 6
        TORSO = 3
        RIGHT_LEG = 4
        LEFT_LEG = 5

class GamrCategory(str, Enum):
        UNKNOWN = 'unknown'
        EMBLEM = 'emblem'
        DEIXIS = 'deixis'

class GamrTarget(str, Enum):
        UNKNOWN = 'unknown'
        SCALE = 'scale'
        RED_BLOCK = 'red_block'
        BLUE_BLOCK = 'blue_block'
        YELLOW_BLOCK = 'yellow_block'
        GREEN_BLOCK = 'green_block'
        PURPLE_BLOCK = 'purple_block'
        BROWN_BLOCK = 'brown_block'
        MYSTERY_BLOCK = 'mystery_block'
        BLOCKS = 'blocks'

class Object:
    def __init__(self, id, threeD):
        self.threeD = threeD
        self.id = id

#BGR
# Red, Green, Orange, Blue, Purple
colors = [(0, 0, 255), (0, 255, 0), (0, 140, 255), (255, 0, 0), (139,34,104)]
dotColors = [(0, 0, 139), (20,128,48), (71,130,170), (205,95,58), (205,150,205)]


# camera calibration utils

def getCalibrationFromFile(cameraCalibration):
    if(cameraCalibration != None):
        cameraMatrix = np.array([np.array([float(cameraCalibration["fx"]),0,float(cameraCalibration["cx"])]), 
                        np.array([0,float(cameraCalibration["fy"]),float(cameraCalibration["cy"])]), 
                        np.array([0,0,1])])
                
        rotation = np.array([
            np.array([float(cameraCalibration["rotation"][0]),float(cameraCalibration["rotation"][1]),float(cameraCalibration["rotation"][2])]), 
            np.array([float(cameraCalibration["rotation"][3]),float(cameraCalibration["rotation"][4]),float(cameraCalibration["rotation"][5])]), 
            np.array([float(cameraCalibration["rotation"][6]),float(cameraCalibration["rotation"][7]),float(cameraCalibration["rotation"][8])])])
        
        translation = np.array([float(cameraCalibration["translation"][0]), float(cameraCalibration["translation"][1]), float(cameraCalibration["translation"][2])])

        dist = np.array([
            float(cameraCalibration["k1"]), 
            float(cameraCalibration["k2"]),
            float(cameraCalibration["p1"]),
            float(cameraCalibration["p2"]),
            float(cameraCalibration["k3"]),
            float(cameraCalibration["k4"]),
            float(cameraCalibration["k5"]),
            float(cameraCalibration["k6"])])
        
        return cameraMatrix, rotation, translation, dist

# pulled from average values in this file https://colostate-my.sharepoint.com/:x:/g/personal/vanderh_colostate_edu/EQW__wjH4DVMu3N7isnMOqEBIXpQAwuWdwbDJW2He2tv-Q?e=hvem8i 
def getMasterCameraMatrix():
    return np.array([np.array([880.7237639, 0, 951.9401562]), 
                             np.array([0, 882.9017308, 554.4583557]), 
                             np.array([0, 0, 1])])


##################################################################################################

# hand utils

def processHands(image, hand):
    normalized = []
    landmarkArray = calc_landmark_list(image, hand)
    for landmark in pre_process_landmark(landmarkArray):
        normalized.append(landmark)
    return normalized

def getAverageHandLocations(body, w, h, rotation, translation, cameraMatrix, dist):
    hand_left_x_max = 0
    hand_left_y_max = 0
    hand_left_x_min = w
    hand_left_y_min = h

    hand_right_x_max = 0
    hand_right_y_max = 0
    hand_right_x_min = w
    hand_right_y_min = h

    rightXTotal = 0
    leftXTotal = 0
    rightYTotal = 0
    leftYTotal = 0

    #print(body["joint_positions"])
    for jointIndex, joint in enumerate(body["joint_positions"]):
        bodyLocation = getPointSubcategory(Joint(jointIndex))
        if(bodyLocation == BodyCategory.LEFT_HAND): #this should really be a method
            points2D, _ = cv2.projectPoints(
                np.array(joint), 
                rotation,
                translation,
                cameraMatrix,
                dist)

            x = points2D[0][0][0]
            y = points2D[0][0][1]
            leftXTotal += x
            leftYTotal += y

            if x > hand_left_x_max:
                hand_left_x_max = x
            if x < hand_left_x_min:
                hand_left_x_min = x
            if y > hand_left_y_max:
                hand_left_y_max = y
            if y < hand_left_y_min:
                hand_left_y_min = y  
        
        if(bodyLocation == BodyCategory.RIGHT_HAND):
            points2D, _ = cv2.projectPoints(
                np.array(joint), 
                rotation,
                translation,
                cameraMatrix,
                dist)

            x = points2D[0][0][0]
            y = points2D[0][0][1]
            rightXTotal += x
            rightYTotal += y
            if x > hand_right_x_max:
                hand_right_x_max = x
            if x < hand_right_x_min:
                hand_right_x_min = x
            if y > hand_right_y_max:
                hand_right_y_max = y
            if y < hand_right_y_min:
                hand_right_y_min = y  
    
    leftXAverage = leftXTotal / 4
    leftYAverage = leftYTotal / 4
    rightXAverage = rightXTotal / 4
    rightYAverage = rightYTotal / 4  

    return leftXAverage, leftYAverage, rightXAverage, rightYAverage

def createBoundingBox(xAverage, yAverage):
    # xMax = xAverage + (xAverage * 0.05)
    # xMin = xAverage - (xAverage * 0.05)  
    # yMax = yAverage + (yAverage * 0.05) 
    # yMin = yAverage - (yAverage * 0.05)

    xMax = xAverage + (32)
    xMin = xAverage - (32)  
    yMax = yAverage + (32) 
    yMin = yAverage - (32)
    xSpan = xMax - xMin
    ySpan = yMax - yMin

    return [int(xMin - (xSpan)), int(yMin - (ySpan)), int(xMax + (xSpan)), int(yMax + (ySpan))]  

def createHeadBoundingBox(xAverage, yAverage, xRange, yRange):
    xMax = xAverage + xRange
    xMin = xAverage - xRange 
    yMax = yAverage + yRange
    yMin = yAverage - yRange
    xSpan = xMax - xMin
    ySpan = yMax - yMin

    return [int(xMin - (xSpan)), int(yMin - (ySpan)), int(xMax + (xSpan)), int(yMax + (ySpan))]  

def getPointSubcategory(joint):
     if(joint == Joint.PELVIS or joint == Joint.NECK or joint == Joint.SPINE_NAVEL or joint == Joint.SPINE_CHEST):
          return BodyCategory.TORSO
     if(joint == Joint.CLAVICLE_LEFT or joint == Joint.SHOULDER_LEFT or joint == Joint.ELBOW_LEFT):
           return BodyCategory.LEFT_ARM
     if(joint == Joint.WRIST_LEFT or joint == Joint.HAND_LEFT or joint == Joint.HANDTIP_LEFT or joint == Joint.THUMB_LEFT):
          return BodyCategory.LEFT_HAND
     if(joint == Joint.CLAVICLE_RIGHT or joint == Joint.SHOULDER_RIGHT or joint == Joint.ELBOW_RIGHT):
        return BodyCategory.RIGHT_ARM
     if(joint == Joint.WRIST_RIGHT or joint == Joint.HAND_RIGHT or joint == Joint.HANDTIP_RIGHT or joint == Joint.THUMB_RIGHT):
          return BodyCategory.RIGHT_HAND
     if(joint == Joint.HIP_LEFT or joint == Joint.KNEE_LEFT or joint == Joint.ANKLE_LEFT or joint == Joint.FOOT_LEFT):
          return BodyCategory.LEFT_LEG
     if(joint == Joint.HIP_RIGHT or joint == Joint.KNEE_RIGHT or joint == Joint.ANKLE_RIGHT or joint == Joint.FOOT_RIGHT):
          return BodyCategory.RIGHT_LEG
     if(joint == Joint.HEAD or joint == Joint.NOSE or joint == Joint.EYE_LEFT 
        or joint == Joint.EAR_LEFT or joint == Joint.EYE_RIGHT or joint == Joint.EAR_RIGHT):
          return BodyCategory.HEAD

def calc_landmark_list(image, hand):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(hand.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

######################################################################

# pointing vector logic 

def getDirectionalVector(terminal, initial):
      vectorX = terminal[0] - initial[0]
      vectorY = terminal[1] - initial[1]
      vectorZ = terminal[2] - initial[2]
      return np.array([vectorX, vectorY, vectorZ], dtype = int)

def getDirectionalVector2D(terminal, initial):
      vectorX = terminal[0] - initial[0]
      vectorY = terminal[1] - initial[1]
      return (vectorX, vectorY)

def convertTo3D(cameraMatrix, dist, depth, u, v):
    dv, du = depth.shape

    ## ignore frames with invalid depth info
    if(u > du - 1 or v > dv - 1):
        return [], ParseResult.InvalidDepth
    
    z = depth[v, u]
    # print("X: " + str(u) + " Y: " + str(v))
    print("Z: " + str(z))
    if(z == 0):
        print("Invalid Depth, Z returned 0")
        return [], ParseResult.InvalidDepth 

    f_x = cameraMatrix[0, 0]
    f_y = cameraMatrix[1, 1]
    c_x = cameraMatrix[0, 2]
    c_y = cameraMatrix[1, 2]

    points_undistorted = np.array([])
    points_undistorted = cv2.undistortPoints((u,v), cameraMatrix, dist, P=cameraMatrix)
    points_undistorted = np.squeeze(points_undistorted, axis=1)

    result = []
    for idx in range(points_undistorted.shape[0]):
        #try:
            
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
            if(index == 0):
                bx, by = lm[0], lm[1] 
            if(index == 8):  
                tx, ty = lm[0], lm[1]


        tip3D, tSuccess = convertTo3D(cameraMatrix, dist, depth, tx, ty)
        base3D, bSuccess = convertTo3D(cameraMatrix, dist, depth, bx, by)

        if(tSuccess == ParseResult.InvalidDepth or bSuccess == ParseResult.InvalidDepth):
            return (0, 0, 0, 0, 0, 0, 0, ParseResult.InvalidDepth)

        vector3D = getDirectionalVector(tip3D, base3D)
        nextPoint = getVectorPoint(tip3D, vector3D)
        nextPoint = getVectorPoint(nextPoint, vector3D)
        i = 1
        while i < 3:
            nextPoint = getVectorPoint(nextPoint, vector3D)
            i += 1

        #distance = distance3D(base3D, nextPoint)
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

##############################################################################################

# 2D Object location

# https://www.geeksforgeeks.org/check-whether-a-given-point-lies-inside-a-triangle-or-not/
# A utility function to calculate area
# of triangle formed by (x1, y1),
# (x2, y2) and (x3, y3)
 
def area(x1, y1, x2, y2, x3, y3):
 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)
 
 
# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside(x1, y1, x2, y2, x3, y3, x, y):
 
    # Calculate area of triangle ABC
    A = area (x1, y1, x2, y2, x3, y3)
 
    # Calculate area of triangle PBC
    A1 = area (x, y, x2, y2, x3, y3)
     
    # Calculate area of triangle PAC
    A2 = area (x1, y1, x, y, x3, y3)
     
    # Calculate area of triangle PAB
    A3 = area (x1, y1, x2, y2, x, y)
     
    # Check if sum of A1, A2 and A3
    # is same as A
    if(round(A) == round(A1 + A2 + A3)):
        return True
    else:
        return False
    
##############################################################################################

# 3D Object Location

def distance3D(point1, point2):
    # TODO fix depth with the z channel
    # print(point1)
    # print(point2)
    #return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2) + ((point2[2] - point1[2]) ** 2))
    return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))
                     
# get point on vector perpendicular to block
def projectedPoint(p1, p2, p3):
    P12 = getDirectionalVector(p2,p1) #pointing vector
    P13 = getDirectionalVector(p3,p1) #vertex to point

    proj_13over12 = np.dot(P13, P12)*P12/norm(P12)**2
    perpendicular = proj_13over12 - P13

    return getVectorPoint(p3, perpendicular) 

def checkBlocks(blocks, blockStatus, cameraMatrix, dist, depth, cone, frame, shift, gaze):
    for block in blocks:
        targetPoint = [(block.p1[0] + block.p2[0])/2,(block.p1[1] + block.p2[1]) / 2]
        print("Check Block: " + str(block.description))

        try:
            object3D, success = convertTo3D(cameraMatrix, dist, depth, int(targetPoint[0]), int(targetPoint[1]))
            if(success == ParseResult.InvalidDepth):
                print("Ignoring invalid depth")

                if block.description not in blockStatus:
                    cv2.rectangle(frame, 
                        (int(block.p1[0] * 2**shift), int(block.p1[1] * 2**shift)),
                        (int(block.p2[0] * 2**shift), int(block.p2[1] * 2**shift)),
                        color=(0,0,0),
                        thickness=5, 
                        shift=shift)
                    cv2.circle(frame, (int(targetPoint[0] * 2**shift), int(targetPoint[1] * 2**shift)), radius=10, color=(0,0,0), thickness=10, shift=shift)  
                continue
        except:
            continue

        block.target = cone.ContainsPoint(object3D[0], object3D[1], object3D[2], frame, True)
        if(block.target):
            width = 5
            if gaze:
                if block.description not in blockStatus:
                    blockStatus[block.description] = 1
                else:
                    blockStatus[block.description] += 1
                width *= blockStatus[block.description]

            cv2.rectangle(frame, 
                (int(block.p1[0] * 2**shift), int(block.p1[1] * 2**shift)),
                (int(block.p2[0] * 2**shift), int(block.p2[1] * 2**shift)),
                color=(0,255,0),
                thickness=width, 
                shift=shift)
            cv2.circle(frame, (int(targetPoint[0] * 2**shift), int(targetPoint[1] * 2**shift)), radius=10, color=(0,0,0), thickness=10, shift=shift)
        else:
            if block.description not in blockStatus:
                cv2.rectangle(frame, 
                    (int(block.p1[0] * 2**shift), int(block.p1[1] * 2**shift)),
                    (int(block.p2[0] * 2**shift), int(block.p2[1] * 2**shift)),
                    color=(0,0,255),
                    thickness=5, 
                    shift=shift)
                cv2.circle(frame, (int(targetPoint[0] * 2**shift), int(targetPoint[1] * 2**shift)), radius=10, color=(0,0,0), thickness=10, shift=shift)  
    #return blockStatus

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
        print(("Height: {0:0.2f}\n").format(self.Height))

    def conePointsBase(self):
        return [self.BaseX, self.BaseY + self.farRadius, self.BaseZ], [self.BaseX, self.BaseY - self.farRadius, self.BaseZ], [self.BaseX, self.BaseY, self.BaseZ + self.farRadius], [self.BaseX, self.BaseY, self.BaseZ - self.farRadius]
    
    def conePointsVertex(self):
        return [self.VertexX, self.VertexY + self.nearRadius, self.VertexZ], [self.VertexX, self.VertexY - self.nearRadius, self.VertexZ], [self.VertexX, self.VertexY, self.VertexZ + self.nearRadius], [self.VertexX, self.VertexY, self.VertexZ - self.nearRadius]
    
    def projectRadiusLines(self, shift, frame, includeY, includeZ, gaze):
        baseUpY, baseDownY, baseUpZ, baseDownZ = self.conePointsBase()
        vertexUpY, vertexDownY, vertexUpZ, vertexDownZ = self.conePointsVertex()

        if(gaze):
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
            
            pointUpY = (int(baseUp2DY[0] * 2**shift),int(baseUp2DY[1] * 2**shift))
            pointDownY = (int(baseDown2DY[0] * 2**shift),int(baseDown2DY[1] * 2**shift))

            vertexPointUpY = (int(vertexUp2DY[0] * 2**shift),int(vertexUp2DY[1] * 2**shift))
            vertexPointDownY = (int(vertexDown2DY[0] * 2**shift),int(vertexDown2DY[1] * 2**shift))
            
            cv2.line(frame, vertexPointUpY, pointUpY, color=yColor, thickness=5, shift=shift)
            cv2.line(frame, vertexPointDownY, pointDownY, color=yColor, thickness=5, shift=shift)

        if includeZ:
            vertexUp2DZ = convert2D(vertexUpZ, self.cameraMatrix, self.dist)
            vertexDown2DZ = convert2D(vertexDownZ, self.cameraMatrix, self.dist)
            baseUp2DZ = convert2D(baseUpZ, self.cameraMatrix, self.dist)      
            baseDown2DZ = convert2D(baseDownZ, self.cameraMatrix, self.dist)

            pointUpZ = (int(baseUp2DZ[0] * 2**shift),int(baseUp2DZ[1] * 2**shift))
            pointDownZ = (int(baseDown2DZ[0] * 2**shift),int(baseDown2DZ[1] * 2**shift))

            vertexPointUpZ = (int(vertexUp2DZ[0] * 2**shift),int(vertexUp2DZ[1] * 2**shift))
            vertexPpointDownZ = (int(vertexDown2DZ[0] * 2**shift),int(vertexDown2DZ[1] * 2**shift))

            cv2.line(frame, vertexPointUpZ, pointUpZ, color=ZColor, thickness=5, shift=shift)
            cv2.line(frame, vertexPpointDownZ, pointDownZ, color=ZColor, thickness=5, shift=shift)

    def ContainsPoint(self, x, y, z, frame, includeOverlay = False, shift = 7):        
        # cone radius relative to the height at perpindicular point on vector
        proj = projectedPoint(self.vertex, self.base, [x,y,z]) 

        if (math.isnan(proj[0]) or math.isnan(proj[1]) or math.isnan(proj[2])):
            return False #11178

        if includeOverlay:
            proj2D = convert2D(proj, self.cameraMatrix, self.dist)             
            projShifted = (int(proj2D[0] * 2**shift),int(proj2D[1] * 2**shift))
            cv2.circle(frame, projShifted, radius=15, color=(255,0,255), thickness=15, shift=shift)

        dot = np.dot(getDirectionalVector(self.vertex, proj), self.vector)   
        if (dot < 0):
            return False
            
        distVertex = distance3D(proj, self.vertex)
        print(("Distance on Vector: {0:0.2f}").format(distVertex))
        if (distVertex > self.Height):
            return False

        coneRadius = self.nearRadius + (self.farRadius - self.nearRadius) * (distVertex / self.Height)
        print(("Cone Radius: {0:0.2f}").format(coneRadius))

        # point radius relative to the plane/vector
        pointRadius = distance3D(proj, [x,y,z])
        print(("Point Radius: {0:0.2f}").format(pointRadius))

        if (pointRadius <= coneRadius):
            print("Target\n")
            return True
        print("\n")
        return False

##############################################################################################

# robust keyframe annotation utils (read in features, get csv files)

def loadKeyFrameFeatures(csvFiles):
    #Features from the dataset
    featuresArray = []
    for c in csvFiles:
         features = np.loadtxt(c, delimiter=',', dtype=str, ndmin=2, usecols=list(range(0, 8)), skiprows=1)
         if features.size != 0: 
            for f in features:
                featuresArray.append(f)
    return featuresArray

# currently based on the peak frames
def featureFilter(feature, frameNumber):
    if (frameNumber >= int(feature[4]) and frameNumber <= int(feature[5])):
        return True
    else:
        return False
    
# currently based on the peak frames
def bodyFilter(feature, bodyId, hand):
    if (bodyId == int(feature[2]) and hand == feature[1]):
        return True
    else:
        return False
    
def objectAnnotationFilter(key, frameNumber):
    return int(key.split("_")[2].split(".")[0]) == frameNumber
   
def getKeyFrameCsv(dataset):
    csvFiles = []
    for path, subdir, files in os.walk(dataset):
        for file in glob(os.path.join(path, "*.csv")):
            csvFiles.append(file)
    return csvFiles

################################################################################

#GAMR utils

def gamrFeatureFilter(feature, timestamp):
    if (timestamp >= float(feature[1]) and timestamp <= float(feature[2])):
        return True
    else:
        return False
    
def loadGamrFeatures(csvFile):
    featuresArray = []
    features = np.loadtxt(csvFile, delimiter=',', ndmin=2, dtype=str, usecols=list(range(0, 4)))
    if features.size != 0: 
        for f in features:
            featuresArray.append(f)
        return featuresArray

def convertGamrValues(totalSeconds, gamrFeatures):
    gamrArray = []
    applicableGamr = list(filter(lambda g: (gamrFeatureFilter(g, totalSeconds)), gamrFeatures))
    for app in applicableGamr:
        gamrArray.append(Gamr(app[3]))
    return gamrArray

def singleFrameGamrRecall(frame, gamr, blocks, path):
    falseNegative = 0
    falsePositive = 0
    trueNegative = 0
    truePositive = 0
    #for gamr in gamrs:
    if(gamr.category != GamrCategory.DEIXIS):
        return False,0,0,0

    blockDescriptions = []

    # ignore unknowns
    if(GamrTarget.UNKNOWN in gamr.targets):
        return False,0,0,0
    #if the target is the scale no blocks should be selected (true negative), every selected block is a false positive
    if(GamrTarget.SCALE in gamr.targets):
        return False,0,0,0
        # if(len(blocks) > 0):
        #     falsePositive += len(blocks) 
        # else:
        #     trueNegative += 1

        # for b in blocks:
        #     blockDescriptions.append(b.description)

    # if the target is the "blocks" each selected block greater than one is a true positive, if none are selected it's a false negative
    elif(GamrTarget.BLOCKS in gamr.targets):
        if(len(blocks) > 1):
            truePositive += len(blocks)
        else:
            falseNegative += 1

        for b in blocks:
            blockDescriptions.append(b.description)

    # for each block in the gamr if the description matches the gamr target it's a true positive, else it's a false positive
    # if no blocks are selected it's a false negative
    else:
        if(len(blocks) > 0):
            for b in blocks:
                blockDescriptions.append(b.description)
                #IOU (interestion over union) jaccard index, dice's coeff, over samples
                if(b.description in gamr.targets):
                    truePositive += 1
                else:
                    falsePositive += 1
            #if a block is a gamr target but not in the block list, it's a false negative
            for t in gamr.targets:
                if(t not in blockDescriptions):
                    falseNegative +=1

        else:
            for t in gamr.targets:
                falseNegative +=1

    cv2.putText(frame, "False+: " + str(falsePositive), (50,150), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "True+: " + str(truePositive), (50,200), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "False-: " + str(falseNegative), (50,250), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "True-: " + str(trueNegative), (50,300), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)

    if truePositive > 0 and (falseNegative > 0 or falsePositive > 0):
        precision = round(truePositive / (truePositive + falsePositive), 2)
        recall = round(truePositive / (truePositive + falseNegative), 2)
        f1 = round(2 * (precision * recall) / (precision + recall), 2)

        #return recall
        return True, recall, precision, f1
    return True, 0, 0, 0 #if true postitives are 0 is recall/precision 0?

class GamrStats:
    def __init__(self):
        self.falsePositive = 0
        self.truePositive = 0
        self.falseNegative = 0
        self.trueNegative = 0
        self.totalGamr = 0
        self.failedParse = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.totalIou = 0
        self.averageIou = 0
        self.totalIouFrames = 0

    def analyzeGamr(self, frameIndex, gamr, blocks, path):
        if(gamr.category != GamrCategory.DEIXIS):
            return

        dictionary = {
            "frameNumber": frameIndex,
            "category": gamr.category,
            "targets": gamr.targets,
            "blocks": [],
            "IOU": 0
        }

        blockDescriptions = []
        self.totalGamr += 1
        #print("Target: " + gamr.target)
        # ignore unknowns
        if(GamrTarget.UNKNOWN in gamr.targets):
            return
        #if the target is the scale no blocks should be selected (true negative), every selected block is a false positive
        if(GamrTarget.SCALE in gamr.targets):
            if(len(blocks) > 0):
                self.falsePositive += len(blocks) 
            else:
                self.trueNegative += 1

            for b in blocks:
                dictionary["blocks"].append(b.toJSON())
                blockDescriptions.append(b.description)

        # if the target is the "blocks" each selected block greater than one is a true positive, if none are selected it's a false negative
        elif(GamrTarget.BLOCKS in gamr.targets):
            if(len(blocks) > 1):
                self.truePositive += len(blocks)
            else:
                self.falseNegative += 1

            for b in blocks:
                dictionary["blocks"].append(b.toJSON())
                blockDescriptions.append(b.description)

        # for each block in the gamr if the description matches the gamr target it's a true positive, else it's a false positive
        # if no blocks are selected it's a false negative
        else:
            self.totalIouFrames += 1
            if(len(blocks) > 0):
                for b in blocks:
                    dictionary["blocks"].append(b.toJSON())
                    blockDescriptions.append(b.description)
                    #IOU (interestion over union) jaccard index, dice's coeff, over samples
                    if(b.description in gamr.targets):
                        self.truePositive += 1
                    else:
                        self.falsePositive += 1
                #if a block is a gamr target but not in the block list, it's a false negative
                for t in gamr.targets:
                    if(t not in blockDescriptions):
                        self.falseNegative +=1

            else:
                for t in gamr.targets:
                    falseNegative +=1

        intersection = list(np.intersect1d(blockDescriptions, gamr.targets))
        union = list(np.union1d(blockDescriptions, gamr.targets))
        iou = len(intersection) / len(union)
        dictionary["IOU"] = iou
        self.totalIou += iou
        if(self.totalIou > 0):
            self.averageIou =  round(self.totalIou / self.totalIouFrames, 2)

        with open(path, 'a') as f:
            f.write(json.dumps(dictionary) + ",")
            f.close()

        if self.truePositive > 0 and (self.falseNegative > 0 or self.falsePositive > 0):
            self.precision = round(self.truePositive / (self.truePositive + self.falsePositive), 2)
            self.recall = round(self.truePositive / (self.truePositive + self.falseNegative), 2)
            self.f1 = round(2 * (self.precision * self.recall) / (self.precision + self.recall), 2)

class BlockEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__

class Block:
    def __init__(self, description, p1, p2):
        self.p1 = p1
        self.p2 = p2
        # self.height = height
        # self.width = width
        self.target = False

        if(description == 0):
            self.description = GamrTarget.RED_BLOCK
        
        if(description == 1):
            self.description = GamrTarget.YELLOW_BLOCK

        if(description == 2):
            self.description = GamrTarget.GREEN_BLOCK

        if(description == 3):
            self.description = GamrTarget.BLUE_BLOCK

        if(description == 4):
            self.description = GamrTarget.PURPLE_BLOCK

        if(description == 5):
            self.description = GamrTarget.MYSTERY_BLOCK

    def toJSON(self):
        return {
            "description": self.description,
            "x": self.x,
            "y": self.y,
            "height": self.height,
            "width": self.width,
            "target": self.target
        }
        

class Gamr:
    def __init__(self, string):
        self.string = string  
        split = string.split(":") 
        self.targets = []

        if(GamrCategory.DEIXIS in split[0]):
            self.category = GamrCategory.DEIXIS
        elif(GamrCategory.EMBLEM in split[0]):
            self.category = GamrCategory.EMBLEM
        else:
            self.category = GamrCategory.UNKNOWN

        for s in split:
            if("ARG1" in s):
                if(GamrTarget.SCALE in s):
                    self.targets.append(GamrTarget.SCALE)
                elif(GamrTarget.BLOCKS in s):
                    self.targets.append(GamrTarget.BLOCKS)
                elif(GamrTarget.RED_BLOCK in s):
                    self.targets.append(GamrTarget.RED_BLOCK)
                elif(GamrTarget.YELLOW_BLOCK in s):
                    self.targets.append(GamrTarget.YELLOW_BLOCK)
                elif(GamrTarget.PURPLE_BLOCK in s):
                    self.targets.append(GamrTarget.PURPLE_BLOCK)
                elif(GamrTarget.GREEN_BLOCK in s):
                    self.targets.append(GamrTarget.GREEN_BLOCK)
                elif(GamrTarget.BLUE_BLOCK in s):
                    self.targets.append(GamrTarget.BLUE_BLOCK)
                elif(GamrTarget.MYSTERY_BLOCK in s):
                    self.targets.append(GamrTarget.MYSTERY_BLOCK)
                else:
                    self.targets.append(GamrTarget.UNKNOWN)


################################################################################

# Annotation utils
                    
def initalizeRadiusCsv(path):
    if os.path.exists(path):
        os.remove(path)

    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frameIndex", "nearRadius", "farRadius", "recall", "precision", "f1", "targets"])
    return  

def LogRadiusCsv(path, frameIndex, nearRadius, farRadius, recall, precision, f1, gamr):
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([frameIndex, nearRadius, farRadius, recall, precision, f1, gamr.targets])
    return  

def LogAverages(path, label, key, average):
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, key, average])
    return                  

def initalizeCsv(outputFolder, path):
    if os.path.exists(outputFolder):
        shutil.rmtree(outputFolder)
    os.makedirs(outputFolder)

    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase_type", "hand_index", "gesture_type", "group", "participant", "landmarks"])
    return

def LogCsv(path, handIndex, gestureType, gesturePhase, group, participant, landmark_list):
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gesturePhase, handIndex, gestureType, group, participant, *landmark_list])
    return

def initalizeGamrFile(path):
    if os.path.exists(path):
        shutil.rmtree(path)

################################################################################

#drawing utils

def convert2D(point3D, cameraMatrix, dist):
    point, _ = cv2.projectPoints(
        np.array(point3D), 
        np.array([0.0,0.0,0.0]),
        np.array([0.0,0.0,0.0]),
        cameraMatrix,
        dist)
    
    return point[0][0]

################################################################################

# random utils

def convertTimestamp(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
     
    return "%d:%02d:%02d" % (hour, minutes, seconds)

