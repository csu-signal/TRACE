import cv2
from PIL import Image
import os
import numpy as np
from demo.featureModules.utils import *

def headBoundingBox(body, rotation, translation, cameraMatrix, dist):
    for jointIndex, joint in enumerate(body["joint_positions"]):
        if(Joint(jointIndex) == Joint.EYE_RIGHT):
            points2D, _ = cv2.projectPoints(
                np.array(joint), 
                rotation,
                translation,
                cameraMatrix,
                dist)
            nose_x = points2D[0][0][0]
            nose_y = points2D[0][0][1]

            return createHeadBoundingBox(nose_x, nose_y, 25, 25), nose_x, nose_y


def head_cord(face):
    head_x = face['box'][0]+face['box'][2]/2
    head_y = face['box'][1]+face['box'][3]/2
    return (head_x,head_y)

def head_cord_azure(face):
    head_x = (face[0]+face[2])/2
    head_y = (face[1]+face[3])/2
    return (head_x,head_y)

def load_frame_azure(frame, framergb, bodies, rotation, translation, cameraMatrix, dist, shift):
    faces_li = []
    heads = []
    images = []
    bodyIds = []

    im = cv2.resize(framergb, (256,256))
    im_ar = np.array(im)

    for _, body in enumerate(bodies):  
        box, nosex, nosey = headBoundingBox(body,rotation, translation, cameraMatrix, dist)
        h, w, c = framergb.shape
        (head_x,head_y) = head_cord_azure(box)
        #print((head_x/w,head_y/h))
        heads.append((head_x/w,head_y/h))
        ltx = int(box[0])
        lty = int(box[1])
        rbx = int(box[2])
        rby = int(box[3])
        #print((ltx,lty,rbx,rby))

        # cv2.rectangle(frame, 
        #     (int(ltx* 2**shift), int(lty * 2**shift)), 
        #     (int(rbx * 2**shift), int(rby * 2**shift)), 
        #     color=(255,255,255),
        #     thickness=3, 
        #     shift=shift)

        face_ima = frame[lty:rby, ltx:rbx]    
        face_ima = cv2.resize(face_ima, (32,32))

        face_ar = np.array(face_ima)/ 255.0
        faces_li.append(face_ar)
        images.append(im_ar.astype("float") / 255.0)
        bodyIds.append(int(body['body_id']))

    return faces_li,heads,images,bodyIds

def load_frame(frame, framergb, detector, shift):
    faces_li = []
    heads = []
    record = []
    images = []

    im = cv2.resize(framergb, (256,256))
    im_ar = np.array(im)

    #use mtcnn to detect the faces and append to record(image,faces)
    faces = detector.detect_faces(framergb)
    record.append(faces)

    # for each detected face
    for im_faces in record:
        h, w, c = framergb.shape
        for face in im_faces:
            (head_x,head_y) = head_cord(face)
            print((head_x/w,head_y/h))
            heads.append((head_x/w,head_y/h))
            ltx = face['box'][0]
            lty = face['box'][1]
            rbx = ltx + face['box'][2]
            rby = lty +face['box'][3]
            print((ltx,lty,rbx,rby))

            # cv2.rectangle(frame, 
            #     (ltx* 2**shift, lty * 2**shift), 
            #     (rbx * 2**shift, rby * 2**shift), 
            #     color=(255,255,255),
            #     thickness=3, 
            #     shift=shift)

            face_ima = frame[lty:rby, ltx:rbx]    
            face_ima = cv2.resize(face_ima, (32,32))

            face_ar = np.array(face_ima)/ 255.0
            faces_li.append(face_ar)
            images.append(im_ar.astype("float") / 255.0)

    return faces_li,heads,images



