from mtcnn import MTCNN
import cv2
from PIL import Image
import os
import numpy as np

IMG_DIR  = "Demo_img/"

def image(base_dir):
#     viList = os.listdir(base_dir)
    viList =[f for f in os.listdir(base_dir) if not f.startswith('.')]
    res = {}
    for video in viList:
#         im =load_img('{}/{}'.format(base_dir, video), grayscale=False, target_size=(256, 256))
#         im_ar = img_to_array(im, data_format='channels_last')
        im = Image.open(base_dir+'/' + video).convert('RGB')
        im = im.resize((256, 256))
        im_ar = np.array(im)
        a = im_ar.astype("float") / 255.0
        # print(np.shape(a))
        res[video]=a
    return res

def head_cord(face):
    head_x = face['box'][0]+face['box'][2]/2
    head_y = face['box'][1]+face['box'][3]/2
    return (head_x,head_y)

def load_test(base_dir,images_dic):
    filenames = []
    faces_li = []
    heads = []
    record = []
    ImaList = [f for f in os.listdir(base_dir) if not f.startswith('.')]
#     ImaList = os.listdir(base_dir)
    det = []
    #use mtcnn to detect the faces and append to record(image,faces)
    for image in ImaList:
        print(image)
        img = cv2.cvtColor(cv2.imread(base_dir+'/'+image), cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        faces = detector.detect_faces(img)
        record.append([image,faces])
    # for each detected face,
    for pair in record:
        im_faces = pair[1]
        filename = pair[0]
        img = Image.open(base_dir+'/' +filename)
        w= np.size(img)[0]
        h= np.size(img)[1]
#         print("pair:",pair)
#         print("im_faces:",im_faces)
#         print("file:",filename)
        for face in im_faces:
            filenames.append(filename)
            (head_x,head_y) = head_cord(face)
            heads.append((head_x/w,head_y/h))
            ltx = face['box'][0]
            lty = face['box'][1]
            rbx = ltx + face['box'][2]
            rby = lty +face['box'][3]
            # print((ltx,lty,rbx,rby))
            img = Image.open(base_dir+'/'+filename ).convert('RGB')
            face_ima = img.crop((ltx,lty,rbx,rby))
#                 (face['box'][0], face['box'][1],(face['box'][0]+face['box'][2]),(face['box'][1]+face['box'][3])))
            face_ima = face_ima.resize((32,32))
            face_ar = np.array(face_ima)/ 255.0
            faces_li.append(face_ar)
    images=[]
    for file in filenames:
        images.append(images_dic[file])
    return filenames,images,faces_li,heads


if __name__=="__main__":
    images_dic = image(IMG_DIR)
    filenames,images,faces,heads=load_test(IMG_DIR,images_dic)


