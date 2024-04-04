from tensorflow import keras
from tensorflow.keras.metrics import categorical_accuracy
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
from Face_Detection import image, load_test
import os.path as osp
import os
import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir')
parser.add_argument('--out_dir')
parser.add_argument('--out_file')
args = parser.parse_args()

IMG_DIR  = args.img_dir
OUT_DIR = args.out_dir
if not osp.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
OUT_FILE_NAME = args.out_file


def load_model():
    model = keras.models.load_model("Model/1", custom_objects={'euclideanLoss': euclideanLoss,
                                                               'categorical_accuracy': categorical_accuracy})
    return model

def euclideanLoss(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))

def predict_gaze(model, images, faces, heads):
    preds = model.predict([np.array(images),np.array(faces),np.array(heads)])
    return preds

def visualize_save(filenames,faces,heads, preds):
    df_valfnames = pd.DataFrame(zip(filenames, range(len(filenames))), columns=['filenames', 'index'])
    grouped_df = df_valfnames.groupby(['filenames'], as_index=False).groups
    save_file = collections.defaultdict(dict)
    for i, k in enumerate(grouped_df.keys()):
        ima = Image.open(osp.join(IMG_DIR,k))
        w = np.size(ima)[0]
        h = np.size(ima)[1]
        fa_in = grouped_df[k]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        img_faces=[]
        img_heads = []
        img_gazes = []
        for res in fa_in:
            plt.arrow(heads[res][0] * w, heads[res][1] * h, preds[0][res][0] * w - heads[res][0] * w,
                      preds[0][res][1] * h - heads[res][1] * h, color="red", width=1, head_width=20)
            img_faces.append(faces[res]) # shape:(32, 32, 3)
            img_heads.append(heads[res]) # here the cordinate can be computed by multiply 
            img_gazes.append(preds[0][res]) 
        save_file[k]["faces"]=img_faces
        save_file[k]["heads"]=img_heads
        save_file[k]["gazes"]=img_gazes
        plt.imshow(ima.convert('RGB'))
        plt.savefig(osp.join(OUT_DIR,k))
        np.save(osp.join(OUT_DIR,OUT_FILE_NAME),save_file)

if __name__=="__main__":
    images_dic = image(IMG_DIR)
    filenames,images,faces,heads=load_test(IMG_DIR,images_dic)
    model = load_model()
    preds = predict_gaze(model, images, faces, heads)
    visualize_save(filenames,faces, heads, preds)

    print("finished")