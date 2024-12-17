import numpy as np
#import matplotlib.pyplot as plt
import datetime
import csv
from numpy.linalg import norm
import math
    
if __name__ == "__main__":
    print(datetime.datetime.now())
    predictions = []
    targets = []
    with open('predictions_no_gazeline.csv','r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            line_split = line.replace('"','').replace('[','').replace(']','').split(',')
            prediction_row = [[float(line_split[j]) for j in range(i,i+5)] for i in range(0,len(line_split),5)]
            predictions.append(prediction_row)
    print("len(predictions)",len(predictions))
    print("len(predictions[0])",len(predictions[0]))
    print("len(predictions[1])",len(predictions[1]))
    print("len(predictions[2])",len(predictions[2]))
    print("len(predictions[3])",len(predictions[3]))
    print("len(predictions[4])",len(predictions[4]))
    predictions = [element for row in predictions for element in row]
    print("len(predictions)",len(predictions))
    print("len(predictions[0])",len(predictions[0]))
    print("len(predictions[1])",len(predictions[1]))
    print("len(predictions[2])",len(predictions[2]))
    print("len(predictions[3])",len(predictions[3]))
    print("len(predictions[4])",len(predictions[4]))
    actual_direction_vectors = []
    predicted_direction_vectors = []
    with open("nose_ear_left_ear_right_by_frame_camera_coords_Group10.csv","r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if len(line) > 1: ###needs to be > 1 rather than > 0 because .csv inserts a comma for a blank line
                ###print(">>>",i,"len(line)",len(line),"\n","*"+line+"*")    
            ###remove the formatting of the .csv file to get a list of eyes/ear_left/ear_right 3D camera coordinates
                line_split = line.replace('[','').replace(']','').split(',')

                ###print("*"+line_split[0]+"*")
                ###print("*"+line_split[1]+"*")
                ###print("*"+line_split[2]+"*")
                for j in range(len(line_split)):
                    while line_split[j][0] == ' ':
                    ###while line_split[j].find('  ') != -1:
                        line_split[j] = line_split[j].replace(' ','',1)
                    while line_split[j].find('  ') != -1:
                        line_split[j] = line_split[j].replace('  ',' ',1)
                ###print("*"+line_split[0]+"*")
                ###print("*"+line_split[1]+"*")
                ###print("*"+line_split[2]+"*")
                line_split = [line_split[j].split(' ')[k] for j in range(len(line_split)) for k in range(3)]
                ###print("line_split\n",line_split)
                ###line_split = [float(line_split[j].split(' ')[k]) for k in range(3) for j in range(len(line_split))]
                ###line_split = [float(line_split[j].split(' ')[k]) for k in range(3) for j in range(len(line_split))]
                nose = np.array([float(line_split[0]),float(line_split[1]),float(line_split[2])])
                ear_left = np.array([float(line_split[3]),float(line_split[4]),float(line_split[5])])
                ear_right = np.array([float(line_split[6]),float(line_split[7]),float(line_split[8])])
                ###print("actual: nose",nose,"ear_left",ear_left,"ear_right",ear_right)
                ear_center = (ear_left + ear_right) / 2
                actual_direction = nose - ear_center
                actual_direction /= np.linalg.norm(nose - ear_center)
                actual_direction_vectors.append(actual_direction)
                ###use predictions
                nose[2] = predictions[i//2][0]
                ear_left[2] = predictions[i//2][3]
                ear_right[2] = predictions[i//2][4]
                ###print("predicted: nose",nose,"ear_left",ear_left,"ear_right",ear_right)
                ear_center = (ear_left + ear_right) / 2
                predicted_direction = nose - ear_center
                predicted_direction /= np.linalg.norm(nose - ear_center)
                predicted_direction_vectors.append(predicted_direction)
    actual_direction_vectors = np.array(actual_direction_vectors)
    print("len(actual_direction_vectors)",len(actual_direction_vectors))
    with open('actual_direction_vectors_test_Group10_nose_eyes_ears.csv','w') as f:
        writer = csv.writer(f)
        ###split the output into chunks fo 16000 so there are "only" 16000 columns in each row
        for i in range(0,len(actual_direction_vectors),16000):
            if i+16000 > len(actual_direction_vectors):
                section = [actual_direction_vectors[j] for j in range(i,len(actual_direction_vectors))]
            else: ###last chunk will likely be less than the full 16000
                section = [actual_direction_vectors[j] for j in range(i,i+16000)]
            writer.writerow(section)
    predicted_direction_vectors = np.array(predicted_direction_vectors)
    print("len(predicted_direction_vectors)",len(predicted_direction_vectors))
    with open('predicted_direction_vectors_test_Group10_nose_eyes_ears.csv','w') as f:
        writer = csv.writer(f)
        ###split the output into chunks fo 16000 so there are "only" 16000 columns in each row
        for i in range(0,len(predicted_direction_vectors),16000):
            if i+16000 > len(predicted_direction_vectors):
                section = [predicted_direction_vectors[j] for j in range(i,len(predicted_direction_vectors))]
            else: ###last chunk will likely be less than the full 16000
                section = [predicted_direction_vectors[j] for j in range(i,i+16000)]
            writer.writerow(section)
    for moving_range in range(5,16,5):
    ###for moving_range in [30,60]:
        print("moving_range",moving_range)
        actual_direction_vectors_moving_average = [sum(actual_direction_vectors[i:i+moving_range])/len(actual_direction_vectors[i:i+moving_range]) for i in 
                                                   range(0,len(actual_direction_vectors)-moving_range+1)]
        predicted_direction_vectors_moving_average = [sum(predicted_direction_vectors[i:i+moving_range])/len(predicted_direction_vectors[i:i+moving_range]) for i in 
                                                   range(0,len(predicted_direction_vectors)-moving_range+1)]
        print("len(actual_direction_vectors_moving_average)",len(actual_direction_vectors_moving_average))
        print("len(predicted_direction_vectors_moving_average)",len(predicted_direction_vectors_moving_average))
        actual_direction_vectors_moving_average = np.array(actual_direction_vectors_moving_average)
        predicted_direction_vectors_moving_average = np.array(predicted_direction_vectors_moving_average)
        cosine_rad = np.sum(actual_direction_vectors_moving_average*predicted_direction_vectors_moving_average,axis=1)/ \
            (norm(actual_direction_vectors_moving_average,axis=1)*norm(predicted_direction_vectors_moving_average,axis=1))
        cosine_degrees = [math.degrees(math.acos(cosine_rad[i])) for i in range(len(cosine_rad))]
        with open('cosine_similarity_test_Group10_nose_eyes_ears_moving_average'+str(moving_range)+'.csv','w') as f:
            writer = csv.writer(f)
            ###split the output into chunks fo 16000 so there are "only" 16000 columns in each row
            for i in range(0,len(cosine_degrees),16000):
                if i+16000 > len(cosine_degrees):
                    section = [cosine_degrees[j] for j in range(i,len(cosine_degrees))]
                else: ###last chunk will likely be less than the full 16000
                    section = [cosine_degrees[j] for j in range(i,i+16000)]
                writer.writerow(section)

    print(datetime.datetime.now())
    