import numpy as np
#import matplotlib.pyplot as plt
import datetime
import csv
 
def get_predictions(file_name,zscore):
    predictions = []
    with open(file_name,"r") as f:
        for i, line in enumerate(f):
            line = line.strip().split(',')
            [predictions.append(abs(float(line[j]))) for j in range(len(line)) if len(line[j]) > 0 and line[0] != 'zscore']
    predictions = np.array(predictions)
    zscore_mean = np.mean(predictions)
    zscore_std_dev = np.std(predictions)
    if zscore == 3:
        print("len predictions",len(predictions))
        print("zscore_mean zscore_std_dev",zscore_mean,zscore_std_dev)
    predictions = [predictions[j] for j in range(len(predictions)) if np.absolute(predictions[j]-zscore_mean)/zscore_std_dev < zscore]
    print("len predictions",len(predictions))
    new_mean = np.mean(predictions)
    new_std_dev = np.std(predictions)
    with open('cosine_similarity_zscore_no_averaging.csv','a') as f:
        writer = csv.writer(f)
        ###split the output into chunks fo 16000 so there are "only" 16000 columns in each row
        writer.writerow(['zscore',zscore])
        for i in range(0,len(predictions),16000):
            if i+16000 > len(predictions):
                section = [predictions[j] for j in range(i,len(predictions))]
            else: ###last chunk will likely be less than the full 16000
                section = [predictions[j] for j in range(i,i+16000)]
            writer.writerow(section)
    return new_mean, new_std_dev

if __name__ == "__main__":
    print(datetime.datetime.now())
    '''
    file_names = ["depth_predictions_from_json_bignet_Group_01_all_frames_gpu_20epochs_all_features2.csv",
                  "depth_predictions_from_json_bignet_Group_01_all_frames_gpu_20epochs_no_gazeline.csv",
                  ".csv and old .py\depth_predictions_from_json_bignet_all_groups_all_frames_gpu_20epochs_all_features4.csv",
                  "depth_predictions_from_json_bignet_all_groups_all_frames_gpu_20epochs_no_gazeline3.csv"]
    
    file_names = [###"depth_predictions_test_Group10_no_gazeline.csv",
                  ###"depth_predictions_test_Group10_all_features.csv"
                  ".csv and old .py\depth_predictions_from_json_bignet_all_groups_all_frames_gpu_20epochs_all_features4.csv",
                  ".csv and old .py\depth_predictions_from_json_bignet_all_groups_all_frames_gpu_20epochs_no_gazeline3.csv",
                  ]
    file_names = ["cosine_similarity_test_Group10_all_features.csv"]
    '''
    file_names = [###"cosine_similarity_test_Group10_nose_eyes_ears_moving_average5.csv",
                  ###"cosine_similarity_test_Group10_nose_eyes_ears_moving_average10.csv",
                  ###"cosine_similarity_test_Group10_nose_eyes_ears_moving_average15.csv",
                  ###"cosine_similarity_test_Group10_nose_eyes_ears_moving_average30.csv",
                  ###"cosine_similarity_test_Group10_nose_eyes_ears_moving_average60.csvcapitalize",
                  "cosine_similarity_test_Group10_nose_eyes_ears.csv"]
    for file_name in file_names:
        print()
        zscore = 3
        new_mean, new_std_dev = get_predictions(file_name,zscore)
        print("zscore mean std dev", zscore, new_mean, new_std_dev)
        zscore = 2
        new_mean, new_std_dev = get_predictions(file_name,zscore)
        print("zscore mean std dev", zscore, new_mean, new_std_dev)
        zscore = 1.28
        new_mean, new_std_dev = get_predictions(file_name,zscore)
        print("zscore mean std dev", zscore, new_mean, new_std_dev)
    print(datetime.datetime.now())