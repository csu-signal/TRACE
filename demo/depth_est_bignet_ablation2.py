import numpy as np
from sklearn.model_selection import train_test_split
import torch
#import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import datetime
from tqdm import tqdm
import csv
 
###The dataset is stored in a .csv file containing preprocessed outputs from the Group_01 to Group_10 json files.
###The .csv file format is nose,eyes,ears pixel locations that are mapped to 2D from an Azure Kinect 3D camera,
###as well as a gaze line in the direction a body is looking that is also mapped to 2D. The data from the preprocessed
###json files is in y,x format, so once the dataset is input from the .csv file as a series of pixel locations, each
###two numbers represent a pixel location, with the first being the y coordinate and the second being the x coordinate.
def get_dataset(feature_to_remove):
    ###This function removes the feature from the data set that is sent from the main routine as a way of ablation 
    ###testing. The features are nose = feature 0, eye_left=1, eye_right=2, ear_left=3, ear_right=4, gaze line=5.
    ###If all features are to be included (none removed), the input will be 6. The features are the inputs to the
    ###ML algorithm.
    def _joint_x_joint_y(feature_to_remove,joint_pixels_frame_orientation_chars):
        ###If the gaze line feature is to be removed, this consists of many points. Only the first 10 coorinates are 
        ###kept, nose 0,1; eye_left 2,3; eye_right 4,5: ear_left 6,7; and ear_right 8,9.
        if feature_to_remove == 5:
            joint_pixels_frame_orientation_chars = joint_pixels_frame_orientation_chars[:10]
            ###joint_y and joint_x are used to find the minimum location to build a bounding box around the pixels of
            ###the features
            joint_y = [int(joint_pixels_frame_orientation_chars[j]) for j in range(0,len(joint_pixels_frame_orientation_chars)-1,2)]
            joint_x = [int(joint_pixels_frame_orientation_chars[j]) for j in range(1,len(joint_pixels_frame_orientation_chars),2)]
        else:
            ###If the feature to be removed is one of the nose/eyes/ears (<5), only one pixel location (two coorindates)
            ###needs to be removed.
            if feature_to_remove < 5: ###one of the five joints
                joint_pixels_frame_orientation_chars.pop(feature_to_remove*2)
                joint_pixels_frame_orientation_chars.pop(feature_to_remove*2+1)
                ###joint_y and joint_x are used to find the minimum location to build a bounding box around the pixels of
                ###the features
                joint_y = [int(joint_pixels_frame_orientation_chars[j]) for j in range(0,len(joint_pixels_frame_orientation_chars)-1,2) 
                           if j != feature_to_remove*2]
                joint_x = [int(joint_pixels_frame_orientation_chars[j]) for j in range(1,len(joint_pixels_frame_orientation_chars),2) 
                           if j != (feature_to_remove*2+1)]
            else:
                ###This case is where all features are used, so none are removed
                joint_y = [int(joint_pixels_frame_orientation_chars[j]) for j in range(0,len(joint_pixels_frame_orientation_chars)-1,2)]
                joint_x = [int(joint_pixels_frame_orientation_chars[j]) for j in range(1,len(joint_pixels_frame_orientation_chars),2)]
        return joint_pixels_frame_orientation_chars, joint_x, joint_y
    
    ###joint_pixels_frame_orientation contains the pixel locations of the features input to the ML algorithm,
    ###which consists of nose/eyes/ears/gaze line coordinates (less whatever is ablated) for each frame.
    joint_pixels_frame_orientation = [] 
    ###joint_depths has the targets for the ML algorithm which consist of the distance (depth) from the Azure
    ###Kinect 3D camera to the corresponding nose/eyes/ears 2D pixel locations
    joint_depths = [] 
    print("depth_predictions_from_json_bignet_all_groups_all_frames_gpu_20epochs_no_gazeline3")
    ###with open("TRACE-emnlp-demo/depth_joints_from_json_Group_01_every_5frames.csv","r") as f:
    ###max_x = 0
    ###max_y = 0
    with open("depth_joints_from_json_all_groups_all_frames.csv","r") as f:
        for i, line in enumerate(f):
            frame = []
            line = line.strip()
            if len(line) > 1: ###needs to be > 1 rather than > 0 because .csv inserts a comma for a blank line
                ###remove the formatting of the .csv file to get a list of pixel coordinates
                line_split = line.replace('"','').split("]]")
                line_split[0] = line_split[0].replace('[','').replace(']','')
                line_split[1] = line_split[1].replace('[','').replace(']','')
                joint_pixels_frame_orientation_chars = line_split[0].split(',')
                ###extract the feature inputs for the ML algorithm
                joint_pixels_frame_orientation_chars, joint_x, joint_y = _joint_x_joint_y(feature_to_remove,joint_pixels_frame_orientation_chars)
                ###A bounding box is defined that will be 270 pixels in the x direction and 480 pixels in
                ###the y direction to shrink the processing field down from 1080x1920. The box will have it's
                ###lower, left corner 20 pixes below the lowest y coordinate and 20 pixels to the left of the
                ###x coordinate.
                left_x = min(joint_x) - 20
                bottom_y = min(joint_y) - 20
                ###In case the data exceeds the size of the bounding box, pop coorinates off of the gaze line
                ###(which are at the end of the joint_pixels_frame_orientation_chars) until it fits in the
                ###bounding box. Pop twice, one for y and one for x coordinate.
                if (max(joint_x) - left_x) >= 270:
                    ###print("i, x", max(joint_x) - left_x)
                    while (max(joint_x) - left_x) >= 270:
                        joint_pixels_frame_orientation_chars.pop()
                        joint_pixels_frame_orientation_chars.pop()
                        ###update the feature inputs for the ML algorithm
                        joint_pixels_frame_orientation_chars, joint_x, joint_y = _joint_x_joint_y(feature_to_remove,joint_pixels_frame_orientation_chars)
                        left_x = min(joint_x) - 20 ###recalculate the left side of the bounding box
                    ###print("i, x", max(joint_x) - left_x)
                ###if (max(joint_x) - left_x) > max_x:
                ###    max_x = max(joint_x) - left_x
                ###In case the data exceeds the size of the bounding box, pop coorinates off of the gaze line
                ###(which are at the end of the joint_pixels_frame_orientation_chars) until it fits in the
                ###bounding box. Pop twice, one for y and one for x coordinate.
                if (max(joint_y) - bottom_y) >= 480:
                    ###print("i, y", max(joint_y) - bottom_y)
                    while (max(joint_y) - bottom_y) >= 480:
                        joint_pixels_frame_orientation_chars.pop()
                        joint_pixels_frame_orientation_chars.pop()
                        joint_pixels_frame_orientation_chars, joint_x, joint_y = _joint_x_joint_y(feature_to_remove,joint_pixels_frame_orientation_chars)
                        bottom_y = min(joint_y) - 20
                ###if (max(joint_y) - bottom_y) > max_y:
                ###    max_y = max(joint_y) - bottom_y
                ###
                ###move the coordinates for the nose/eyes/ears/gaze line to the size of the bounding box
                for j in range(0,len(joint_pixels_frame_orientation_chars),2):
                    frame.append((int(joint_pixels_frame_orientation_chars[j+1])-left_x,int(joint_pixels_frame_orientation_chars[j])-bottom_y))
                joint_pixels_frame_orientation.append(frame)
                ###split out the targets for the machine learning algorithm, collect as a list of floats
                depths_chars = line_split[1].split(',')
                depths = [float(depths_chars[j]) for j in range(1,len(depths_chars))]
                joint_depths.append(depths)
    ###print("max_x max_y",max_x,max_y)
    return joint_pixels_frame_orientation, joint_depths

###return the root mean squared of two numpy arrays
def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

###The class definition of the neural network model. 
class net(nn.Module):
    ###def __init__(self,input_size, input_len):
    def __init__(self):
        super(net,self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(11,11))
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5,5))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3))
        ###the first number has to change for every model, let it run and bomb and use the b number in the error
        ###RuntimeError: mat1 and mat2 shapes cannot be multiplied(a*b and c*d)
        ###self.fc1 = nn.Linear(256*265*475, 1024)
        ###self.fc1 = nn.Linear(96768,8192) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(96768, 1024)
        self.fc3 = nn.Linear(1024, 32)
        self.fc4 = nn.Linear(32,5)

    def forward(self,x):
        ###print("x.size()",x.size())
        output = self.conv1(x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv3(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv4(output)
        output = self.relu(output)
        output = self.pool(output)
        ###print("output.shape",output.shape)
        output = torch.flatten(output, 1)
        ###output = self.fc1(output)
        ###output = self.relu(output)
        ###output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc3(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc4(output)
        return output

###This custom dataset is used to expand the data from the .csv file input in function get_dataset. In subfuction
###__getitem__, the bounding box is set to an array of zeros. The data input from the .csv file is used to set
###each of those locations to a 1. This means that the "image" is mostly zeros, with only the locations
###corresponding to the pixels of the input set to a 1. The ML algorithm will then use a CNN to train to predict
###the target depths.
class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        frame = np.zeros((1,270,480), dtype=np.int8)
        ###print("idx",idx,"frame.shape",frame.shape)
        for pixel in self.X[idx]:
            frame[0][pixel[0]][pixel[1]] = 1
        return torch.tensor(self.y[idx],dtype=torch.float32), torch.tensor(frame,dtype=torch.float32)
    
if __name__ == "__main__":
    print(datetime.datetime.now())
    feature_to_remove = 5 ###nose = feature 0, eye_left=1, eye_right=2, ear_left=3, ear_right=4, gaze line=5, all features = 6
    X, y = get_dataset(feature_to_remove) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
    print("len(X_train)",len(X_train),"len(y_train)",len(y_train))
    train_dataset = MyDataset(X_train, y_train, None)
    print("len(train_dataset.X)",len(train_dataset.X),"len(train_dataset.y)",len(train_dataset.y))
    print("len(train_dataset)",len(train_dataset))
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=12, shuffle=False)
    test_dataset = MyDataset(X_test, y_test, None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=12, shuffle=False)
    ###device = torch.device('cpu')
    device = torch.device('cuda:0')
    model = net().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    ###print out the number of parameters in the model as a reality check
    print(sum(p.numel() for p in model.parameters()) / 1e6)
    ###train the model
    losses = []
    epochs = 20
    batch_range = 0
    train_err = []
    best_loss = 9999999
    ###for epoch in tqdm(range(epochs)):
    for epoch in tqdm(range(epochs)):
        print("epoch", epoch)
        for batch in train_loader:
            ###print("len(batch)",len(batch))
            y_train_batch_t, X_train_batch_t = batch
            y_train_batch_t, X_train_batch_t = y_train_batch_t.to(device), X_train_batch_t.to(device)
            ###print("y_train_batch_t.shape",y_train_batch_t.shape,"X_train_batch_t.shape",X_train_batch_t.shape)
            y_pred = model(X_train_batch_t)
            ###print("batch_range",batch_range,"y_pred", y_pred.shape,"y_train_batch_t",y_train_batch_t.shape)
            loss = criterion(y_pred,y_train_batch_t) #calculating loss
            ###backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%1 == 0:
            print("\n", epoch, datetime.datetime.now(), "\ntrain err this step", loss.detach().item())
            train_err.append(loss)
        if best_loss > loss.detach().item():
            best_loss = loss.detach().item()
            ###torch.save(model, 'depth_est_model2')
    ###evaluate the model on the held out test dataset
    model.eval()
    predictions = []
    targets = []
    for batch in test_loader:
        y_test_batch_t, X_test_batch_t = batch
        y_test_batch_t, X_test_batch_t = y_test_batch_t.to(device), X_test_batch_t.to(device)
        y_pred = model(X_test_batch_t)
        ###print("y_pred",y_pred.shape,"y_test",y_test_batch_t.shape)
        [predictions.append(x) for x in y_pred.tolist()]
        [targets.append(x) for x in y_test_batch_t.tolist()]
    print("targets",len(targets),"predictions",len(predictions))
    print("RMSE", rmse(np.array(predictions),np.array(targets)),"\n",datetime.datetime.now())
    print("depth_predictions_from_json_bignet_all_groups_all_frames_gpu_20epochs_no_gazeline3")
    ###write the deltas between predictions and targets for further processing in Excel (mostly to get the
    ###average error and std dev)
    with open('depth_predictions_from_json_bignet_all_groups_all_frames_gpu_20epochs_no_gazeline3.csv','w') as f:
        writer = csv.writer(f)
        diffs = np.array(predictions).flatten()-np.array(targets).flatten()
        ###split the output into chunks fo 16000 so there are "only" 16000 columns in each row
        for i in range(0,len(diffs),16000):
            if i+16000 > len(diffs):
                section = [diffs[j] for j in range(i,len(diffs))]
            else: ###last chunk will likely be less than the full 16000
                section = [diffs[j] for j in range(i,i+16000)]
            writer.writerow(section)
    print(model)
    print(datetime.datetime.now())
    