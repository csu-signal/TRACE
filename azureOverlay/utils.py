import numpy as np

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