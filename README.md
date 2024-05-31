# Camera Calibration

In order to save the camera calibration settings I updated the Azure SDK offline processor to include the camera calibration settings directly from Azure. 

# Python Package Versions

- "python" 3.9.12
- "argparse" 1.1
- "cv2" (opencv) 4.5.1 
- "numpy" 1.21.5
- "json" 2.0.9
- "mediapipe" 0.8.9.1

# Demo Setup 

- see the "Code Help" folder for video demos and additional Demo setup notes.
- I recommend copying the "offline_processor.vcxproj" from the code help to the "offline_processor" file and then editing it following these steps (it won't be committed if any changes are made so the build settings won't get wiped out for each setup)

- Camera_Calibration Repo: https://github.com/Blanchard-lab/Camera_Calibration
- Sanity Check Open Camera Script: https://github.com/Blanchard-lab/Camera_Calibration/blob/main/azureOverlay/checkCameraUtil.py 

## Visual Studio Install
- Make sure the C++ compiler for visual studio is installed
	 
## Install OpenCV
- https://github.com/opencv/opencv
- https://github.com/opencv/opencv/releases/tag/4.5.5 
- Place the “opencv” folder one directory above the “Camera_Calibration” folder (the build is setup to look for it there)
- Add opencv to the path variable (example from my configuration):      
- - “C:\Users\Devin\Desktop\GitHub\opencv\build\x64\vc16\bin”

## CONDA
- Make sure conda is installed and the path to the exe is added to the path:
- Create a virtual environment using the “handTrackingEnvironment.yaml” (In the repository)

## Linking Python
- Set the include and libs to the directory of the python libs relative to the virtual environment you are using 
- Add the virtual environment location to the path variable
- If python##_d.lib doesn’t exist copy python##.lib and rename it

## Linking NUMPY
- Add the path to numpy from the virtual environment to the includes:
- C:\Users\vanderh\Anaconda3\envs\handTrackingEnvironment\Lib\site-packages\numpy\core\include

# Build Instructions

- if conda doesn't work on your account in powershell (specifically if activate isn't a valid command) try calling "conda init powershell" and run activate again
- open a powershell at "C:\GitHub\Camera_Calibration\offline_processor\build\bin\Release"
- activate hand tracking enviroment "conda activate C:\ProgramData\anaconda3\envs\handTrackingEnvironment"
- run ".\offline_processor.exe"

- if the torch device isn't the GPU (cuda), you might need to uninstall and reinstall torch and torchvision in the virtual env for windows
- "pip uninstall torchvision"
- "pip uninstall torch"
- "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"

# Object Detection Model
- reach out to Jack Fitzgerald or Hannah VanderHoeven for access to the object detection model
- hannah.vanderhoeven@colostate.edu
- jack.fitzgerald@colostate.edu

- https://colostate-my.sharepoint.com/:u:/g/personal/jhfitzg_colostate_edu/ERqPMvinOUJGr4lSLt1oqtYBpv0fwbGRrc15hV6uHtFnCA?e=F1SIoO

# ASR detection script
- Additional packages are needed (install using pip): sounddevice, espnet, espnet_model_zoo

# Modular Feature Interface
- "featureModules/featureName" - contains all data/relevant files for a feature of interest and gets auto copied to the output folder each build
- IFeature
  - __init__ - initalize any models, setup code
  - __processFrame__ - runs each time a frame is processed

- note that the paths to the any models or loaded data needs to be realive to the location of "offline_professor.exe"

# Example Additional Json Output

```
     "camera_calibration": {
        "cx": 962.8074340820313,
        "cy": 550.3942260742188,
        "fx": 911.5870361328125,
        "fy": 911.545166015625,
        "k1": 0.42810821533203125,
        "k2": -2.7314584255218506,
        "k3": 1.6649699211120605,
        "k4": 0.3093422055244446,
        "k5": -2.5492746829986572,
        "k6": 1.5847551822662354,
        "p1": 0.0004034260637126863,
        "p2": -0.00010884667426580563,
        "rotation": [
            0.9999893307685852,
            0.0016191748436540365,
            -0.004320160020142794,
            -0.001163218286819756,
            0.9946256279945374,
            0.103530153632164,
            0.004464575555175543,
            -0.10352402180433273,
            0.9946169257164001
        ],
        "translation": [
            -32.072898864746094,
            -2.0814547538757324,
            3.8745241165161133
        ]
    }
```

Example Python Code to read in Calibration Information can be found in the "azureOverlay" folder.

Feel free to reach out to Hannah VanderHoeven with any questions (Hannah.VanderHoeven@colostate.edu)

