# Camera Calibration

In order to save the camera calibration settings I updated the Azure SDK offline processor to include the camera calibration settings directly from Azure. 

 - See main.cpp line 97
 - Usage: (build the offline processor project in visual studio) 
    - build/Debug/offline_processor.exe videoPath outputPath


# Example Json Output

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

Example Python Code to read in Calibration Information can be found in the "azureOverlay" folder
Feel free to reach out to Hannah VanderHoeven with any questions (Hannah.VanderHoeven@colostate.edu)

