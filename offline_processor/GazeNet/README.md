# GazeNet

This is a repository for a pre-trained model to figure out where a person looking at in the image. You can choose to use all the pipeline here to get the result, which is like the images in Demo_res, or use your own face detector. You can see our demos in Demo_res.

## Requiremnets
python >= 3.4.1  
tensorflow  
mtcnn 

## Run the new demo
Here we can run the python file Test.py to get the results.  
### Change Images in Demo_img
Firstly, you can change the umages in Demo_img, and delete the images in Demo_res. Then you will get your result images in Demo_img.

### Change the Dictionary
Alternatively, you can build your own dictionary, and modify the IMG_DIR (dictionary of the imput images) and OUT_DIR (output path) in the line 9 and 10 of Test.py.

## Use other Face Detector
If you want to use your own face ditector, you can firstly get the following input:
filenames  
images:256 * 256  
faces:32 * 32  
heads:cordinate  

Then, import the Test.py, and use the following code to run it:

    model = load_model()
    preds = predict_gaze(model, images, faces, heads)
    visualize(filenames, heads, preds)
