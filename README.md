# Setup instructions

## Python Environment
Python 3.10 or higher is required if using conda because of [this unresolved issue](https://github.com/conda/conda/issues/10897). A conda environment can be created with `conda env create --file multimodalDemo.yaml`.

For those using rosch, the "multimodalDemo" environment was created on the C drive for shared use, it can be activated using: `conda activate C:\ProgramData\anaconda3\envs\multimodalDemo`. It requires VSCode to run as admin to use the environment with the interpreter/debugger. If you prefer to install the environment on your local account follow the steps outlined in the README.

## Azure Kinect SDK

Both the [Azure Kinect SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#installation) and [Body Tracking SDK](https://learn.microsoft.com/en-us/azure/kinect-dk/body-sdk-download) are required and can be downloaded/installed for Windows from the linked websites. Use version 1.4.2 of the azure kinect sdk and version 1.1.2 of the body tracking sdk if possible.

Both of these libraries are installed in `C:\Program Files\` on rosch.

Once the installation is complete, open `azure_kinect_wrapper/setup.py` and ensure that `K4A_DIR` and `K4ABT_DIR` are set to the correct locations.

## Wrapper Library
Run `pip install ./azure_kinect_wrapper`.

## Download models
Download the following models from [here](https://colostate-my.sharepoint.com/:f:/g/personal/nkrishna_colostate_edu/EhYic6HBX7hFta6GjQIcb9gBxV_K0yYFhtHagiVyClr7gQ?e=W6Pm6I) and save at the given locations:
- `fasterrcnn-*.pth` ==> `featureModules/objects/objectDetectionModels/best_model-objects.pth`
- `prop_extraction_model/` ==> `featureModules/prop/data/prop_extraction_model/`
- `production_move_classifier.pt` ==> `featureModules/move/production_move_classifier.pt`

# Running the demo
In `demo.py`, make sure `os.add_dll_directory` points to the correct installation location of the Body Tracking SDK. Also change `azure_kinect.Playback(<path to mkv>)` to have a valid path to an mkv file. Finally, run `python demo.py`.

# Modular Feature Interface
- "featureModules/featureName" - contains all data/relevant files for a feature of interest and gets auto copied to the output folder each build
- IFeature
  - __init__ - initalize any models, setup code
  - __processFrame__ - runs each time a frame is processed

- note that the paths to the any models or loaded data needs to be realive to the location of the root directory of the repository.

# TODOs

- [x] remove old code
- [ ] Document `azure_kinect_wrapper`
- [x] integrate ASR
- [ ] multiprocessing for improved performance
- [ ] ensure all features work on multiple devices simultaneously (I was having problems with the gaze feature in particular).
- [x] get the demo working with actual cameras (implement `Camera::open_device`, `Camera::close_device`, `Camera::update_capture_handle` in `device.cpp`)

Feel free to reach out to Hannah VanderHoeven with any questions (Hannah.VanderHoeven@colostate.edu)
