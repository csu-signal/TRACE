# TRACE

TODO: description here

# Setup
## Main package
Python 3.10 or higher is required if using conda because of [this unresolved issue](https://github.com/conda/conda/issues/10897). The conda environment can be created with `conda env create --file multimodalDemo.yaml`.

To update the enviroment using the most current yaml file, activate it and run `conda env update --file multimodalDemo.yaml --prune`

Install the package with `pip install -e .` from the root directory of the repo.

Download the following models from [here](https://colostate-my.sharepoint.com/:f:/g/personal/nkrishna_colostate_edu/EhYic6HBX7hFta6GjQIcb9gBxV_K0yYFhtHagiVyClr7gQ?e=W6Pm6I) and save at the given locations:

- `fasterrcnn-7-19-demo-finetuned.pth` ==> `mmdemo/features/objects/objectDetectionModels/best_model-objects.pth`
- `steroid_model/` ==> `mmdemo/features/proposition/data/prop_extraction_model/`
- `production_move_classifier.pt` ==> `mmdemo/features/move/production_move_classifier.pt`

## Azure Kinect features (optional, only for Windows)

Both the [Azure Kinect SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#installation) and [Body Tracking SDK](https://learn.microsoft.com/en-us/azure/kinect-dk/body-sdk-download) are required and can be downloaded/installed for Windows from the linked websites. Use version 1.4.2 of the azure kinect sdk and version 1.1.2 of the body tracking sdk if possible.

Once the installation is complete, open `mmdemo-azure-kinect\azure_kinect_config.py` and ensure that `K4A_DIR`, `K4ABT_DIR`, and `K4A_DLL_DIR` are set to the correct locations.

Run `pip install .\mmdemo-azure-kinect` from the root directory of the repo.


## Development

The source code is formatted using `black` and `isort`. This can be set up to run automatically by running `pre-commit install`.

TODO: testing, `pytest tests/ -m "not model_dependent"`

Feel free to reach out to Hannah VanderHoeven with any questions: Hannah.VanderHoeven@colostate.edu
