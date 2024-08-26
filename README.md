# TRACE

This repository allows for easily constructing and running complex multimodal demonstrations.

A demo is organized as a collection of “features”, each of which serves a specific purpose. Examples of features are body tracking, gaze, gesture, audio transcriptions, proposition extraction, common ground tracking, logging, and more.

All features specify an output “interface”, which is a class representing the data a feature will output. Features also specify zero or more input interfaces, which they require in order to calculate the output. For example, the Proposition feature has PropositionInterface as its output interface and TranscriptionInterface as its only input interface.

If a feature A needs input interface X, it can set another feature B with output interface X as a “dependency”, and the output of feature B will be automatically passed into feature A. The full demo is structured as a directed graph with features as vertices and edges between a feature and all of its dependencies. This framework allows for easily creating, modifying, and running any multimodal demo which can be organized into modular features.

This repository contains a python package called “mmdemo” that provides a “Demo” class to run a demo according to its dependency graph structure. This package also contains premade features used in our common ground tracking demo and a framework to easily create new features. Another package in this repository is “mmdemo-azure-kinect”, which provides features for interacting with Azure Kinect cameras and recordings (only availible on Windows). Finally, we have comprehensive tests to make sure all of the premade features and demo logic works as expected.

## Usage

## Setup Instructions

### Main package
Python 3.10 or higher is required if using conda because of [this unresolved issue](https://github.com/conda/conda/issues/10897). The conda environment can be created with `conda env create --file multimodalDemo.yaml`.

To update the enviroment using the most current yaml file, activate it and run `conda env update --file multimodalDemo.yaml --prune`

Install the package with `pip install -e .` from the root directory of the repo.

Download the following models from [here](https://colostate-my.sharepoint.com/:f:/g/personal/nkrishna_colostate_edu/EhYic6HBX7hFta6GjQIcb9gBxV_K0yYFhtHagiVyClr7gQ?e=W6Pm6I) and save at the given locations:

- `fasterrcnn-7-19-demo-finetuned.pth` ==> `mmdemo/features/objects/objectDetectionModels/best_model-objects.pth`
- `steroid_model/` ==> `mmdemo/features/proposition/data/prop_extraction_model/`
- `production_move_classifier.pt` ==> `mmdemo/features/move/production_move_classifier.pt`

### Azure Kinect features (optional, only for Windows)

See [mmdemo-azure-kinect/README.md](mmdemo-azure-kinect/README.md).

## Development

Every feature must inherit from `BaseFeature[T]`, where `T` is an output interface which inherits from `BaseInterface`. See `mmdemo/features/` for many examples of how to create new features.

Pytest is used for all of the tests in this project. Tests which require our own machine learning models are marked as "model_dependent" and can be executed with `pytest -m "model_dependent"`. These will likely not all pass. Other tests can be executed with `pytest -m "not model_dependent"`, and these should all pass if there are no bugs. To execute all tests at once, just run `pytest`.

The source code is formatted using `black` and `isort`. This can be set up to run automatically on commit by running `pre-commit install`.

Feel free to reach out to Hannah VanderHoeven with any questions: Hannah.VanderHoeven@colostate.edu
