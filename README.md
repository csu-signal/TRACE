# TRACE

This repository allows for easily constructing and running complex multimodal demonstrations.

A demo is organized as a collection of "features", each of which serves a specific purpose. Examples of features are body tracking, gaze, gesture, audio transcriptions, proposition extraction, common ground tracking, logging, and more.

All features specify an output "interface", which is a class representing the data a feature will output. Features also specify zero or more input interfaces, which they require in order to calculate the output. For example, the Proposition feature has PropositionInterface as its output interface and TranscriptionInterface as its only input interface.

If a feature A needs input interface X, it can set another feature B with output interface X as a "dependency", and the output of feature B will be automatically passed into feature A. The full demo is structured as a directed graph with features as vertices and edges between a feature and all of its dependencies. This framework allows for easily creating, modifying, and running any multimodal demo which can be organized into modular features.

This repository contains a python package called "mmdemo" that provides a "Demo" class to run a demo according to its dependency graph structure. This package also contains premade features used in our common ground tracking demo and a framework to easily create new features. Another package in this repository is "mmdemo-azure-kinect", which provides features for interacting with Azure Kinect cameras and recordings (only availible on Windows). Finally, we have comprehensive tests to make sure all of the premade features and demo logic works as expected.

## Example Usage

Any number of "target" features can be given to the Demo constructor. These targets and their dependencies will be evaluated such that all dependencies of a feature are done evaluating before the feature itself evaluates. The following script will perform common ground tracking using microphone input.

```python
from mmdemo.demo import Demo
from mmdemo.features import ( CommonGroundTracking, Log,
    MicAudio, Move, Proposition, VADUtteranceBuilder, 
    WhisperTranscription )

if __name__ == "__main__":
    mic = MicAudio(device_id=6)
    utterances = VADUtteranceBuilder(mic)
    transcription = WhisperTranscription(utterances)
    props = Proposition(transcription)
    moves = Move(transcription, utterances)
    cgt = CommonGroundTracking(moves, props)

    demo = Demo(targets=[Log(transcription, props, moves, cgt, stdout=True)])
    demo.run()
```

Dependency graph visualizations can also be generated automatically by calling `demo.show_dependency_graph()`, which can be useful for making sure the demo is structured correctly. In the example above, this would create the following image.
![dependency graph](images/dependency_graph.png)

## Setup Instructions

### Main package
Python 3.10 or higher is required if using conda because of [this unresolved issue](https://github.com/conda/conda/issues/10897). The conda environment can be created with `conda env create --file multimodalDemo.yaml`.

Install the package with `pip install -e .` from the root directory of the repo.

Download the following models from [here](https://colostate-my.sharepoint.com/:f:/g/personal/nkrishna_colostate_edu/EhYic6HBX7hFta6GjQIcb9gBxV_K0yYFhtHagiVyClr7gQ?e=W6Pm6I) and save at the given locations:

- `fasterrcnn-7-19-demo-finetuned.pth` ==> `mmdemo/features/objects/objectDetectionModels/best_model-objects.pth`
- `steroid_model/` ==> `mmdemo/features/proposition/data/prop_extraction_model/`
- `production_move_classifier.pt` ==> `mmdemo/features/move/production_move_classifier.pt`

### Azure Kinect features (optional, only for Windows)

See [mmdemo-azure-kinect/README.md](mmdemo-azure-kinect/README.md).

## Development

### Environment

After setting up the environment by following the instructions above, run `pre-commit install` to set up formatters to run automatically on commit. If the conda environment file changes, update the environment by running `conda env update --file multimodalDemo.yaml --prune`.

### Creating new features
Every feature must inherit from `BaseFeature[T]`, where `T` is an output interface which inherits from `BaseInterface`. The required methods are documented in [mmdemo/base_feature.py](mmdemo/base_feature.py). For example, if we wanted to create a feature which takes a color image as input and outputs a predicted depth image, we would do something along the lines of the following:
```python
@final
class DepthPredictor(BaseFeature[DepthImageInterface]):
    def __init__(self, color: BaseFeature[ColorImageInterface]):
        super().__init__(color)

    def initialize(self):
        # Initialize model
        pass

    def get_output(self, color: ColorImageInterface) -> DepthImageInterface | None:
        if not color.is_new():
            return None
        # evaluate model on color.frame
        pred = ...
        return DepthImageInterface(frame=pred, frame_count=color.frame_count)
```
This feature could now seamlessly be used as a dependency to any feature that requires a depth image as input. See `mmdemo/features/` for examples of how existing features are implemented. Also note that a feature should never directly modify any of its input interfaces or dependent features. This breaks the modularity of the program and could cause other features to break in unexpected ways.

### Testing

Pytest is used for all of the tests in this project. Tests which require our own machine learning models are marked as "model_dependent" and can be executed with `pytest -m "model_dependent"`. These will likely not all pass. Other tests can be executed with `pytest -m "not model_dependent"`, and these should all pass if there are no bugs. To execute all tests at once, just run `pytest`.

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

Feel free to reach out to Hannah VanderHoeven (Hannah.VanderHoeven@colostate.edu) with any questions.
