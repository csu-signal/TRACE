import torch

BATCH_SIZE = 8  # increase / decrease according to GPU memeory
RESIZE_TO = 416  # resize the image for training and transforms
NUM_EPOCHS = 10  # number of epochs to train for
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training images and XML files directory
TRAIN_DIR = "data/weights-dataset/train"
# validation images and XML files directory
VALID_DIR = "data/weights-dataset/valid"

# classes: 0 index is reserved for background
CLASSES = ["__background__", 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = "outputs"
