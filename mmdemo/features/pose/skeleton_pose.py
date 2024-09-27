import torch.nn as nn


class SkeletonPoseClassifier(nn.Module):
    """
    Base model, input single body, binary output. Two feedforward layers.
    Note: label of a frame is a very strong predictor of the next, how to incorporate without risk?
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SkeletonPoseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
