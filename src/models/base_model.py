import torchvision
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def setup_loss(self):
        """
        Setup the loss function.
        This method should be overridden by subclasses if they require any custom loss functions.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def setup(self):
        """
        Setup the model.
        This method should be overridden by subclasses if they require any custom setup.
        """
        self.setup_loss()

    def forward(self, x):
        """
        Forward pass through the model.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError
