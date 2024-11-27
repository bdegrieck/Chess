import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Model.constants import ModelParams
from Model.model import ConvNet

params = ModelParams(
    channel_values=[12, 32, 64, 128],  # Example channel sizes for a 3-layer CNN
    num_classes=2,                 # Number of output classes (e.g., for CIFAR-10)
    num_layers=3,                   # Number of convolutional layers
    kernel_size=3,                  # Size of the convolution kernel
    stride=1,                       # Stride size for convolution
    padding=1,                      # Padding size
    x_size=8                       # Size of the input image (e.g., 32x32 for CIFAR-10)
)

model = ConvNet(params)




