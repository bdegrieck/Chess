import torch

from Model.constants import ModelParams
from Model.model import ConvNet

dtype = torch.float32

class TestModel:

    def test_convmodel(self):
        x = torch.zeros((64, 12, 8, 8), dtype=torch.float32)
        params = ModelParams(
            channel_values=[12, 32, 64],  # Start with 12 input channels, increase as features get more complex
            num_classes=2,  # Binary classification
            num_layers=2,  # 3 convolutional layers
            kernel_size=3,  # Common choice for CNNs
            stride=1,  # Preserve spatial resolution
            padding=1,  # Maintain dimensions with 3x3 kernel
            x_size=8  # Chess board size is 8x8
        )
        model = ConvNet(params=params)
        scores = model.forward(x)
        assert list(scores.size()) == [64, params.num_classes]
