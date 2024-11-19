import torch

from Model.constants import ModelParams
from Model.model import ConvNet

dtype = torch.float32

class TestModel:

    def test_convmodel(self):
        x = torch.zeros((64, 3, 32, 32), dtype=dtype)
        params = ModelParams(
            channel_values=[3, 12, 8],
            num_classes=10,
            num_layers=2,
            kernel_size=3,
            stride=1,
            padding=1,
            x_size=32
        )
        model = ConvNet(params=params)
        scores = model.forward(x)
        assert list(scores.size()) == [64, 10]
