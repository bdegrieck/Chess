import torch
import torch.nn as nn

from Model.constants import ModelParams


class ConvNet(nn.Module):

    def __init__(self, params: ModelParams):
        super().__init__()
        self.params = params
        self.convs = nn.ModuleList()

        channel_values = params.channel_values
        in_channels = channel_values[0]

        for out_channels in channel_values:
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=params.kernel_size,
                    stride=params.stride,
                    padding=params.padding
                )
            )
            in_channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        # Dynamically determine flattened size
        flattened_size = self._get_flattened_size()

        self.fc = nn.Linear(flattened_size, params.num_classes)
        self._initialize_weights()

    def _get_flattened_size(self):
        dummy_input = torch.zeros(1, self.params.channel_values[0], self.params.x_size, self.params.x_size)
        x = dummy_input
        for conv in self.convs:
            x = self.pool(self.relu(conv(x)))
        return x.numel()

    def _initialize_weights(self):
        for layer in self.convs:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        for conv in self.convs:
            x = self.pool(self.relu(conv(x)))
        x = torch.flatten(x, start_dim=1)
        scores = torch.sigmoid(self.fc(x))
        return scores
