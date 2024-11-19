import torch
import torch.nn as nn

from Model.constants import ModelParams


class ConvNet(nn.Module):

    def __init__(self, params: ModelParams):
        super().__init__()
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.params = params

        in_channels = params.channel_values[0]
        channel_vals = params.channel_values

        for out_channels in channel_vals:
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

        conv_output_size = params.x_size
        for _ in range(params.num_layers):
            # Apply convolution
            conv_output_size = (
                    (conv_output_size + 2 * params.padding - params.kernel_size) // params.stride + 1
            )
            conv_output_size = conv_output_size // 2
        flattened_size = channel_vals[-1] * (conv_output_size // 2) ** 2
        self.fc = nn.Linear(flattened_size, params.num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        for conv in self.convs:
            x = self.relu(conv(x))
            x = nn.functional.max_pool2d(x, kernel_size=2)
        x = torch.flatten(x, start_dim=1)
        scores = self.fc(x)
        return scores
