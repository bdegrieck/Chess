from pydantic import BaseModel


class ModelParams(BaseModel):
    channel_values: list[int]
    num_classes: int
    num_layers: int
    kernel_size: int
    stride: int
    padding: int
    x_size: int
