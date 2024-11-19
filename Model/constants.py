from pydantic import BaseModel


class ModelParams(BaseModel):
    """
    channel_values: a list of channel sizes
    num_classes: number of output classification classes
    num_layers: number of convolution layers
    kernel_size: size of the kernel
    stride: amount of steps the padding moves across the image
    padding: adding extra layers of zeros around an image
    x_size: size of the image
    """
    channel_values: list[int]
    num_classes: int
    num_layers: int
    kernel_size: int
    stride: int
    padding: int
    x_size: int
