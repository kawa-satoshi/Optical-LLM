import torch

from .base import AnalogLayerBase
from .conv2d import AnalogConv2d
from .linear import AnalogLinear


class AnalogGELU(torch.nn.GELU):
    pass
