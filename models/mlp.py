import math
from typing import Tuple

import torch

from modules.layers import linear


class AnalogMLP(torch.nn.Module):
    def __init__(
        self,
        input_size: Tuple[int],
        output_dim: int,
        in_bit: int,
        w_bit: int,
        out_bit: int,
        std: float,
        hidden_dim: int = 512,
        type: str = "affine",
    ):
        super().__init__()

        linear_layer = linear.choices[type]
        print(f"linear_layer: {linear_layer}")

        self.flatten = torch.nn.Flatten(start_dim=1)
        self.fc1 = linear_layer(
            math.prod(list(input_size)),
            hidden_dim,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            std=std,
        )
        self.fc2 = linear_layer(
            hidden_dim,
            hidden_dim,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            std=std,
        )
        self.fc3 = linear_layer(
            hidden_dim,
            output_dim,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            std=std,
        )
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.fc3(x)
        return x
