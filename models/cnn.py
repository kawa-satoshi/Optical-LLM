import torch

from modules.layers import conv2d, linear


class AnalogCNN(torch.nn.Module):
    # ref: https://ex-ture.com/blog/2021/01/11/pytorch-cnn/

    def __init__(
        self,
        input_size: tuple[int],
        output_dim: int,
        in_bit: int,
        w_bit: int,
        out_bit: int,
        std: float,
        type: str = "affine",
    ):
        super().__init__()

        in_chans, h, w = input_size

        linear_layer = linear.choices[type]
        conv2d_layer = conv2d.choices[type]
        print(f"linear_layer: {linear_layer}")
        print(f"conv2d_layer: {conv2d_layer}")

        self.pool = torch.nn.MaxPool2d(2)
        self.activation = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)

        self.conv1 = conv2d_layer(
            in_channels=in_chans,
            out_channels=16,
            kernel_size=5,
            padding=2,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            std=std,
        )
        self.conv2 = conv2d_layer(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            padding=2,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            std=std,
        )
        self.fc = linear_layer(
            (h // 4) * (w // 4) * 32,
            output_dim,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            std=std,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
