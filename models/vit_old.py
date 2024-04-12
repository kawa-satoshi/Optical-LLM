import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from models.base import AnalogModuleBase
from modules.layers import conv2d, linear

# ref: https://github.com/NaotoNaka/ViT

# dataset hyperparam
imageWH = 32
channel = 3
imageBatch = 16
shuffle = True
num_workers = 4

# vit hyperparam
patchWH = 8
splitRow = imageWH // 8
splitCol = imageWH // 8
patchTotal = (imageWH // patchWH) ** 2  # (32 / 8)^2 = 16
patchVectorLen = channel * (patchWH**2)  # 3 * 64 = 192
embedVectorLen = int(patchVectorLen / 2)

# transformer layer hyperparam
head = 12
dim_feedforward = embedVectorLen
activation = "gelu"
layers = 12


class AnalogViT(AnalogModuleBase):
    def __init__(
        self,
        in_bit: int = 8,
        w_bit: int = 8,
        out_bit: int = 8,
        std: float = 0.5,
        type: str = "analogvnn",
    ):
        super().__init__()

        linear_layer = linear.choices[type]
        conv2d_layer = conv2d.choices[type]
        print(f"linear_layer: {linear_layer}")
        print(f"conv2d_layer: {conv2d_layer}")

        self.patchEmbedding = linear_layer(
            patchVectorLen,
            embedVectorLen,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            std=std,
        )
        self.cls = torch.nn.Parameter(
            torch.zeros(
                1,
                1,
                embedVectorLen,
            )
        )
        self.positionEmbedding = torch.nn.Parameter(
            torch.zeros(
                1,
                patchTotal + 1,
                embedVectorLen,
            )
        )
        encoderLayer = TransformerEncoderLayer(
            d_model=embedVectorLen,
            nhead=head,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformerEncoder = TransformerEncoder(encoderLayer, layers)
        self.mlpHead = linear_layer(
            embedVectorLen,
            10,
            in_bit=in_bit,
            w_bit=w_bit,
            out_bit=out_bit,
            std=std,
        )

    def patchify(self, img):
        horizontal = torch.stack(torch.chunk(img, splitRow, dim=2), dim=1)
        patches = torch.cat(torch.chunk(horizontal, splitCol, dim=4), dim=1)
        return patches

    def forward(self, x):
        x = self.patchify(x)
        x = torch.flatten(x, start_dim=2)
        x = self.patchEmbedding(x)
        clsToken = self.cls.repeat_interleave(x.shape[0], dim=0)
        x = torch.cat((clsToken, x), dim=1)
        x += self.positionEmbedding
        x = self.transformerEncoder(x)
        x = self.mlpHead(x[:, 0, :])
        return x
