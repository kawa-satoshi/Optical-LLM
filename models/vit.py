import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from modules.layers import AnalogLayerBase, linear

from .base import AnalogModuleBase


class AnalogEmbedding(AnalogLayerBase):
    def __init__(self, dim, num_patches, dropout=0.0, bitwidth: int = 5, leakage: float = 0.5):
        super().__init__(bitwidth=bitwidth, leakage=leakage)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        class_tokens = repeat(self.class_token, "1 1 d -> b 1 d", b=x.shape[0])
        x = torch.concat([class_tokens, x], dim=1)
        x = self.DA_AD(x)
        pos_embedding = self.DA_AD(self.pos_embedding)
        x += pos_embedding
        x = self.detection(x)
        x = self.DA_AD(x)
        x = self.dropout(x)
        return x


class AnalogFeedForward(AnalogLayerBase):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout=0.0,
        bitwidth: int = 5,
        leakage: float = 0.5,
        linear_layer: AnalogLayerBase = None,
    ):
        super().__init__(bitwidth=bitwidth, leakage=leakage)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            linear_layer(dim, hidden_dim, bitwidth=bitwidth, leakage=leakage),
            nn.GELU(),
            nn.Dropout(dropout),
            linear_layer(hidden_dim, dim, bitwidth=bitwidth, leakage=leakage),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AnalogMultiHeadAttention(AnalogLayerBase):
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
        bitwidth: int = 5,
        leakage: float = 0.5,
        linear_layer: AnalogLayerBase = None,
    ):
        super().__init__(bitwidth=bitwidth, leakage=leakage)
        self.dim_head = dim // num_heads
        self.heads = num_heads

        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = linear_layer(dim, dim * 3, bias=False, bitwidth=bitwidth, leakage=leakage)
        self.after_softmax_binarize = linear.BiTLayerBase(bitwidth=1)
        # For alpha and beta after softmax

    def forward(self, x):
        x = self.norm(x)

        # Compute Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # Compute attention weight
        q = self.DA_AD(q)
        k = self.DA_AD(k)
        x = torch.matmul(q, k.transpose(-1, -2)) * (self.dim_head**-0.5)
        x = self.detection(x)
        x = self.DA_AD(x)
        x = self.softmax(x)
        x = self.dropout(x)
        if isinstance(self.to_qkv, linear.BiTLayerBase):
            x = self.after_softmax_binarize.DA_AD(x, after_activation=True)
        else:
            x = self.DA_AD(x)
        v = self.DA_AD(v)
        out = torch.matmul(x, v)
        out = self.detection(out)
        out = self.DA_AD(out)
        out = rearrange(out, "b h n d -> b n (h d)")

        return out


class AnalogTransformer(AnalogLayerBase):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        ffn_dim,
        dropout=0.0,
        bitwidth: int = 5,
        leakage: float = 0.5,
        linear_layer: AnalogLayerBase = None,
    ):
        super().__init__(bitwidth=bitwidth, leakage=leakage)
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.Sequential()
        for _ in range(depth):
            self.layers.append(
                AnalogMultiHeadAttention(dim, num_heads=num_heads, dropout=dropout, linear_layer=linear_layer)
            )
            self.layers.append(AnalogFeedForward(dim, ffn_dim, dropout=dropout, linear_layer=linear_layer))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return self.norm(x)


class AnalogViT(AnalogModuleBase):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        num_heads: int,
        ffn_dim: int,
        num_channels: int = 3,
        dropout: float = 0.0,
        in_bit: int = 8,
        w_bit: int = 8,
        out_bit: int = 8,
        std: float = 0.5,
        type: str = "analogvnn",
    ):
        super().__init__()

        linear_layer = linear.choices[type]
        print(f"linear_layer: {linear_layer}")

        self.depth = depth
        num_patches = (image_size // patch_size) ** 2
        patch_dim = num_channels * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            linear_layer(
                patch_dim,
                dim,
                in_bit=in_bit,
                w_bit=w_bit,
                out_bit=out_bit,
                std=std,
            ),
            nn.LayerNorm(dim),
        )
        self.embedding = AnalogEmbedding(dim, num_patches, bitwidth=bitwidth, leakage=leakage)
        self.transformer = AnalogTransformer(
            dim, depth, num_heads, ffn_dim, dropout, bitwidth=bitwidth, leakage=leakage, linear_layer=linear_layer
        )
        self.mlp_head = linear_layer(dim, num_classes, bitwidth=bitwidth, leakage=leakage)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x
