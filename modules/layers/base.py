import torch
from torch import Tensor

from models.base import AnalogModuleBase
from modules import noise, norm, precision


class AnalogLayerBase(AnalogModuleBase):
    def __init__(
        self,
        noise_class: noise.NoiseBase = noise.GaussianNoise,
        precision_class: precision.PrecisionBase = precision.ReducePrecision,
        norm_class: norm.NormBase = norm.Clamp,
        bitwidth: int = 5,
        leakage: float = 0.5,
        AD_DA_bitwidth: int = None,
        super_init: bool = True,
    ) -> None:
        if super_init:
            super().__init__()
        self.noise_class: noise.NoiseBase = noise_class
        self.precision_class: precision.PrecisionBase = precision_class
        self.norm_class: norm.NormBase = norm_class
        self.bitwidth = bitwidth
        self.precision = 2 ** (bitwidth - 1)
        self.leakage = leakage
        self.AD_DA_bitwidth = AD_DA_bitwidth if AD_DA_bitwidth is not None else bitwidth
        self.AD_DA_precision = 2 ** (self.AD_DA_bitwidth - 1)


class NaiveSmoothQuantLayerBase(AnalogLayerBase):
    def DA_AD(self, x, input=False, **kwargs):
        if not input:
            # Normal quantization for weight and output
            return super().DA_AD(x)
        # Use per-channel quantization instead of normal one
        if self.analog_mode:
            x = precision.PerChannelReducePrecision.apply(x, self.AD_DA_precision)
        return x


class BiTLayerBase(AnalogLayerBase):
    class STERound(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.round()

        @staticmethod
        def backward(ctx, g):
            return g

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bitwidth != 1 or self.AD_DA_bitwidth != 1:
            raise ValueError("Bitwidth has to be 1 in BiTLinear.")
        # Set alpha and beta as trainable parameters
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(0.0))

    def non_negative_binarize(self, x: Tensor):
        return self.alpha * self.STERound.apply(torch.clip((x - self.beta) / self.alpha, 0.0, 1.0))

    def DA_AD(self, x, after_activation=False, **kwargs):
        if not after_activation:
            return super().DA_AD(x)
        # Use 0-1 binarization instead of normal one
        if self.analog_mode:
            x = self.non_negative_binarize(x)
        return x
