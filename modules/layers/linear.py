import torch
from torch import Tensor

from modules import noise, norm, precision

from .base import *


class AnalogLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        noiser,
        normalizer,
        quantizer_in,
        quantizer_w,
        quantizer_out,
        analog_mode: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)

        self.normalizer = normalizer
        self.quantizer_in = quantizer_in
        self.quantizer_w = quantizer_w
        self.quantizer_out = quantizer_out
        self.noiser = noiser
        self.analog_mode = analog_mode

    def forward(self, x: Tensor) -> Tensor:
        if self.analog_mode:
            x_n = self.normalizer(x)
            w_n = self.normalizer(self.weight)

            x_q = self.quantizer_in(x_n)
            w_q = self.quantizer_w(w_n)

            x_qn = self.noiser(x_q)
            w_qn = self.noiser(w_q)

            y = torch.nn.functional.linear(x_qn, w_qn, bias=None)
            y_n = self.noiser(y)
            y_nn = self.normalizer(y_n)
            y_nnq = self.quantizer_out(y_nn)

            if self.bias is not None:
                y_f = y_nnq + self.bias
            else:
                y_f = y_nnq

            return y_f
        else:
            x_n = self.normalizer(x)
            w_n = self.normalizer(self.weight)
            y = torch.nn.functional.linear(x_n, w_n, None)
            y_n = self.normalizer(y)

            if self.bias is not None:
                y_f = y_n + self.bias
            else:
                y_f = y_n

            return y_f

    def set_analog_mode(self, value: bool):
        self.analog_mode = value


class SymmetricQuantizeLinear(AnalogLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        in_bit: int = 8,
        w_bit: int = 8,
        out_bit: int = 8,
        std: float = 0.0,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            noiser=noise.GaussianNoise(std),
            normalizer=norm.DoNothing(),
            quantizer_in=precision.SymmetricQuantize(in_bit),
            quantizer_w=precision.SymmetricQuantize(w_bit),
            quantizer_out=precision.SymmetricQuantize(out_bit),
        )


class AffineQuantizeLinear(AnalogLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        in_bit: int = 8,
        w_bit: int = 8,
        out_bit: int = 8,
        std: float = 0.0,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            noiser=noise.GaussianNoise(std),
            normalizer=norm.DoNothing(),
            quantizer_in=precision.AffineQuantize(in_bit),
            quantizer_w=precision.AffineQuantize(w_bit),
            quantizer_out=precision.AffineQuantize(out_bit),
        )


class AnalogVNNLinear(AnalogLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        in_bit: int = 8,
        w_bit: int = 8,
        out_bit: int = 8,
        std: float = 0.0,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            noiser=noise.GaussianNoise(std),
            normalizer=norm.Clamp(-1, 1),
            quantizer_in=precision.ReducePrecision(1 << (in_bit - 1)),
            quantizer_w=precision.ReducePrecision(1 << (w_bit - 1)),
            quantizer_out=precision.ReducePrecision(1 << (out_bit - 1)),
        )


class PatchverLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_bit: int = 8,
        w_bit: int = 8,
        out_bit: int = 8,
        std: float = 0.0,
        analog_mode: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=False)

        self.quantizer_in = precision.PatchverQuantize(in_bit)
        self.quantizer_w = precision.PatchverQuantize(w_bit)
        self.quantizer_out = precision.PatchverQuantize(out_bit)
        self.noiser = noise.GaussianNoise(std)
        self.analog_mode = analog_mode

    def forward(self, x: Tensor) -> Tensor:
        if self.analog_mode:
            s_x = x.max() - x.min()
            s_w = self.weight.max() - self.weight.min()

            x_r = x / s_x
            w_r = self.weight / s_w

            x_q = self.quantizer_in(x_r)
            w_q = self.quantizer_w(w_r)

            x_qn = self.noiser(x_q)
            w_qn = self.noiser(w_q)

            y = torch.nn.functional.linear(x_qn, w_qn, None)
            y_n = self.noiser(y)
            y_nn = y_n / y_n.abs().max()
            y_nnq = self.quantizer_out(y_nn) * s_x * s_w

            if self.bias is not None:
                # y_f = y_r + bias
                raise NotImplementedError("Currently bias is not supported.")
            else:
                y_f = y_nnq

            return y_f
        else:
            y_n = torch.nn.functional.linear(x, self.weight, None)

            if self.bias is not None:
                # y_f = y_r + bias
                raise NotImplementedError("Currently bias is not supported.")

            return y_n

    def set_analog_mode(self, value: bool):
        self.analog_mode = value


choices = {
    "analogvnn": AnalogVNNLinear,
    "affine": AffineQuantizeLinear,
    "symmetric": SymmetricQuantizeLinear,
    # "patchver": PatchverLinear,
}
