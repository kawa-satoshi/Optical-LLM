import torch
from torch import Tensor
from torch.nn.common_types import _size_2_t

from modules import noise, norm, precision


class AnalogConv2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        noiser=None,
        normalizer=None,
        quantizer_in=None,
        quantizer_w=None,
        quantizer_out=None,
        analog_mode: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

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

            y = torch.nn.functional.conv2d(
                x_qn,
                w_qn,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            y_n = self.noiser(y)
            y_nn = self.normalizer(y_n)
            y_nnq = self.quantizer_out(y_nn)

            return y_nnq
        else:
            x_n = self.normalizer(x)
            w_n = self.normalizer(self.weight)
            y = torch.nn.functional.conv2d(
                x_n,
                w_n,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            y_n = self.normalizer(y)

            return y_n

    def set_analog_mode(self, value: bool):
        self.analog_mode = value


class AnalogVNNConv2d(AnalogConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        in_bit: int = 8,
        w_bit: int = 8,
        out_bit: int = 8,
        std: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            noiser=noise.GaussianNoise(std),
            normalizer=norm.Clamp(-1, 1),
            quantizer_in=precision.ReducePrecision(1 << (in_bit - 1)),
            quantizer_w=precision.ReducePrecision(1 << (w_bit - 1)),
            quantizer_out=precision.ReducePrecision(1 << (out_bit - 1)),
        )


class SymmetricQuantizeConv2d(AnalogConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        in_bit: int = 8,
        w_bit: int = 8,
        out_bit: int = 8,
        std: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            noiser=noise.GaussianNoise(std),
            normalizer=norm.DoNothing(),
            quantizer_in=precision.SymmetricQuantize(in_bit),
            quantizer_w=precision.SymmetricQuantize(w_bit),
            quantizer_out=precision.SymmetricQuantize(out_bit),
        )


class AffineQuantizeConv2d(AnalogConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        in_bit: int = 8,
        w_bit: int = 8,
        out_bit: int = 8,
        std: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            noiser=noise.GaussianNoise(std),
            normalizer=norm.DoNothing(),
            quantizer_in=precision.AffineQuantize(in_bit),
            quantizer_w=precision.AffineQuantize(w_bit),
            quantizer_out=precision.AffineQuantize(out_bit),
        )


choices = {
    "analogvnn": AnalogVNNConv2d,
    "symmetric": SymmetricQuantizeConv2d,
    "affine": AffineQuantizeConv2d,
}
