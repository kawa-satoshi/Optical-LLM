import torch
from torch import Tensor


def reduce_precision(input: Tensor, precision: int, divide: float = 0.5):
    if precision <= 1:
        raise NotImplementedError

    g = input * precision
    f = torch.sign(g) * torch.maximum(torch.floor(torch.abs(g)), torch.ceil(torch.abs(g) - divide)) * (1 / precision)
    return f


class PrecisionBase(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class ReducePrecisionInner(PrecisionBase):
    @staticmethod
    def forward(ctx, input: Tensor, precision: int, divide: float = 0.5) -> Tensor:
        input = reduce_precision(input, precision, divide)

        # assert len(torch.unique(input)) <= 2 * precision + 1

        return input


class ReducePrecision(torch.nn.Module):
    def __init__(self, precision: int, devide: float = 0.5):
        super().__init__()

        self.precision = precision
        self.devide = devide

    def forward(self, x):
        return ReducePrecisionInner.apply(x, self.precision, self.devide)


# class PerChannelReducePrecision(PrecisionBase):
#     @staticmethod
#     def forward(ctx, input: Tensor, precision: int, divide: float = 0.5) -> Tensor:
#         x = input
#         max = x.abs().max(dim=1, keepdim=True).values  # TODO: Implement for conv2d
#         quantized = (x * precision / max).round()
#         return quantized * max / precision


class SymmetricQuantizeInner(PrecisionBase):
    @staticmethod
    def forward(ctx, input: Tensor, bitwidth: int) -> Tensor:
        if bitwidth <= 1:
            raise NotImplementedError

        s = input.abs().max()
        if s == 0:
            return input

        r = 2**bitwidth
        input = torch.round(input / s * (r / 2 - 1))  # [-(r / 2 - 1), r / 2 - 1]
        input = input / (r / 2 - 1) * s  # [-s, s]

        # assert len(torch.unique(input)) <= r

        return input


class SymmetricQuantize(torch.nn.Module):
    def __init__(self, bitwidth: int):
        super().__init__()

        self.bitwidth = bitwidth

    def forward(self, x):
        return SymmetricQuantizeInner.apply(x, self.bitwidth)

    def extra_repr(self) -> str:
        return f"bitwidth={self.bitwidth}"


class AffineQuantizeInner(PrecisionBase):
    @staticmethod
    def forward(ctx, input: Tensor, bitwidth: int) -> Tensor:
        if bitwidth <= 1:
            raise NotImplementedError

        alpha, beta = input.max(), input.min()
        if alpha == beta:
            return input

        r = 2**bitwidth
        s = (r - 1) / (alpha - beta)
        z = -beta * s - r / 2

        input = torch.round(s * input + z)  # quantize
        input = (input - z) / s  # dequantize
        return input


class AffineQuantize(torch.nn.Module):
    def __init__(self, bitwidth: int):
        super().__init__()

        self.bitwidth = bitwidth

    def forward(self, x):
        return AffineQuantizeInner.apply(x, self.bitwidth)

    def extra_repr(self) -> str:
        return f"bitwidth={self.bitwidth}"


class PatchverQuantizeInner(PrecisionBase):
    @staticmethod
    def forward(ctx, input: Tensor, bitwidth: int) -> Tensor:
        if bitwidth <= 1:
            raise NotImplementedError

        r = 2**bitwidth
        mask = ((-r / 2 + 1 <= input * r) * (input * r <= r / 2)).type(torch.float)
        ctx.save_for_backward(mask)

        input = torch.clamp(input * r, -r / 2 + 1, r / 2)
        input = torch.round(input) / r

        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]

        return grad_output * mask, None


class PatchverQuantize(torch.nn.Module):
    def __init__(self, bitwidth: int):
        super().__init__()

        self.bitwidth = bitwidth

    def forward(self, x):
        return PatchverQuantizeInner.apply(x, self.bitwidth)

    def extra_repr(self) -> str:
        return f"bitwidth={self.bitwidth}"
