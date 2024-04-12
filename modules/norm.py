import torch
from torch import Tensor


class NormBase(torch.autograd.Function):
    pass


class DoNothingInner(NormBase):
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DoNothing(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return DoNothingInner.apply(x)


class ClampInner(NormBase):
    @staticmethod
    def forward(ctx, input: Tensor, min=-1, max=1) -> Tensor:
        mask = ((min <= input) * (input <= max)).type(torch.float)
        ctx.save_for_backward(mask)

        return torch.clamp(input, min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]

        return grad_output * mask, None, None


class Clamp(torch.nn.Module):
    def __init__(self, min=-1, max=+1):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return ClampInner.apply(x, self.min, self.max)
