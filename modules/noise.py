import torch
from torch import Tensor


class NoiseBase(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class GaussianNoiseInner(NoiseBase):
    @staticmethod
    def forward(ctx, input: Tensor, std: float) -> Tensor:
        # GaussianNoise.forward
        return torch.normal(mean=input, std=std)


class GaussianNoise(torch.nn.Module):
    def __init__(self, std: float):
        super().__init__()

        self.std = std

    def forward(self, x):
        return GaussianNoiseInner.apply(x, self.std)

    def extra_repr(self) -> str:
        return f"std={self.std}"
