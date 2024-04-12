import torch


class AnalogModuleBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.analog_mode = True
        self.enable_noise = True
