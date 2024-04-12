import torch

from modules.layers import conv2d, linear, AnalogLinear, AnalogConv2d


def create_analog_linear(
    target: torch.nn.Linear,
    in_bit: int,
    w_bit: int,
    out_bit: int,
    std: float,
    type: str,
) -> None:
    linear_layer = linear.choices[type]

    new_layer = linear_layer(
        in_features=target.in_features,
        out_features=target.out_features,
        bias=target.bias is not None,
        in_bit=in_bit,
        w_bit=w_bit,
        out_bit=out_bit,
        std=std,
    )
    new_layer.weight.data = target.weight.data.clone()
    new_layer.weight.requires_grad = target.weight.requires_grad

    if target.bias is not None:
        new_layer.bias.data = target.bias.data.clone()
        new_layer.bias.requires_grad = target.bias.requires_grad
    return new_layer


def create_analog_conv2d(
    target: torch.nn.Conv2d,
    in_bit: int,
    w_bit: int,
    out_bit: int,
    std: float,
    type: str,
) -> None:
    conv2d_layer = conv2d.choices[type]

    new_layer = conv2d_layer(
        in_channels=target.in_channels,
        out_channels=target.out_channels,
        kernel_size=target.kernel_size,
        stride=target.stride,
        padding=target.padding,
        dilation=target.dilation,
        groups=target.groups,
        bias=target.bias is not None,
        in_bit=in_bit,
        w_bit=w_bit,
        out_bit=out_bit,
        std=std,
    )
    new_layer.weight.data = target.weight.data.clone()
    new_layer.weight.requires_grad = target.weight.requires_grad

    if target.bias is not None:
        new_layer.bias.data = target.bias.data.clone()
        new_layer.bias.requires_grad = target.bias.requires_grad
    return new_layer


def analog_convert(
    model,
    in_bit: int,
    w_bit: int,
    out_bit: int,
    std: float,
    targets=None,
    convert_linear=True,
    convert_conv2d=True,
    type: str = "affine",
    verbose: bool = False,
) -> None:
    """
    If the target is not None, replace all layers with names contained in the target.
    Otherwise, replace all layers.

    When analog layer is found, this function overrides params of the layer with specified params.
    If you want to turn off this behavier, just comment out or divide this function.
    """

    for name, module in model.named_modules():
        new_layer = None
        if convert_linear and isinstance(module, torch.nn.Linear):
            if targets is not None and not any([target in name for target in targets]):
                if verbose:
                    print(f"ignore: {name}")
                continue

            new_layer = create_analog_linear(
                module,
                in_bit,
                w_bit,
                out_bit,
                std,
                type,
            )
        elif convert_conv2d and isinstance(module, torch.nn.Conv2d):
            if targets is not None and not any([target in name for target in targets]):
                if verbose:
                    print(f"ignore: {name}")
                continue

            new_layer = create_analog_conv2d(
                module,
                in_bit,
                w_bit,
                out_bit,
                std,
                type,
            )
        else:
            if verbose:
                print(f"skip: {name}")
            continue

        target = model
        path = name.split(".")
        for next in path[:-1]:
            target = getattr(target, next)
        setattr(target, path[-1], new_layer)
        if verbose:
            print(f"replace: {name}")
