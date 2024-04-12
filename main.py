from argparse import Namespace

import mlflow
import torch
import yaml
import timm
from timm.data import create_transform, resolve_data_config
from torch.utils.tensorboard import SummaryWriter

import argparser
import datasets
from models import cnn, converter, mlp, transfer
from trainer import run


def main(args):
    print("=== args ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============")

    if not args.w_bit:
        args.w_bit = args.in_bit
    if not args.out_bit:
        args.out_bit = args.in_bit

    no_mlflow = False
    if args.mlflow_url:
        mlflow.set_tracking_uri(args.mlflow_url)

        if args.mlflow_experiment_name:
            experiment = mlflow.get_experiment_by_name(args.mlflow_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(name=args.mlflow_experiment_name)
            else:
                experiment_id = experiment.experiment_id
        else:
            raise RuntimeError("MLFlow Remote URL is given, but Experiment Name is NOT given.")
    else:
        print("[NO MLFLOW MODE]")
        no_mlflow = True

    model_full = "timm-" + args.timm_model_name if args.model == "timm" else args.model
    run_name = f"{model_full}-{args.quantization_type}-{args.dataset}-{args.in_bit}bit-{args.std}std-{args.count}"
    print("START:", run_name)

    if not no_mlflow:
        mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment_id,
        )

    del args.mlflow_url
    del args.mlflow_experiment_name

    if not no_mlflow:
        mlflow.log_params(vars(args))

    device = torch.device(args.device)
    print("Device: {}".format(device))

    dataset = datasets.choices[args.dataset]
    if args.model == "timm":
        if args.dataset == "imagenet1k":
            model = timm.create_model(
                args.timm_model_name,
                pretrained=True,
            )
        else:
            model = transfer.TransferModel(
                model_name=args.timm_model_name,
                num_classes=dataset.num_labels,
                in_chans=dataset.image_size[0],
            )

        converter.analog_convert(
            model,
            args.in_bit,
            args.w_bit,
            args.out_bit,
            args.std,
            convert_linear=True,
            convert_conv2d=True,
            type=args.quantization_type,
        )

        if args.dataset != "imagenet1k":
            model.set_backbone_requires_grad(False)
    elif args.model == "mlp":
        model = mlp.AnalogMLP(
            input_size=dataset.image_size,
            output_dim=dataset.num_labels,
            in_bit=args.in_bit,
            w_bit=args.w_bit,
            out_bit=args.out_bit,
            std=args.std,
            hidden_dim=args.hidden_dim,
            type=args.quantization_type,
        )
    elif args.model == "cnn":
        model = cnn.AnalogCNN(
            input_size=dataset.image_size,
            output_dim=dataset.num_labels,
            in_bit=args.in_bit,
            w_bit=args.w_bit,
            out_bit=args.out_bit,
            std=args.std,
            type=args.quantization_type,
        )
    elif args.model == "mlflow" and not no_mlflow:
        model = mlflow.pytorch.load_model(args.mlflow_model_url)

        # 元モデルが既に置き換え済みだと convert_linear/convert_conv2d が機能しないことに注意
        converter.analog_convert(
            model,
            args.in_bit,
            args.w_bit,
            args.out_bit,
            args.std,
            convert_linear=True,
            convert_conv2d=True,
            type=args.quantization_type,
        )
    else:
        print(f"Invalid args.model: {args.model}")
        return

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.to(device)

    print(model)

    if args.model == "timm" or (args.model == "mlflow" and "timm" in args.mlflow_model_url):
        config = resolve_data_config(model=model)
    else:
        config = resolve_data_config({"input_size": dataset.image_size})
    in_chans = dataset.image_size[0]
    config["input_size"] = (in_chans, *config["input_size"][1:])
    config["mean"] = config["mean"][:in_chans]
    config["std"] = config["std"][:in_chans]
    transform = create_transform(**config)

    if args.dataset == "imagenet1k":
        ds = dataset.dataset(root="/home/takuya.tamori/datasets/imagenet/val/", transform=transform)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    else:
        trainset = dataset.dataset(root="./input", train=True, download=True, transform=transform)
        testset = dataset.dataset(root="./input", train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=0)

    if args.data_parallel and "cuda" in device.type:
        model = torch.nn.DataParallel(model)

    if not no_mlflow:
        mlflow.pytorch.log_model(
            model,
            f"initial-{run_name}",
        )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = None if args.log_dir is None else SummaryWriter(log_dir=args.log_dir)

    if args.saved_model is not None:
        model.load_state_dict(torch.load(args.saved_model + ".model"), strict=False)
        # optimizer.load_state_dict(torch.load(args.saved_model + '.optim'))

    run(
        run_name,
        device,
        model,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        train_mode=args.train_mode,
        epochs=args.epochs,
        save_interval=args.save_interval,
        writer=writer,
        filename_base=None if args.log_dir is None else args.log_dir + "/" + args.name,
        no_mlflow=no_mlflow,
    )

    if not no_mlflow:
        mlflow.pytorch.log_model(
            model,
            f"final-{run_name}",
        )

        mlflow.end_run()

    print("FINISH:", run_name)


if __name__ == "__main__":
    command_args = argparser.parse()
    if command_args.config is not None:
        with open(command_args.config) as f:
            args = yaml.safe_load(f)
        for key, value in vars(command_args).items():
            if value is not None and key not in args:
                print(f"Extra arg: {key} = {value}")
                args[key] = value
        args = Namespace(**args)
    else:
        args = command_args

    main(args)
