import argparse
from typing import List

from modules.layers import linear


def parse(input: List[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config yaml file")

    parser.add_argument("--model", choices=["mlp", "cnn", "timm", "mlflow"])
    parser.add_argument("--timm_model_name", type=str, default=None)
    parser.add_argument("--mlflow_model_url", type=str, default=None)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--train_mode", choices=["analog", "digital"], default="analog")
    parser.add_argument("--quantization_type", choices=list(linear.choices.keys()))
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--data_parallel", type=bool, default=False)

    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--saved_model", default=None)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=128)

    parser.add_argument("--mlflow_url", type=str, default=None)
    parser.add_argument("--mlflow_experiment_name", type=str, default=None)
    parser.add_argument("--commit_hash", type=str, default=None)

    parser.add_argument("--in_bit", type=int, default=16)
    parser.add_argument("--out_bit", type=int, default=0)
    parser.add_argument("--w_bit", type=int, default=0)
    parser.add_argument("--std", type=float, default=0.0)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=512)  # for mlp

    if input is None:
        return parser.parse_args()
    else:
        return parser.parse_args(input)
