from argparse import Namespace
from copy import copy
from itertools import product

import mlflow
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm

import argparser
from main import main

"""
辞書でパラメタを設定しておくと自動で実験が回る。
辞書の key は main.py に渡すオプション名で value は値の列
"""

max_workers = 32
params = {
    # "dataset": ["mnist", "cifar10", "cifar100"],
    # "model_name": ["vit_base_patch16_384", "resnet18"],
    "in_bit": [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
    "std": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "count": list(range(1)),
}


def enumerate_args(base_args, params, df):
    keys = list(params.keys())
    vals = [params[key] for key in keys]

    enumerated_args = list()
    for pattern in product(*vals):
        args = copy(base_args)
        for i in range(len(keys)):
            args.__dict__[keys[i]] = pattern[i]

        # 重複チェック
        if len(df) > 0:
            p = df["run_id"] != ""
            for i in range(len(keys)):
                p = p & (df[f"params.{keys[i]}"] == str(pattern[i]))

            if len(df[p]) > 0:
                continue

        enumerated_args.append(args)
    return enumerated_args


def delete_unfinished_runs(df):
    for run_id in df[df["status"] != "FINISHED"]["run_id"]:
        print(f"delete_run: {run_id}")
        mlflow.delete_run(run_id)


def experiment(args):
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
        raise RuntimeError("MLFlow Remote ULR is NOT given.")

    df = mlflow.search_runs(
        [experiment_id],
        "",
    )
    if args.model == "mlflow":
        try:
            x = df[df["params.mlflow_model_url"] == args.mlflow_model_url]
            y = df[df["params.model_name"] == args.mlflow_model_url]

            if len(x) == 0:
                df = y
            else:
                df = x
        except:
            df = df[df["params.model_name"] == args.mlflow_model_url]

    delete_unfinished_runs(df)
    df = df[df["status"] == "FINISHED"]
    enumerated_args = enumerate_args(args, params, df)

    print(f"max_workers = {max_workers}")
    print(f"params = {params}")
    print(f"Number of cases = {len(enumerated_args)}")
    with mp.Pool(max_workers) as executor:
        list(tqdm(executor.map(main, enumerated_args), total=len(enumerated_args)))

    print("COMPLETE")


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

    experiment(args)
