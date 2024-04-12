from pathlib import Path

import mlflow
from mlflow.entities import ViewType

models = ["mlp", "cnn", "resnet18", "vit"]
datasets = ["mnist", "cifar10", "cifar100"]
wonly = False

mlflow.set_tracking_uri("http://192.168.63.34")

for model in models:
    for dataset in datasets:
        if wonly:
            target = f"20240301-ptq-{model}-{dataset}-wonly"
        else:
            target = f"20240301-ptq-{model}-{dataset}"
        print(f"Plotting {target}...")

        dir = Path("raw_results")
        dir.mkdir(exist_ok=True)

        experiment_id = mlflow.get_experiment_by_name(target).experiment_id

        df = mlflow.search_runs([experiment_id], "attributes.status = 'FINISHED'", run_view_type=ViewType.ACTIVE_ONLY)

        toi = ["params.in_bit"]
        tof = ["params.std", "metrics.analog_test_acc"]

        for key in toi:
            df[key] = df[key].astype("int32")
        for key in tof:
            df[key] = df[key].astype("float32")

        df = df.rename(
            columns={"params.in_bit": "Bitwidth", "params.std": "Noise", "metrics.analog_test_acc": "Accuracy"}
        )

        df = df[["Bitwidth", "Noise", "Accuracy"]]

        df = df.sort_values(["Bitwidth", "Noise"], ascending=[True, True])

        df.to_csv(dir / (target + ".csv"))
