from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import mlflow
from mlflow.entities import ViewType

models = ["mlp", "cnn"]
# models = ["resnet18", "vit_base_patch16_384"]
datasets = ["mnist", "cifar10", "cifar100"]

mlflow.set_tracking_uri("http://192.168.63.34")

target = "tamori-baseline-models-TL"
experiment_id = mlflow.get_experiment_by_name(target).experiment_id
df_original = mlflow.search_runs([experiment_id], "attributes.status = 'FINISHED'", run_view_type=ViewType.ACTIVE_ONLY)
tof = ["metrics.train_acc", "metrics.digital_test_acc"]

for key in tof:
    df_original[key] = df_original[key].astype("float32")

dir = Path("viz_results")
dir.mkdir(exist_ok=True)

for model in models:
    for dataset in datasets:
        print(f"Plotting {model} {dataset}...")

        df = df_original.rename(
            columns={"metrics.train_acc": "Train Accuracy", "metrics.analog_test_acc": "Test Accuracy"}
        )
        df = df[df["params.model"] == model]
        df = df[df["params.dataset"] == dataset]

        client = mlflow.client.MlflowClient("http://192.168.63.34")
        hist = client.get_metric_history(df["run_id"].iloc[0], "train_acc")

        train_accs = []
        for v in hist:
            train_accs.append(v.value)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        ax.plot(
            list(range(31)),
            train_accs,
            label="Train",
            marker="o",
        )

        hist = client.get_metric_history(df["run_id"].iloc[0], "digital_test_acc")

        test_accs = []
        for v in hist:
            test_accs.append(v.value)

        ax.plot(
            list(range(31)),
            test_accs,
            label="Test",
            marker="x",
        )

        ax.grid()
        ax.set_xlim(0, 30)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")

        fig.legend()
        fig.suptitle(f"Training Accuracy of  {model} model in {dataset}")
        fig.savefig(dir / f"{model}_{dataset}_train.png")

        plt.close()
