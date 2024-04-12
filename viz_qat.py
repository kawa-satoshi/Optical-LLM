import itertools
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import mlflow
from mlflow.entities import ViewType
from matplotlib.lines import Line2D

models = ["mlp", "cnn"]
datasets = ["mnist", "cifar10", "cifar100"]
wonly = False

mlflow.set_tracking_uri("http://192.168.63.34")

for model in models:
    for dataset in datasets:
        target = f"20240301-qat-{model}"
        print(f"Plotting {target} {dataset}...")

        dir = Path("viz_results")
        dir.mkdir(exist_ok=True)

        experiment_id = mlflow.get_experiment_by_name(target).experiment_id
        markers = itertools.cycle(Line2D.markers.keys())
        linestyles = itertools.cycle(("solid", "dashed", "dashdot", "dotted"))

        df_original = mlflow.search_runs(
            [experiment_id], "attributes.status = 'FINISHED'", run_view_type=ViewType.ACTIVE_ONLY
        )

        df_original = df_original[df_original["params.dataset"] == dataset]

        toi = ["params.in_bit"]
        tof = ["params.std", "metrics.analog_test_acc"]

        for key in toi:
            df_original[key] = df_original[key].astype("int32")
        for key in tof:
            df_original[key] = df_original[key].astype("float32")

        df = df_original.rename(
            columns={"params.in_bit": "Bitwidth", "params.std": "Noise", "metrics.analog_test_acc": "Accuracy"}
        )

        # ===

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        for noise in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]:
            sub_df = df[df["Noise"] == noise].sort_values("Bitwidth")
            ax.plot(
                sub_df["Bitwidth"],
                sub_df["Accuracy"],
                label=f"{noise}",
                linestyle=next(linestyles),
                marker=next(markers),
            )

        ax.grid()
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.set_xlabel("Bitwidth")
        ax.set_ylabel("Accuracy")
        fig.legend(title="Noise")

        if wonly:
            fig.suptitle(f"Bitwidth vs Accuracy of {model} model in {dataset} (Noise apply only Weight)")
            fig.savefig(dir / f"{model}_{dataset}_bit_vs_acc_wonly.png")
        else:
            fig.suptitle(f"Bitwidth vs Accuracy of {model} model in {dataset}")
            fig.savefig(dir / f"{model}_{dataset}_bit_vs_acc.png")

        # ===

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        for bitwidth in [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]:
            sub_df = df[df["Bitwidth"] == bitwidth].sort_values("Noise")
            ax.plot(
                sub_df["Noise"],
                sub_df["Accuracy"],
                label=f"{bitwidth}",
                linestyle=next(linestyles),
                marker=next(markers),
            )

        ax.grid()
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.set_xlabel("Noise")
        ax.set_ylabel("Accuracy")
        fig.legend(title="Bitwidth")
        if wonly:
            fig.suptitle(f"Noise vs Accuracy of {model} model in {dataset} (Noise apply only Weight)")
            fig.savefig(dir / f"{model}_{dataset}_noise_vs_acc_wonly.png")
        else:
            fig.suptitle(f"Noise vs Accuracy of {model} model in {dataset}")
            fig.savefig(dir / f"{model}_{dataset}_noise_vs_acc.png")

        plt.close()
