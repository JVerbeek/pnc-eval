import argparse
import os
import numpy as np
import pandas as pd
from plotting import plot_cusum_results
from utils import handle_open_file
from metrics import (
    false_positives,
    mean_time_between_false_alarms,
    time_until_detection,
)
parser = argparse.ArgumentParser(description="Parse analysis arguments.")

# Analysis script processes entire results folder to figures and relevant metrics, indiscriminately.
parser.add_argument("-e", "--experiment-name", type=str, required=True)
parser.add_argument("-f", "--generate-figures", action="store_true")
parser.add_argument("-s", "--show-figures", action="store_true")

args = parser.parse_args()

# Define (raw) results directory
results_dir = f"experiments/{args.experiment_name}/results/"
results_raw_dir = results_dir+"raw/"

# Load in changepoints
for result in os.listdir(results_raw_dir):
    hashname = result.split("-")[-1].split(".")[0]   # should be the hash 
    result_npz = np.load(f"{results_raw_dir}/{result}")
    ground_truth = np.load(f"experiments/{args.experiment_name}/data/{hashname}/cps_train.npz")["cps"]
    detections = result_npz["predictions"]
    indices = np.arange(len(detections[0]), dtype=int)
    changepoints = [indices[detections[i].astype(bool)] for i in range(len(detections))]

    # Load in results
    results_df = pd.DataFrame(columns=["fpr", "mtfa", "tud"])

    # Compute metrics
    for column, metric in zip(
        results_df.columns,
        [false_positives, mean_time_between_false_alarms, time_until_detection],
    ):
        results_df[column] = metric(changepoints, ground_truth)

    # Write away metrics to some file format
    results_df.to_csv(f"experiments/{args.experiment_name}/results/analysis/analysis_{hashname}.csv")

    if args.generate_figures:
        os.makedirs(results_dir + "figures/" + hashname, exist_ok=True)
        alpha = handle_open_file(f"experiments/{args.experiment_name}/experiment-config/wald-constant-thresholder.yaml")["alpha"]
        ys = np.load(f"experiments/{args.experiment_name}/data/{hashname}/y_train.npz")["y"]
        ts = np.load(f"experiments/{args.experiment_name}/data/{hashname}/t_train.npz")["t"]
        cps = np.load(f"experiments/{args.experiment_name}/data/{hashname}/cps_train.npz")["cps"]
        plot_cusum_results(
            [ts, ys, cps, result_npz["predictions"], result_npz["scores"], result_npz["regression_predictions"]],
            alpha, # weak point, how do we know that every experiment has that threshold?
            filename=results_dir+ "/figures/" + hashname + "",
            show=args.show_figures
        )