import pandas as pd
from seqbench.metrics import (
    false_positives,
    mean_time_until_false_alarm,
    time_until_detection,
)

# Load in changepoints
hash = "some_hash"
results = np.load(f"results/{hash}.npz")
changepoints = results["predictions"]
ground_truth = np.load("generated_datasets/{hash}/cps.npz")

# Load in results
results_df = pd.DataFrame(columns["fpr", "mtfa", "tud"])

# Compute metrics
for column, metric in zip(
    results_df.columns,
    [false_positives, mean_time_until_false_alarm, time_until_detection],
):
    results_df[column] = metric(changepoints, ground_truth)

# Write away metrics to some file format
pd.to_csv("results/analysis_{hash}.csv", results_df)

# Question: how to specify which experiment?
