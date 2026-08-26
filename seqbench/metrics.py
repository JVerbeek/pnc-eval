import numpy as np


# FPR
def false_positives(changepoints, ground_truth):
    """
    changepoints (list, n_datasets x n_detected, inhomogeneous): estimated changepoints
    ground_truth (np.ndarray, n_datasets x 1): true location of the changepoint (singular)
    """
    fprs = []
    for i, detected in enumerate(changepoints):
        fp = np.sum(detected < ground_truth[i])
        fprs.append(fp / detected.shape[0])
    return np.array(fprs)


def time_until_detection(changepoints, ground_truth):
    tuds = []
    for i, detected in enumerate(changepoints):
        if detected.size == 0:
            tuds.append(np.nan)
            continue
        relevant_points = detected[detected > ground_truth[i]]
        if relevant_points.size > 0:  # Calculate TUD
            tud = np.min(detected[detected > ground_truth[i]]) - ground_truth[i]
            tuds.append(tud)
        else:  # if not, the changepoint is detected with perfect accuracy
            tuds.append(0)
    return np.array(tuds)


def mean_time_between_false_alarms(changepoints, ground_truth):
    mbtfas = []
    for i, detected in enumerate(changepoints):
        if detected.size == 0:
            mbtfas.append(0)
            continue
        relevant_points = detected[detected < ground_truth[i]]
        if relevant_points.size > 0:
            distances = relevant_points[1:] - relevant_points[:-1]
            mbtfas.append(np.mean(distances))
        else:
            mbtfas.append(np.nan)
    return np.array(mbtfas)
