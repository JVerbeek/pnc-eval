import numpy as np

def cusum_score(predictions, targets):
    n = len(targets)
    cusum_score = np.cumsum((predictions - targets) ** 2 / n)
    return cusum_score