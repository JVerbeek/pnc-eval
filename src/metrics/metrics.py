# F1 score
def f1_score(fps, tps, fns, tns):
    precision = tps / (tps + fps) if (tps + fps) > 0 else 0
    recall = tps / (tps + fns) if (tps + fns) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Accuracy
def accuracy(fps, tps, fns, tns):
    return (tps + tns) / (tps + tns + fps + fns) if (tps + tns + fps + fns) > 0 else 0

# Precision
def precision(fps, tps, fns, tns):
    return tps / (tps + fps) if (tps + fps) > 0 else 0

# Recall
def recall(fps, tps, fns, tns):
    return tps / (tps + fns) if (tps + fns) > 0 else 0

# F-beta score
def fbeta_score(fps, tps, fns, tns, beta=1):
    precision_val = precision(fps, tps, fns, tns)
    recall_val = recall(fps, tps, fns, tns)
    if (precision_val + recall_val) == 0:
        return 0
    return (1 + beta**2) * (precision_val * recall_val) / (beta**2 * precision_val + recall_val)