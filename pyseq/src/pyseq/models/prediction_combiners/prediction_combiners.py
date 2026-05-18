import numpy as np

def select_first(window_predictions, window_indices):
    """
    When multiple predictions exist for a single time point due to overlapping windows,
    this function selects the first prediction in the window.
    This function assumes the end index is non-inclusive. 
    """
    predictions = np.zeros(window_indices[-1][1]) # Initialize an array to hold final predictions.

    first_prediction_index = np.min(window_indices) # set all predictions before the first prediction to NaN, since they are not covered by any window and thus we have no basis for a prediction there
    predictions[:first_prediction_index] = np.nan

    #Easiest way to do this is loop in reverse order, so the first prediction is the last one written
    for preds, (start_idx, end_idx) in zip(reversed(window_predictions), reversed(window_indices)):
        predictions[start_idx:end_idx] = preds

    return predictions

def select_last(window_predictions, window_indices):
    """
    When multiple predictions exist for a single time point due to overlapping windows,
    this function selects the last prediction in the window.
    This function assumes the end index is non-inclusive.
    """
    predictions = np.zeros(window_indices[-1][1]) # Initialize an array to hold final predictions.

    first_prediction_index = np.min(window_indices) # set all predictions before the first prediction to NaN, since they are not covered by any window and thus we have no basis for a prediction there
    predictions[:first_prediction_index] = np.nan

    #Easiest way to do this is loop so the last prediction is the last one written
    for preds, (start_idx, end_idx) in zip(window_predictions, window_indices):
        predictions[start_idx:end_idx] = preds

    return predictions

def select_mean(window_predictions, window_indices):
    """
    When multiple predictions exist for a single time point due to overlapping windows,
    this function selects the mean of the predictions in the window.
    This function assumes the end index is non-inclusive.
    """
    predictions = np.zeros(window_indices[-1][1]) # Initialize an array to hold final predictions.
    counts = np.zeros_like(predictions)  # To count how many predictions contribute to each point

    first_prediction_index = np.min(window_indices) # set all predictions before the first prediction to NaN, since they are not covered by any window and thus we have no basis for a prediction there
    predictions[:first_prediction_index] = np.nan

    for preds, (start_idx, end_idx) in zip(window_predictions, window_indices):
        predictions[start_idx:end_idx] += preds
        counts[start_idx:end_idx] += 1

    # Avoid division by zero
    counts[counts == 0] = 1
    predictions /= counts

    return predictions
