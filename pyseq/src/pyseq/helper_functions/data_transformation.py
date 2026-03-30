import numpy as np

def transform_for_autoregressive_fit(X, y):
    # X: numpy array of shape (n_windows, window_size)
    # y: numpy array of shape (n_windows, prediction_window_size)

    # In autoregressive setting, we need to create a new dataset where each input window is shifted by one step
    # and the corresponding target is the next step in the original y.

    n_windows, window_size = X.shape
    _, prediction_window_size = y.shape

    X_ar = np.zeros((n_windows * prediction_window_size, window_size))
    y_ar = np.flatten(y, order='C')

    for i in range(n_windows):
        for j in range(prediction_window_size):
            X_ar[i * prediction_window_size + j, :(window_size-j)] = X[i, j:]
            X_ar[i * prediction_window_size + j, (window_size-j):] = y[i, :j]

    return X_ar, y_ar
