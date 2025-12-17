from models.regressors.base_regression_models import BatchFittableRegressionModel

from helper_functions.data_transformation import transform_for_autoregressive_fit

from sklearn.ensemble import RandomForestRegressor
import numpy as np

class MultiOutputRandomForest(BatchFittableRegressionModel):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        self.fittable = True
        self.multi_output = True
        self.auto_regressive = False

        self.trained_prediction_window_size = None

        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        # X: array-like of shape (n_windows, window_size)
        # y: array-like of shape (n_windows, prediction_window_size)
        self.model.fit(X, y)

    def predict(self, input_window, prediction_window_size=1):
        # input_window: array-like of shape (1, window_size)
        # prediction_window_size is passed for compatibility but not used in multi-output setting, as it is defined by the training-phase prediction window size.

        #check if input dimension is correct, fix otherwise:
        if input_window.ndim == 1:
            input_window = input_window.reshape(1, -1)
        
        return self.model.predict(input_window)


class AutoRegressiveRandomForest(BatchFittableRegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True
        self.auto_regressive = True
        self.multi_output = False

    def fit(self, X, y):
        # X: array-like of shape (n_windows, window_size)
        # y: array-like of shape (n_windows, prediction_window_size)

        # In autoregressive setting, we -always- predict only one step ahead, even though y may contain multiple steps.
        # We therefore need to fit the model multiple times, each time shifting the input window by one step and using the next step in y as target.
        # Note: this is somewhat whacky in the multivariate setting, so it's not yet supported.

        # Apply helper function to transform X and y into the required format:

        if y.ndim > 1 and y.shape[1] > 1:
            X_ar, y_ar = transform_for_autoregressive_fit(X, y)
        else:
            X_ar, y_ar = X, y

        self.model.fit(X_ar, y_ar)


    def predict(self, input_window, prediction_window_size=1):
        # input_window: array-like of shape (1, window_size)
        # prediction_window_size: int, number of steps to predict ahead

        if input_window.ndim == 1:
            input_window = input_window.reshape(1, -1)

        y_pred = np.zeros((prediction_window_size,))
        for i in range(prediction_window_size):
            y_pred[i] = self.model.predict(input_window)
            # Append the predicted value to the input window for next prediction
            input_window = np.roll(input_window, -1)
            input_window[0, -1] = y_pred[i]

        return y_pred