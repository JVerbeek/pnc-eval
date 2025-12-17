from models.regressors.base_regression_models import BatchFittableRegressionModel

class MultiOutputRandomForest(BatchFittableRegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True
        self.multi_output = True

    def fit(self, X, y):
        # X: array-like of shape (n_windows, window_size)
        # y: array-like of shape (n_windows, prediction_window_size)
        pass

    def predict(self, input_window, prediction_window_size=1):
        # input_window: array-like of shape (1, window_size) 
        pass


class AutoRegressiveRandomForest(BatchFittableRegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True
        self.auto_regressive = True

    def fit(self, X, y):
        # X: array-like of shape (n_windows, window_size)
        # y: array-like of shape (n_windows, prediction_window_size)

        # In autoregressive setting, we -always- predict only one step ahead, even though y may contain multiple steps.
        # We therefore need to fit the model multiple times, each time shifting the input window by one step and using the next step in y as target.
        # Note: this is somewhat whacky in the multivariate setting, so it's not yet supported.

        
        pass

    def predict(self, input_window, prediction_window_size=1):
        # input_window: array-like of shape (1, window_size)
        pass