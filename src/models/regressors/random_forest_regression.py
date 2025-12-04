
from models.regressors.base_regression_models import BatchFittableRegressionModel

class RandomForestRegressionModel(BatchFittableRegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True 


    def fit(self, X, y):
        # X: array-like of shape (n_windows, window_size) or (n_windows, n_features, window_size) for multivariate
        # y: array-like of shape (n_windows, prediction_window_size)
        pass


    def batch_predict(self, X, y):
        # X: array-like of shape (n_windows, window_size) or (n_windows, n_features, window_size) for multivariate
        # y: array-like of shape (n_windows, prediction_window_size)
        pass

    def predict(self, input_window, prediction_window_size=1):
        # input_window: array-like of shape (1, window_size) or (1, n_features, window_size) for multivariate
        pass