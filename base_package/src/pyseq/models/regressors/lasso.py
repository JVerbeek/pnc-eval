from sklearn.linear_model import Lasso
from .base_regression_models import BaseRegressionModel
import numpy as np
import matplotlib.pyplot as plt


class LassoModel(BaseRegressionModel):
    def __init__(self):
        model = Lasso(alpha=0.1)
        super().__init__(model)
        self.is_fittable = False


    def predict(self, input_window, prediction_window_size=1):
        # input window is y, x we can just use range
        x = np.arange(len(input_window)).reshape(-1, 1)
        self.model.fit(x, input_window)

        x_pred = np.arange(len(input_window), len(input_window) + prediction_window_size).reshape(-1, 1)
        y_pred = self.model.predict(x_pred)
        return y_pred
