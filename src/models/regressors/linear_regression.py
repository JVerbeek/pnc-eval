from sklearn.linear_model import LinearRegression
from .base_regression_models import BaseRegressionModel
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionModel(BaseRegressionModel):
    def __init__(self):
        model = LinearRegression()
        super().__init__(model)
        self.is_fittable = False # Linear regression in sklearn is not fittable in the sense we need here, i.e. it does not learn form normal data at large


    def predict(self, input_window, prediction_window_size=1):
        """
        input window: (N,)
        y_pred: 
        """
        x = np.arange(len(input_window)).reshape(-1, 1)
        self.model.fit(x, input_window)

        x_pred = np.arange(len(input_window), len(input_window) + prediction_window_size).reshape(-1, 1)
        y_pred = self.model.predict(x_pred)
        return y_pred 


lr = LinearRegression()