from sklearn.linear_model import LinearRegression

from .base_regression_models import BaseRegressionModel
    

class LinearRegressionModel(BaseRegressionModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        self.fittable = False # Linear regression in sklearn is not fittable in the sense we need here, i.e. it does not learn form normal data at large


    def predict(self, input_window, prediction_window_size=1):
        
        # input window is y, x we can just use range
        x = range(len(input_window))
        
        self.model.fit(x, input_window)
        
        x_pred = range(len(input_window), len(input_window) + prediction_window_size)
        y_pred = self.model.predict(x_pred)

        return y_pred


