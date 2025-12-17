import abc


class BaseRegressionModel(abc.ABC):
    def __init__(self, model):
        self.model = model
        self.fittable = False # By default, models are not fittable
        pass

    @abc.abstractmethod
    def predict(self, input_window, prediction_window_size=1):
        pass

class FittableRegressionModel(BaseRegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True 

    @abc.abstractmethod
    def predict(self, input_window, prediction_window_size=1):
        pass

class OnlineFittableRegressionModel(FittableRegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True 

    @abc.abstractmethod
    def fit(self, input_window, prediction_window):
        pass

class BatchFittableRegressionModel(FittableRegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True

    @abc.abstractmethod
    def fit(self, X, y):
        # X: array-like of shape (n_windows, window_size)
        # y: array-like of shape (n_windows, prediction_window_size)
        pass
