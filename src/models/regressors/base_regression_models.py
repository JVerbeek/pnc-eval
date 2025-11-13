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
    def fit(self, input_window, prediction_window):
        pass

    @abc.abstractmethod
    def predict(self, input_window, prediction_window_size=1):
        pass