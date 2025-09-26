import abc

class BaseThresholder(abc.ABC):
    def __init__(self):
        self.is_fittable = False  # By default, thresholder is not fittable
        pass

    @abc.abstractmethod
    def threshold(self, scores):
        pass


class FittableThresholder(BaseThresholder):
    def __init__(self):
        super().__init__()
        self.is_fittable = True
        self.is_fitted = False

    @abc.abstractmethod
    def fit(self, scores, y=None):
        pass

    @abc.abstractmethod
    def threshold(self, scores):
        pass