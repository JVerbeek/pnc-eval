import abc

class BaseThresholder(abc.ABC):
    def __init__(self):
        self.fittable = False  # By default, thresholder is not fittable

    @abc.abstractmethod
    def threshold(self, scores):
        pass

class FittableThresholder(BaseThresholder):
    def __init__(self):
        super().__init__()
        self.fittable = True
        self.fitted = False

    @abc.abstractmethod
    def fit(self, scores, y=None):
        pass

    @abc.abstractmethod
    def threshold(self, scores):
        pass
