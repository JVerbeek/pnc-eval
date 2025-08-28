import abc

class RegressionModel(abc.ABC):
    def __init__(self, model):
        self.model = model
        pass
    
    @abc.abstractmethod
    def fit(self, Xtrain, ytrain):
        pass

    @abc.abstractmethod
    def predict(self, Xtest):
        pass

    @abc.abstractmethod
    def predict_and_score(self, Xtest, ytest):
        pass
