import abc
import gpflow as gpf
import numpy as np
from score_functions import cusum_score

class RegressionModel(abc.ABC):
    def __init__(self, model):
        self.model = model
        self.cusum_threshold = 0.1

    @abc.abstractmethod
    def fit(Xtrain, ytrain):
        pass

    @abc.abstractmethod
    def predict(Xtest, ytest):
        pass

    @abc.abstractmethod
    def predict_and_score(Xtest, ytest):


class GPRModel(RegressionModel):
    def __init__(self, model: gpf.models.BayesianModel):
        super().__init__(model)

    def fit(self, X, y):
        data = gpf.models.util.data_input_to_tensor(
            (X.reshape(-1, 1), y.reshape(-1, 1)))
        self.model.data = data
        opt = gpf.optimizers.Scipy()
        opt.minimize(self.model.training_loss, self.model.trainable_variables)
        return

    def predict(self, X_test, y_test, return_latent=False):
        fmean, fvar = self.model.predict_f(X_test)
        ymean, yvar = self.model.predict_y(X_test)
        if return_latent:
            return fmean, fvar, ymean, yvar
        return ymean, yvar

    def predict_and_score(self, X_test, y_test):
        ymean, yvar = self.predict(X_test, y_test, return_latent=False)
        score = cusum_score(ymean, y_test)
        return score

