import abc
import gpflow as gpf
from score_functions import cusum_score
from sklearn.linear_model import LinearRegression
class RegressionModel(abc.ABC):
    def __init__(self, model):
        self.model = model
        self.cusum_threshold = 0.1

    @abc.abstractmethod
    def fit(self, Xtrain, ytrain):
        pass

    @abc.abstractmethod
    def predict(self, Xtest):
        pass

    @abc.abstractmethod
    def predict_and_score(self, Xtest, ytest):
        pass


class GPRModel(RegressionModel):
    def fit(self, Xtrain, ytrain):
        data = gpf.models.util.data_input_to_tensor(
            (Xtrain.reshape(-1, 1), ytrain.reshape(-1, 1)))
        self.model.data = data
        opt = gpf.optimizers.Scipy()
        opt.minimize(self.model.training_loss, self.model.trainable_variables)
        return

    def predict(self, Xtest, return_latent=False):
        fmean, fvar = self.model.predict_f(Xtest)
        ymean, yvar = self.model.predict_y(Xtest)
        if return_latent:
            return fmean, fvar, ymean, yvar
        return ymean, yvar

    def predict_and_score(self, Xtest, ytest):
        ymean, _ = self.predict(Xtest)
        score = cusum_score(ymean, ytest)
        return score
    
    
class SKLearnModel(RegressionModel):
    def __init__(self, model=LinearRegression()):
        super().__init__(model)
        
    def fit(self, Xtrain, ytrain):
        return self.model.fit(Xtrain, ytrain)

    def predict(self, Xtest):
        return self.model.predict(Xtest)
        
    def predict_and_score(self, Xtest, ytest):
        y_hat = self.predict(Xtest)
        score = cusum_score(y_hat, ytest)
        return score

