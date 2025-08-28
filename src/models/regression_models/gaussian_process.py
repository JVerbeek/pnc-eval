import gpflow as gpf
from base_model import RegressionModel

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

    
