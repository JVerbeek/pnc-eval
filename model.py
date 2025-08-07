import abc
import gpflow as gpf 
import numpy as np
class Model(abc.ABC):
    def __init__(self, model):
        self.model = model 
    
    @abc.abstractmethod
    def fit():
        pass
    
    @abc.abstractmethod
    def predict():
        pass 
    
    
class GPRModel(Model):
    def __init__(self, model: gpf.models.BayesianModel):
        super().__init__(model)
    
    def fit(self, X, y):
        data = gpf.models.util.data_input_to_tensor((X.reshape(-1, 1), y.reshape(-1, 1)))
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
        
    def predict_and_compare(self, X_test, y_test):
        return 42   
    
if __name__=="__main__":
    mod = GPRModel(gpf.models.GPR((np.ones((2, 1)), np.ones((2, 1))), kernel=gpf.kernels.RBF()))
    mod.fit(np.linspace(0, 1, 100).reshape(-1, 1), np.sin(0.1*np.linspace(0, 1, 100)).reshape(-1, 1))