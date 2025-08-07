import abc
import gpflow as gpf 

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
    def __init__(model: gpf.models.BayesianModel):
        super().__init__(model)
    
    def fit(self, X, y):
        data = gpf.models.util.data_input_to_tensor((X, y))
        self.model.data = data
        opt = gpf.optimizers.Scipy()
        opt.minimize(self.model.training_loss, self.model.trainable_variables)
        return 
    
    def predict(self, X_test, y_test):
        fmean, fvar = self.model.predict_f(X_test)
        ymean, yvar = self.model.predict_y(X_test)
        

