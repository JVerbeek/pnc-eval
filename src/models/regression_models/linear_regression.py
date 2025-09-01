from sklearn.linear_model import LinearRegression
from .base_model import RegressionModel
    
class LinearRegressionModel(RegressionModel):
    def __init__(self, model=LinearRegression()):
        super().__init__(model)
        
    def fit(self, Xtrain, ytrain):
        return self.model.fit(Xtrain, ytrain)

    def predict(self, Xtest):
        return self.model.predict(Xtest)

