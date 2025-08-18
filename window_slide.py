import numpy as np 
from model import RegressionModel, GPRModel

import matplotlib.pyplot as plt 
import gpflow as gpf
import abc

class Slider():   # Look ma, a strategy pattern! 
    def __init__(self, regressor: RegressionModel, X, y):
        self.regressor = regressor
        self.data = (X, y)
        self.data_length = len(X)

    @abc.abstractmethod
    def do_slide(self, timeseries, *args):
        pass

class MirrorSlider(Slider):
    pass

class NonOverlappingWindowSlider(Slider):
    def __init__(self, regressor, X, y):
        super().__init__(regressor, X, y)

    def get_window_train_test(self, window_start, window_stop, window):
        X, y = self.data
        Xtrain = X[window_start:window_stop].reshape(-1, 1)
        ytrain = y[window_start:window_stop].reshape(-1, 1)
        Xtest = X[window_stop:window_stop+window].reshape(-1, 1)
        ytest = y[window_stop:window_stop+window].reshape(-1, 1)
        return Xtrain, Xtest, ytrain, ytest
    
    def do_slide(self, window_size):
        window_start = 0
        window_stop = window_size
        scores = np.empty((10, 1))  # The first training window has no score.
        
        while window_stop < self.data_length + window_size:
            Xtrain, Xtest, ytrain, ytest = self.get_window_train_test(window_start, window_stop, window_size)
            self.regressor.fit(Xtrain, ytrain)
            score = self.regressor.predict_and_score(
                Xtest, ytest).reshape(-1, 1)
            scores = np.concatenate((scores, score))
            
            window_start = window_stop   # Update window
            window_stop = window_start + window_size
            
        return scores

if __name__ == "__main__":
    data = np.load("test_change_type.npz")
    X, y = data["Xs"][42], data["ys"][42]
    mod = GPRModel(gpf.models.GPR(
        (np.ones((2, 1)), np.ones((2, 1))), kernel=gpf.kernels.RBF()))
    
    slider = NonOverlappingWindowSlider(mod, X, y)
    scores = slider.do_slide(10)
   
    plt.plot(X, y)
    plt.plot(X, scores)
    plt.show()
    