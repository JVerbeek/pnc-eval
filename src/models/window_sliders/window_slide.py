import numpy as np 
import sys
sys.path.append("src")
from models.regression_models.base_model import RegressionModel, GPRModel, SKLearnModel

import matplotlib.pyplot as plt 
import gpflow as gpf
import abc

class Slider():   # Look ma, a strategy pattern! 
    def __init__(self, regressor: RegressionModel, data: tuple):
        self.regressor = regressor
        self.data = data
        self.data_length = len(data[0])

    @abc.abstractmethod
    def do_slide(self, window_size):
        pass

# Not implemented yet
# class MirrorSlider(Slider):
#     def do_slide(self, window_size):
#         pass

class NonOverlappingWindowSlider(Slider):
    def get_window_train_test(self, window_start, window_stop, window):
        Xall, yall = self.data
        Xtrain = Xall[window_start:window_stop].reshape(-1, 1)
        ytrain = yall[window_start:window_stop].reshape(-1, 1)
        Xtest = Xall[window_stop:window_stop+window].reshape(-1, 1)
        ytest = yall[window_stop:window_stop+window].reshape(-1, 1)
        return Xtrain, Xtest, ytrain, ytest
    
    def do_slide(self, window_size):
        window_start = 0
        window_stop = window_size
        scores = np.empty((10, 1))  # The first training window has no score.
        
        while window_stop < self.data_length:
            Xtrain, Xtest, ytrain, ytest = self.get_window_train_test(window_start, window_stop, window_size)
            self.regressor.fit(Xtrain, ytrain)
            print(Xtest.shape)
            score = self.regressor.predict_and_score(
                Xtest, ytest).reshape(-1, 1)
            scores = np.concatenate((scores, score))
            
            window_start = window_stop   # Update window
            window_stop = window_start + window_size
            
        return scores

if __name__ == "__main__":
    test_data = np.load("test_change_type.npz")
    X, y = test_data["Xs"][42], test_data["ys"][42]
    mod = SKLearnModel()
    
    slider = NonOverlappingWindowSlider(mod, (X, y))
    cusum_scores = slider.do_slide(10) 
    plt.plot(X, y)
    plt.plot(X, cusum_scores)
    plt.show()
    