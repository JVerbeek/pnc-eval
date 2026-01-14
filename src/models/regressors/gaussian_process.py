import sys
sys.path.append("/home/janneke/repos/pnc-eval/")
from src.models.regressors.base_regression_models import BaseRegressionModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared
import numpy as np
import matplotlib.pyplot as plt
from gpflow.models import GPR
from gpflow.kernels import RBF
from gpflow.optimizers import Scipy

class GPRModel(BaseRegressionModel):
    def __init__(self):
        #super().__init__(model)
        self.is_fittable = False


    def predict(self, input_window, prediction_window_size=1):
        # input window is y, x we can just use range
        x = np.linspace(0, 1, len(input_window)).reshape(-1, 1)
        diff = prediction_window_size / len(input_window)
        x_new = np.linspace(1, 1+diff, prediction_window_size).reshape(-1, 1)
        input_window = np.reshape(input_window, (-1, 1))
        #kernel = 1.0 * RBF(length_scale=1) + WhiteKernel(0.1) 
        kernel = RBF()
        model = GPR((x, input_window), kernel=kernel)
        optimizer = Scipy()
        optimizer.minimize(model.training_loss, model.trainable_variables)
        y_pred = model.predict_y(x_new)[0].numpy().flatten()
        return y_pred
    