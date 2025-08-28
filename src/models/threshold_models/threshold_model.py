import numpy as np
from sklearn.metrics import _binary_clf_curve

class ThresholdModel:
    # default metric is F1 score
    def __init__(self, submodel, score_function, metric):
        self.submodel = submodel #submodel must be fitted before passing to ThresholdModel
        self.metric = metric
        self.score_function = score_function

        self.threshold = None

    #both X_train and y_train are n_D-by-t numpy arrays
    # .fit scores based on submodel, then optimizes threshold based on the metric
    # the threshold is stored in self.threshold
    def fit(self, X_s, Y_s):
        _, scores = self.regress_and_score(X_s, Y_s)

        self.fps, self.tps, self.thresholds = _binary_clf_curve(Y_s.flatten(), scores.flatten())
        self.fns = self.tps[-1] - self.tps
        self.tns = self.fps[-1] - self.fps

        self.metric_per_threshold = self.metric(self.fps, self.tps, self.fns, self.tns)
        self.best_threshold_index = np.argmax(self.metric_per_threshold)

    def regress_and_score(self, X_s, Y_s):
        scores = np.empty(Y_s.shape)
        regressions = np.empty(Y_s.shape)

        for i in range(X_s.shape[0]):
            regressions[i,:] = self.submodel.predict(X_s[i,:], Y_s[i,:])

            scores[i,:] = self.score_function(regressions[i,:])

        return regressions, scores
    
    def visualize_optimization(self):
        import matplotlib.pyplot as plt 

        plt.plot(self.thresholds, self.metric_per_threshold)
        plt.axvline(self.thresholds[self.best_threshold_index], color='red', linestyle='--')
        plt.xlabel("Threshold")
        plt.ylabel("Metric value")
        plt.title("Threshold optimization")
        plt.show()