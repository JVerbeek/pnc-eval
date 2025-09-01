import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve #import might not work in older sklearn versions, chimport needs to be conditional

class ThresholdModel:
    # default metric is F1 score
    def __init__(self, submodel, score_function, metric):
        self.submodel = submodel #submodel must be fitted before passing to ThresholdModel
        self.metric = metric
        self.score_function = score_function

        self.threshold = None
        
    # .fit scores based on submodel, then optimizes threshold based on the metric
    # the threshold is stored in self.threshold
    def fit(self, X_s, Y_s):
        _, scores = self.regress_and_score(X_s, Y_s)

        # Below flatten Y_s and scores to 1D numpy arrays
        self.fps, self.tps, self.thresholds = _binary_clf_curve(np.vstack(Y_s).ravel(), np.vstack(scores).ravel(), pos_label=1)
        self.fns = self.tps[-1] - self.tps
        self.tns = self.fps[-1] - self.fps

        self.metric_per_threshold = self.metric(self.fps, self.tps, self.fns, self.tns)
        self.best_threshold_index = np.argmax(self.metric_per_threshold)

    def regress_and_score(self, X_s, Y_s):
        scores = []
        regressions = []

        for X, y in zip(X_s, Y_s):
            regressions.append(self.submodel.predict(X))

            scores.append(self.score_function(regressions[-1], y).reshape(-1, 1))

        return regressions, scores
    
    def visualize_optimization(self):
        import matplotlib.pyplot as plt 

        plt.plot(self.thresholds, self.metric_per_threshold)
        plt.axvline(self.thresholds[self.best_threshold_index], color='red', linestyle='--')
        plt.xlabel("Threshold")
        plt.ylabel("Metric value")
        plt.title("Threshold optimization")
        plt.show()