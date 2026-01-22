import numpy as np

class BidirectionalCUSUMScorer:
    def __init__(self, decay=0):
        """
        Initialize the BidirectionalCUSUMScorer with a decay parameter.
        Args:
            decay (float): The constant decay to apply at each timestep.
        """
        self.decay = decay

    def score(self, X_s, predictions_list):
        """
        Aggregate a list of prediction arrays using bidirectional CUSUM.
        Args:
            X_s (list of np.ndarray): List of input data arrays corresponding to predictions.
            predictions_list (list of np.ndarray): List of prediction arrays from different models.
        Returns:
            np.ndarray: Aggregated predictions.
        """
        if not predictions_list:
            raise ValueError("predictions_list cannot be empty.")

        scores_list = [self._score_array(X, preds) for X, preds in zip(X_s, predictions_list)]

        return scores_list

    def _score_array(self, X, predictions):
        """
        Apply CUSUM scoring to the predictions.
        Args:
            X (np.ndarray): Input data array.
            predictions (np.ndarray): Array of model predictions.
        Returns:
            np.ndarray: CUSUM scores.
        """
        scores = np.zeros_like(predictions)
        cusum_pos = 0
        cusum_neg = 0

        for i, (x, pred) in enumerate(zip(X,predictions)):
            cusum_pos = max(0, cusum_pos + (x - pred) - self.decay)
            cusum_neg = min(0, cusum_neg + (x - pred) + self.decay)
            scores[i] = max(cusum_pos, -cusum_neg)

        return scores

class PositiveCUSUMScorer:
    def __init__(self, decay=0):
        self.decay = decay

    def score(self, X_s, predictions_list):
        """
        Aggregate a list of prediction arrays using positive CUSUM.
        Args:
            X_s (list of np.ndarray): List of input data arrays corresponding to predictions.
            predictions_list (list of np.ndarray): List of prediction arrays from different models.
        Returns:
            np.ndarray: Aggregated predictions.
        """
        if not predictions_list:
            raise ValueError("predictions_list cannot be empty.")

        scores_list = [self._score_array(X, preds) for X, preds in zip(X_s, predictions_list)]

        return scores_list

    def _score_array(self, X, predictions):
        """
        Apply positive CUSUM scoring to the predictions.
        Args:
            X (np.ndarray): Input data array.
            predictions (np.ndarray): Array of model predictions.
        Returns:
            np.ndarray: Positive CUSUM scores.
        """
        scores = np.zeros_like(predictions)
        cusum_pos = 0

        for i, (x, pred) in enumerate(zip(X, predictions)):
            cusum_pos = max(0, cusum_pos + (x - pred) - self.decay)
            scores[i] = cusum_pos

        return scores

class NegativeCUSUMScorer:
    def __init__(self, decay=0):
        self.decay = decay

    def score(self, X_s, predictions_list):
        """
        Aggregate a list of prediction arrays using negative CUSUM.
        Args:
            X_s (list of np.ndarray): List of input data arrays corresponding to predictions.
            predictions_list (list of np.ndarray): List of prediction arrays from different models.
        Returns:
            np.ndarray: Aggregated predictions.
        """
        if not predictions_list:
            raise ValueError("predictions_list cannot be empty.")

        scores_list = [self._score_array(X, preds) for X, preds in zip(X_s, predictions_list)]

        return scores_list

    def _score_array(self, X, predictions):
        """
        Apply negative CUSUM scoring to the predictions.
        Args:
            X (np.ndarray): Input data array.
            predictions (np.ndarray): Array of model predictions.
        Returns:
            np.ndarray: Negative CUSUM scores.
        """
        scores = np.zeros_like(predictions)
        cusum_neg = 0

        for i, (x, pred) in enumerate(zip(X, predictions)):
            cusum_neg = min(0, cusum_neg - (x - pred) + self.decay)
            scores[i] = -cusum_neg

        return scores
