import numpy as np

class BidirectionalCUSUMScorer:
    def __init__(self, decay=0):
        self.decay = decay

    def score(self, predictions_list):
        """
        Aggregate a list of prediction arrays using bidirectional CUSUM.
        Args:
            predictions_list (list of np.ndarray): List of prediction arrays from different models.
        Returns:
            np.ndarray: Aggregated predictions.
        """
        if not predictions_list:
            raise ValueError("predictions_list cannot be empty.")

        scores_list = [self._score_array(preds) for preds in predictions_list]

        return scores_list

    def _score_array(self, predictions):
        """
        Apply CUSUM scoring to the predictions.
        Args:
            predictions (np.ndarray): Array of model predictions.
        Returns:
            np.ndarray: CUSUM scores.
        """
        scores = np.zeros_like(predictions)
        cusum_pos = 0
        cusum_neg = 0

        for i, pred in enumerate(predictions):
            cusum_pos = max(0, cusum_pos + pred - self.decay)
            cusum_neg = min(0, cusum_neg + pred + self.decay)
            scores[i] = max(cusum_pos, -cusum_neg)

        return scores

class PositiveCUSUMScorer:
    def __init__(self, decay=0):
        self.decay = decay

    def score(self, predictions_list):
        """
        Aggregate a list of prediction arrays using positive CUSUM.
        Args:
            predictions_list (list of np.ndarray): List of prediction arrays from different models.
        Returns:
            np.ndarray: Aggregated predictions.
        """
        if not predictions_list:
            raise ValueError("predictions_list cannot be empty.")

        scores_list = [self._score_array(preds) for preds in predictions_list]

        return scores_list

    def _score_array(self, predictions):
        """
        Apply positive CUSUM scoring to the predictions.
        Args:
            predictions (np.ndarray): Array of model predictions.
        Returns:
            np.ndarray: Positive CUSUM scores.
        """
        scores = np.zeros_like(predictions)
        cusum_pos = 0

        for i, pred in enumerate(predictions):
            cusum_pos = max(0, cusum_pos + pred - self.decay)
            scores[i] = cusum_pos

        return scores

class NegativeCUSUMScorer:
    def __init__(self, decay=0):
        self.decay = decay

    def score(self, predictions_list):
        """
        Aggregate a list of prediction arrays using negative CUSUM.
        Args:
            predictions_list (list of np.ndarray): List of prediction arrays from different models.
        Returns:
            np.ndarray: Aggregated predictions.
        """
        if not predictions_list:
            raise ValueError("predictions_list cannot be empty.")

        scores_list = [self._score_array(preds) for preds in predictions_list]

        return scores_list

    def _score_array(self, predictions):
        """
        Apply negative CUSUM scoring to the predictions.
        Args:
            predictions (np.ndarray): Array of model predictions.
        Returns:
            np.ndarray: Negative CUSUM scores.
        """
        scores = np.zeros_like(predictions)
        cusum_neg = 0

        for i, pred in enumerate(predictions):
            cusum_neg = min(0, cusum_neg + pred + self.decay)
            scores[i] = -cusum_neg

        return scores