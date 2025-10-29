import numpy as np

from .base_thresholders import BaseThresholder


class WaldConstantThresholder(BaseThresholder):
    def __init__(self, alpha=0.05):
        """
        Initialize the Wald's constant thresholder.
        This thresholder assumes that the scores result from a CUSUM scorer.
        Typically, the input data to the CUSUM should be normally distributed with constant mean and variance.
        When the CUSUM is calculated over residuals, as is the case with most CUSUM scorers in this package, instead the residuals should be normally distributed with constant mean and variance.

        Note: any scores are assumed to be non-negative, and are made positive otherwise.
        Args:
            alpha (float): Acceptable false alarm rate (default is 0.05).
        """
        self.alpha = alpha
        super().__init__()
        # Input checks:
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1.")

    def threshold(self, scores):
        """
        Apply Wald's constant to threshold the scores.
        Args:
            scores (np.ndarray): Array of scores to be thresholded.
        Returns:
            np.ndarray: Binary array indicating anomalies (1) and normal points (0).
        """
        
        
        #scores should not be empty
        if len(scores) == 0:
            raise ValueError("scores cannot be empty.")

        # Calculate the threshold based on Wald's SPRT
        threshold = -np.log(self.alpha)

        # Apply the threshold to the scores

        anomalies = [ (np.abs(score_array) > threshold).astype(int) for score_array in scores]


        return anomalies
