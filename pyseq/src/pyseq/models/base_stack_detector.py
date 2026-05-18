import numpy as np
import abc

from .prediction_combiners.prediction_combiners import select_first, select_last, select_mean
from .window_sliders.window_slide import Slider

class StackDetector:
    def __init__(self, window_slider, regressor, scorer, thresholder, prediction_selection_strategy='first'):
        self.window_slider = window_slider
        self.regressor = regressor
        self.scorer = scorer
        self.thresholder = thresholder
        self.prediction_selection_strategy = prediction_selection_strategy

        self.target_window_size = self.window_slider.target_window_size
        self.predictor_window_size = self.window_slider.predictor_window_size

        # Input checks:
        # target_window_size must be positive integer
        if not (isinstance(self.target_window_size, int) and self.target_window_size > 0):
            raise ValueError("target_window_size must be a positive integer.")
        
        if not (isinstance(self.predictor_window_size, int) and self.predictor_window_size > 0):
            raise ValueError("predictor_window_size must be a positive integer.")
        
        # prediction_selection_strategy must be one of 'first', 'last', 'mean'
        if prediction_selection_strategy not in ['first', 'last', 'mean']:
            raise ValueError("prediction_selection_strategy must be one of 'first', 'last', 'mean'.")
        else:
            self.prediction_selector = {
                'first': select_first,
                'last': select_last,
                'mean': select_mean
            }[prediction_selection_strategy]

        # window_slider must be a Slider instance
        if not isinstance(window_slider, Slider):
            raise ValueError("window_slider must be an instance of Slider.")
        
        # Prediction window must be at least as large as the window sliders skip length (if it has any), otherwise there is not a prediction for each point
        # For future: consider allowing for sparse predictions when subsampling
        if hasattr(window_slider, 'skip_length'):
            if self.target_window_size < window_slider.skip_length:
                raise ValueError("target_window_size must be at least as large as the window_slider's skip_length.")

        # Determine if the StackDetector is fittable
        if self.regressor.fittable or self.thresholder.fittable:
            self.fittable = True
            self.is_fitted = False
        else:
            self.fittable = False
            #self.is_fitted = True
        
    def online_fit(self, y, t=None, X=None):
        raise NotImplementedError("Online fitting is not yet implemented for StackDetector.")

    def fit(self, y_s, t_s=None, X_s=None, cps_s=None):

        #Input checks:
        if not self.fittable:
           raise ValueError("This StackDetector has no fittable components. Neither regressor nor thresholder is fittable.")


        # Only give regressor access to normal data (before changepoint)
        # This requires that if no changepoints are present, y in y_s is the length of the data
        if cps_s is not None:
            y_s_normal = [y[:cp] for y, cp in zip(y_s, cps_s)]
            if t_s is not None:
                t_s_normal = [t[:cp] for t, cp in zip(t_s, cps_s)]
            else:
                t_s_normal = [None] * len(y_s_normal)
            if X_s is not None:
                X_s_normal = [X[:cp] for X, cp in zip(X_s, cps_s)]
            else:
                X_s_normal = [None] * len(y_s_normal)
        else: 
            raise UserWarning("cps_s is None, assuming all data is normal for regressor fitting.")
        

        # Apply window slider to normal data to get numpy arrays for fitting somewhat fast
        predictor_windows_list = []
        target_windows_list = []
        
        for y, t, X in zip(y_s_normal, t_s_normal, X_s_normal):
            self.window_slider.new_slide(y=y, t=t, X=X)

            predictor_windows, target_windows = self.window_slider.get_all_windows()
            predictor_windows_list.append(predictor_windows)
            target_windows_list.append(target_windows)

        # Combine all windows from all sequences
        predictor_windows_combined = np.vstack(predictor_windows_list)
        target_windows_combined = np.vstack(target_windows_list)

        if self.regressor.fittable:

            if self.regressor.fit_method == "online":
                raise ValueError("Online fitting is only supported through .online_fit(). Please use that method for online fitting.")
            elif self.regressor.fit_method != "batch":
                raise ValueError(f"Unknown regressor fit_method: {self.regressor.fit_method}. Supported methods are 'batch' and 'online' (only through .online_fit()).")

            self.regressor.fit(X=predictor_windows_combined, y=target_windows_combined)

        if self.thresholder.fittable:

            # Get regressor scores on all data
            regressor_predictions = self._get_regressor_predictions(y_s_normal, t_s_normal, X_s_normal) 

            scores = self.scorer.score(y_s, regressor_predictions)
             
            self.thresholder.fit(scores, y_s)

        self.is_fitted = True

    #Possible future feature: use cps_s for early stopping, but this would also need model support
    def predict(self, y_s, t_s=None, X_s=None, cps_s=None, return_scores=False, return_regressor_predictions=False):

        #early stopping not yet implemented
        if cps_s is not None:
            raise NotImplementedError("Using cps_s for early stopping during prediction is not yet implemented.")

        if (self.thresholder.fittable or self.regressor.fittable) and not self.is_fitted:
            raise ValueError("This StackDetector is fittable but has not been fitted yet. Please call fit() before predict().")

        regressor_predictions = self._get_regressor_predictions(y_s, t_s, X_s, cps_s) 
        scores = self.scorer.score(y_s, regressor_predictions)
        predictions = self.thresholder.threshold(scores)

        if return_scores and return_regressor_predictions:
            return predictions, scores, regressor_predictions
        elif return_scores:
            return predictions, scores
        elif return_regressor_predictions:
            return predictions, regressor_predictions
        else:
            return predictions


    def fit_predict(self, X_s, y_s):
        self.fit(X_s, y_s)
        return self.predict(X_s, y_s)

    #If we want early stopping, we need a mirror of this function which has the functionality to use y_s and self.thresholder to determine whether the change point has been detected yet
    def _get_regressor_predictions(self,  y_s, t_s=None, X_s=None, cps_s=None):

        regressor_predictions = []

        if t_s is None:
            t_s = [None] * len(y_s)
        if X_s is None:
            X_s = [None] * len(y_s)

        for y, t, X in zip(y_s, t_s, X_s):
            self.window_slider.new_slide(y, t, X)

            target_window_predictions = []
            target_window_indices = []

            #TODO: check below to see if it is correct now that next_window yields predictor/target pairs
            for (predictor_window, _), (predictor_window_start_index, predictor_window_end_index) in self.window_slider.next_window(return_indices=True):

                target_window_pred = self.regressor.predict(predictor_window, self.target_window_size)
                target_window_predictions.append(target_window_pred)

                target_window_indices.append((predictor_window_end_index, predictor_window_end_index+self.target_window_size)) 
        
            combined_predictions = self.prediction_selector(target_window_predictions, target_window_indices)

            
            regressor_predictions.append(combined_predictions)

        return regressor_predictions
