import numpy as np
import abc

from src.models.window_sliders.window_slide import Slider

from src.models.prediction_combiners.prediction_combiners import select_first, select_last, select_mean

class StackDetector:
    def __init__(self, window_slider, regressor, scorer, thresholder, prediction_window_size=1, prediction_selection_strategy='first'):
        self.window_slider = window_slider
        self.regressor = regressor
        self.scorer = scorer
        self.thresholder = thresholder
        self.prediction_window_size = prediction_window_size
        self.prediction_selection_strategy = prediction_selection_strategy

        # Input checks:
        # prediction_window_size must be positive integer
        if not (isinstance(prediction_window_size, int) and prediction_window_size > 0):
            raise ValueError("prediction_window_size must be a positive integer.")
        
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
            if prediction_window_size < window_slider.skip_length:
                raise ValueError("prediction_window_size must be at least as large as the window_slider's skip_length.")

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

        if self.regressor.fittable:

            if self.regressor.fit_method == "online":
                raise ValueError("Online fitting is only supported through .online_fit(). Please use that method for online fitting.")
            elif self.regressor.fit_method != "batch":
                raise ValueError(f"Unknown regressor fit_method: {self.regressor.fit_method}. Supported methods are 'batch' and 'online' (only through .online_fit()).")

            # Only give regressor access to normal data (before changepoint)
            # This requires that if no changepoints are present, y in y_s is the length of the data
            if cps_s is not None:
                y_s_normal = [y[:cp] for y, cp in zip(y_s, cps_s)]
                if t_s is not None:
                    t_s_normal = [t[:cp] for t, cp in zip(t_s, cps_s)]
                else:
                    t_s_normal = None
                if X_s is None:
                    X_s_normal = [X[:cp] for X, cp in zip(X_s, cps_s)]
                else:
                    X_s_normal = None
            else: 
                raise UserWarning("cps_s is None, assuming all data is normal for regressor fitting.")
            
                y_s_normal, t_s_normal, X_s_normal = y_s, t_s, X_s

            # Apply window slider to normal data to get numpy arrays for fitting somewhat fast


            #self.regressor.fit(y_combined=, t_combined=, y_s=)

        if self.thresholder.fittable:

            if y_s is None:
                raise ValueError("y_s cannot be None if the thresholder is fittable.")
            
            # Get regressor scores on all data
            regressor_predictions = self._get_regressor_predictions(X_s)

            scores = self.scorer.score(y_s, regressor_predictions)
             
            self.thresholder.fit(scores, y_s)

        self.is_fitted = True

    #Possible future feature: use y_s for early stopping, but this would also need model support
    def predict(self, y_s, t_s=None, X_s=None, cps_s=None):

        if (self.thresholder.fittable or self.regressor.fittable) and not self.is_fitted:
            raise ValueError("This StackDetector is fittable but has not been fitted yet. Please call fit() before predict().")

        regressor_predictions = self._get_regressor_predictions(X_s)
        scores = self.scorer.score(X_s, regressor_predictions)
        predictions = self.thresholder.threshold(scores)

        return predictions

    def fit_predict(self, X_s, y_s):

        self.fit(X_s, y_s)

        return self.predict(X_s, y_s)

    #If we want early stopping, we need a mirror of this function which has the functionality to use y_s and self.thresholder to determine whether the change point has been detected yet
    def _get_regressor_predictions(self, X_s, y_s=None):

        regressor_predictions = []

        for X in X_s:
            self.window_slider.new_slide(X)
            window_predictions = []
            prediction_window_indices = []

            for window in self.window_slider.next_window():
                window_pred = self.regressor.predict(window, self.prediction_window_size)
                window_predictions.append(window_pred)
                _, window_end_index = self.window_slider.get_window_indices()
                prediction_window_indices.append((window_end_index, window_end_index+self.prediction_window_size))
        
            # Combine predictions using specified strategy:
            combined_predictions = self.prediction_selector(window_predictions, prediction_window_indices)
            skip = self.window_slider.get_window_indices()[1] - self.window_slider.get_window_indices()[0]
            combined_predictions = combined_predictions[skip+self.prediction_window_size:]
            regressor_predictions.append(combined_predictions)

        return regressor_predictions
