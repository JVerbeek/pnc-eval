import numpy as np
import abc

from .prediction_combiners.prediction_combiners import select_first, select_last, select_mean
from .window_sliders.window_slide import Slider

class StackDetector:
    # window_slider is optional: it is what cuts a series into (predictor, target) window pairs,
    # so a "windowed" regressor cannot work without one. A "streaming" regressor does its own
    # windowing from the geometry it carries, and can therefore run with no slider at all.
    def __init__(self, regressor, scorer, thresholder, window_slider=None, prediction_selection_strategy='first'):
        self.window_slider = window_slider
        self.regressor = regressor
        self.scorer = scorer
        self.thresholder = thresholder
        self.prediction_selection_strategy = prediction_selection_strategy

        # Input checks:
        # prediction_selection_strategy must be one of 'first', 'last', 'mean'
        if prediction_selection_strategy not in ['first', 'last', 'mean']:
            raise ValueError("prediction_selection_strategy must be one of 'first', 'last', 'mean'.")
        else:
            self.prediction_selector = {
                'first': select_first,
                'last': select_last,
                'mean': select_mean
            }[prediction_selection_strategy]

        if window_slider is None:
            # No slider: only streaming regressors can supply their own windowing.
            if self.regressor.processing != "streaming":
                raise ValueError(
                    f"window_slider is required for regressors with processing="
                    f"'{self.regressor.processing}': {type(self.regressor).__name__} is given "
                    "pre-cut windows and has nothing to cut them with. Pass a window_slider, or "
                    "use a regressor with processing='streaming'."
                )

            # The streaming regressor is the sole source of geometry in this case.
            self.predictor_window_size = self.regressor.predictor_window_size
            self.target_window_size = self.regressor.target_window_size
            self.skip_length = self.regressor.skip_length

        else:
            # window_slider must be a Slider instance
            if not isinstance(window_slider, Slider):
                raise ValueError("window_slider must be an instance of Slider.")

            self.target_window_size = self.window_slider.target_window_size
            self.predictor_window_size = self.window_slider.predictor_window_size
            self.skip_length = getattr(window_slider, 'skip_length', 1)

        # target_window_size must be positive integer
        if not (isinstance(self.target_window_size, int) and self.target_window_size > 0):
            raise ValueError("target_window_size must be a positive integer.")

        if not (isinstance(self.predictor_window_size, int) and self.predictor_window_size > 0):
            raise ValueError("predictor_window_size must be a positive integer.")

        # Prediction window must be at least as large as the skip length, otherwise there is not a prediction for each point
        # For future: consider allowing for sparse predictions when subsampling
        if self.target_window_size < self.skip_length:
            raise ValueError("target_window_size must be at least as large as skip_length.")

        if window_slider is not None:
            # Hand the slider's geometry to the regressor; the slider is the single source of
            # truth whenever there is one. Streaming regressors need skip_length to know how many
            # points of each window they have not already consumed; everything else ignores this.
            self.regressor.set_window_geometry(
                predictor_window_size=self.predictor_window_size,
                target_window_size=self.target_window_size,
                skip_length=self.skip_length,
            )

        # Determine if the StackDetector is fittable
        if self.regressor.fittable or self.thresholder.fittable:
            self.fittable = True
            self.is_fitted = False
        else:
            self.fittable = False
            #self.is_fitted = True

    def online_fit(self, y, t=None, X=None):
        raise NotImplementedError("Online fitting is not yet implemented for StackDetector.")

    def _check_streaming_inputs(self, t_s, X_s):
        # Streaming regressors are handed the raw series y only: they do their own windowing and
        # have no place to put timestamps or exogenous predictors as of the current implementation. (though especially exogenous variables should be implemented later)
        if self.regressor.processing != "streaming":
            return

        for name, value in (("t_s", t_s), ("X_s", X_s)):
            if value is not None and any(element is not None for element in value):
                raise ValueError(
                    f"{name} is not supported for streaming regressors: "
                    f"{type(self.regressor).__name__} has processing='streaming' and is given "
                    "only y_s. Pass y_s alone, or use a windowed regressor."
                )

    def fit(self, y_s, t_s=None, X_s=None, cps_s=None):

        #Input checks:
        if not self.fittable:
           raise ValueError("This StackDetector has no fittable components. Neither regressor nor thresholder is fittable.")

        self._check_streaming_inputs(t_s, X_s)


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
        

        if self.regressor.fittable:

            if self.regressor.fit_method == "online":
                raise ValueError("Online fitting is only supported through .online_fit(). Please use that method for online fitting.")
            elif self.regressor.fit_method != "batch":
                raise ValueError(f"Unknown regressor fit_method: {self.regressor.fit_method}. Supported methods are 'batch' and 'online' (only through .online_fit()).")

            if self.regressor.processing == "streaming":
                # Streaming regressors do their own windowing so that each timepoint is fed
                # exactly once, in order; hand them the normal series untouched.
                self.regressor.fit(y_s=y_s_normal)

            elif self.regressor.processing == "windowed":
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

                self.regressor.fit(X=predictor_windows_combined, y=target_windows_combined)

            else:
                raise ValueError(f"Unknown regressor processing: {self.regressor.processing}. Supported values are 'windowed' and 'streaming'.")

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

        self._check_streaming_inputs(t_s, X_s)

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

            if self.regressor.processing == "streaming":
                # One call per series: the regressor streams the whole series itself and returns
                # one prediction per window position, in window order. No slider is involved, so
                # the window positions are derived from the geometry directly. Both stay as
                # arrays: the combiners only iterate and index them, so splitting the (n, T)
                # block into a list of row views would buy nothing.
                target_window_predictions = self.regressor.predict_series(y)
                target_window_indices = self._target_window_indices(len(y))

                if len(target_window_predictions) != len(target_window_indices):
                    raise ValueError(
                        f"Streaming regressor returned {len(target_window_predictions)} predictions "
                        f"for a series with {len(target_window_indices)} windows."
                    )

            else:
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

    def _target_window_indices(self, series_length):
        # (start, stop) of each window's target, mirroring what the window slider yields:
        # window k predicts y[P + k*S : P + k*S + T], for as long as that target fits in the
        # series. Used on the streaming path, which has no slider to walk.
        #
        # Built as an (n_windows, 2) array rather than a list of tuples: the combiners run
        # np.min() over the whole thing, which would otherwise re-convert the list every call.
        n_windows = max((series_length - self.predictor_window_size - self.target_window_size) // self.skip_length + 1, 0)

        starts = self.predictor_window_size + self.skip_length * np.arange(n_windows)

        return np.column_stack((starts, starts + self.target_window_size))
