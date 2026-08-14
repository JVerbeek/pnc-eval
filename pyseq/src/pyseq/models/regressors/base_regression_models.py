import abc

import numpy as np

from pyseq.helper_functions.data_transformation import transform_for_autoregressive_fit


class BaseRegressionModel(abc.ABC):
    # How StackDetector should hand data to this model:
    #   "windowed"  -> pre-cut (predictor_window, target_window) rows, consumed as i.i.d. samples.
    #                  StackDetector slides the window, the model sees one window per predict().
    #   "streaming" -> whole series in temporal order. StackDetector passes the raw series and
    #                  the model does its own windowing, carrying state across the series.
    processing = "windowed"

    def __init__(self, model):
        self.model = model
        self.fittable = False # By default, models are not fittable
        pass

    @abc.abstractmethod
    def predict(self, input_window, prediction_window_size=1):
        pass

    def set_window_geometry(self, predictor_window_size, target_window_size, skip_length):
        # StackDetector pushes the window slider's geometry in at construction time. Windowed
        # models get their windows pre-cut and ignore this; streaming models need it to know
        # where predictions are due and how far apart consecutive ones sit.
        pass

class FittableRegressionModel(BaseRegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True
    @abc.abstractmethod
    def predict(self, input_window, prediction_window_size=1):
        pass

class OnlineFittableRegressionModel(FittableRegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True
        self.fit_method = "online"

    @abc.abstractmethod
    def fit(self, input_window, prediction_window):
        pass

class BatchFittableRegressionModel(FittableRegressionModel):
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True
        self.fit_method = "batch"

    @abc.abstractmethod
    def fit(self, X, y):
        # X: array-like of shape (n_windows, window_size+exogenous_features_size)
        # y: array-like of shape (n_windows, prediction_window_size)
        pass

class StreamingRegressionModel(FittableRegressionModel):
    # Base for models that consume whole series in temporal order instead of pre-cut windows,
    # so that every timepoint is fed exactly once and state can be carried along the series.
    # StackDetector routes to fit(y_s) / predict_series(y) on seeing processing == "streaming".
    processing = "streaming"

    def __init__(self, model=None, predictor_window_size=1, target_window_size=1, skip_length=1):
        super().__init__(model)
        self.fittable = True
        self.fit_method = "batch"

        # Geometry given here is what the model uses when it is driven without a window slider
        # (either standalone, or by a StackDetector constructed with window_slider=None). When a
        # slider is present the StackDetector overwrites all three through set_window_geometry(),
        # so that the slider stays the single source of truth.
        self.predictor_window_size = predictor_window_size
        self.target_window_size = target_window_size
        self.skip_length = skip_length

    def set_window_geometry(self, predictor_window_size, target_window_size, skip_length):
        self.predictor_window_size = predictor_window_size
        self.target_window_size = target_window_size
        self.skip_length = skip_length

    def n_windows(self, series_length):
        # Must match Slider.get_all_windows(): window k covers y[k*S : k*S+P] with target
        # y[k*S+P : k*S+P+T], for as long as that target fits inside the series.
        n = (series_length - self.predictor_window_size - self.target_window_size) // self.skip_length + 1
        return max(n, 0)

    def target_positions(self, series_length):
        # Index of the last input timestep before window k's target: the prediction for window
        # k is made from the state after having consumed y[0 .. P + k*S - 1].
        return (self.predictor_window_size - 1) + self.skip_length * np.arange(self.n_windows(series_length))

    @abc.abstractmethod
    def fit(self, y_s):
        # y_s: list of 1D arrays, each one full series in temporal order
        pass

    @abc.abstractmethod
    def predict_series(self, y):
        # y: one full series. Returns (n_windows, target_window_size) predictions, one row per
        # window position, in the same order the window slider yields them.
        pass

    def predict(self, input_window, prediction_window_size=1):
        raise NotImplementedError(
            f"{type(self).__name__} has processing='streaming' and is driven one series at a "
            "time through predict_series(); it has no per-window predict()."
        )

class MultiOutputRegressor(BatchFittableRegressionModel):
    # Wraps any sklearn-like regressor (fit(X, y) / predict(X)) that natively supports
    # multi-output regression, e.g. RandomForestRegressor, XGBRegressor.
    def __init__(self, model):
        super().__init__(model)
        self.fittable = True
        self.multi_output = True
        self.auto_regressive = False

        self.trained_prediction_window_size = None

    def fit(self, X, y):
        # X: array-like of shape (n_windows, window_size+exogenous_features_size)
        # y: array-like of shape (n_windows, prediction_window_size)
        self.model.fit(X, y)

    def predict(self, input_window, prediction_window_size=1):
        # input_window: array-like of shape (1, window_size)
        # prediction_window_size is passed for compatibility but not used in multi-output setting, as it is defined by the training-phase prediction window size.

        #check if input dimension is correct, fix otherwise:
        if input_window.ndim == 1:
            input_window = input_window.reshape(1, -1)

        return self.model.predict(input_window)


class AutoRegressiveRegressor(BatchFittableRegressionModel):
    # Wraps any sklearn-like regressor (fit(X, y) / predict(X)) and drives it autoregressively,
    # always predicting a single step ahead and feeding that prediction back into the window.
    def __init__(self, model):
        super().__init__(model)

        self.fittable = True
        self.auto_regressive = True
        self.multi_output = False

    def fit(self, X, y):
        # X: array-like of shape (n_windows, window_size+exogenous_features_size)
        # y: array-like of shape (n_windows, prediction_window_size)

        # In autoregressive setting, we -always- predict only one step ahead, even though y may contain multiple steps.
        # We therefore need to fit the model multiple times, each time shifting the input window by one step and using the next step in y as target.
        # Note: this is somewhat whacky in the multivariate setting, so it's not yet supported.

        # Apply helper function to transform X and y into the required format:

        if y.ndim > 1 and y.shape[1] > 1:
            X_ar, y_ar = transform_for_autoregressive_fit(X, y)
        else:
            X_ar, y_ar = X, y

        self.model.fit(X_ar, y_ar)

    def predict(self, input_window, prediction_window_size=1):
        # input_window: array-like of shape (1, window_size)
        # prediction_window_size: int, number of steps to predict ahead

        if input_window.ndim == 1:
            input_window = input_window.reshape(1, -1)

        y_pred = np.zeros((prediction_window_size,))
        for i in range(prediction_window_size):
            y_pred[i] = self.model.predict(input_window)
            # Append the predicted value to the input window for next prediction
            input_window = np.roll(input_window, -1)
            input_window[0, -1] = y_pred[i]

        return y_pred
