from ..regressors.base_regression_models import MultiOutputRegressor, AutoRegressiveRegressor

from sklearn.ensemble import RandomForestRegressor

class MultiOutputRandomForest(MultiOutputRegressor):
    def __init__(self, **kwargs):
        super().__init__(RandomForestRegressor(**kwargs))


class AutoRegressiveRandomForest(AutoRegressiveRegressor):
    def __init__(self, **kwargs):
        super().__init__(RandomForestRegressor(**kwargs))
