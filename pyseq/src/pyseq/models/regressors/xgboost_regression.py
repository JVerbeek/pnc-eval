from ..regressors.base_regression_models import MultiOutputRegressor, AutoRegressiveRegressor

from xgboost import XGBRegressor

class MultiOutputXGBoost(MultiOutputRegressor):
    def __init__(self, **kwargs):
        super().__init__(XGBRegressor(**kwargs))


class AutoRegressiveXGBoost(AutoRegressiveRegressor):
    def __init__(self, **kwargs):
        super().__init__(XGBRegressor(**kwargs))
