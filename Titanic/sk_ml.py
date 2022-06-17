from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
from joblib import dump, load
import os
from src._base import Regressor


class TitanicXgbRegressor(Regressor):

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self._init_data()
        self._init_regressor()
        self.models = {}

    def _init_data(self):
        self.X_train = load(os.path.join(self.data_path, 'X_train.pkl'))
        self.X_test = load(os.path.join(self.data_path, 'X_test.pkl'))
        self.y_train = load(os.path.join(self.data_path, 'y_train.pkl'))

    def _init_regressor(self):
        xgb_reg = xgb.XGBRegressor(
            objective='reg:squarederror', colsample_bytree=0.5, learning_rate=0.05,
            max_depth=15, alpha=5, n_estimators=500, min_child_weight=7, gamma=0
        )
        self.models['xgb_reg'] = xgb_reg

    def res_visualize(self):
        pass

