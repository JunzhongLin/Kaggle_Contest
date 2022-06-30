from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import xgboost as xgb
from src.base_models import SuperviseModel, rmsle_cv, rmsle
from joblib import dump, load
import os
import pandas as pd
import numpy as np


class HousePriceReg(SuperviseModel):

    def __init__(self, data_path):
        super(HousePriceReg, self).__init__()
        self.data_path = data_path
        self.model = {}
        self._init_data()
        self._init_regressor()

    def _init_data(self):
        self.X_train = load(os.path.join(self.data_path, 'X_train.pkl'))
        self.X_test = load(os.path.join(self.data_path, 'X_test.pkl'))
        self.y_train = load(os.path.join(self.data_path, 'y_train.pkl'))

    def _init_regressor(self):
        lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.01, random_state=1))

        ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.01, l1_ratio=.9, random_state=3))

        KRR = KernelRidge()

        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10,
                                           loss='huber', random_state=5)

        model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                     learning_rate=0.05, max_depth=3,
                                     min_child_weight=1.7817, n_estimators=2200,
                                     reg_alpha=0.4640, reg_lambda=0.8571,
                                     subsample=0.5213, random_state=7, nthread=-1)
        voting = VotingRegressor(
            estimators=[
                ('lasso', clone(lasso)),
                ('enet', clone(ENet)),
                ('KRR', clone(KRR)),
                ('gboost', clone(GBoost)),
                ('model_xgb', clone(model_xgb))
            ],
            weights=[0.15, 0.15, 0.3, 0.2, 0.3]
        )
        self.model = {
            'lasso': lasso,
            'ENet': ENet,
            'KRR': KRR,
            'GBoost': GBoost,
            'model_xgb': model_xgb,
            'voting': voting
        }

    def search_hyper_params(self, model_name, hyper_params):
        searchCV = GridSearchCV(self.model[model_name], hyper_params, n_jobs=-1, refit=True)
        searchCV.fit(self.X_train, self.y_train)
        self.searchCV = searchCV

        return None

    def prepare_submission(self, model_name):
        y_pred = np.expm1(self.model[model_name].predict(self.X_test))

        test_df = pd.read_csv('./house_price/data/test.csv')
        submission_df = pd.DataFrame(
            {
                'Id': test_df['Id'].values,
                'SalePrice': y_pred
            }
        )
        submission_df.to_csv('./house_price/submission.csv', index=False)

        return None


if __name__ == '__main__':
    house_price_model = HousePriceReg('./house_price/transformed_data/2022-06-29')
    # house_price_model.model_train('voting', {})
    house_price_model.model_train('voting', {})
    house_price_model.prepare_submission('voting')




