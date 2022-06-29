# This a file contains some base class
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def rmsle_cv(model, X, y, n_folds=5):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class SuperviseModel:
    '''
    base object for the regressor which can provide some basic functionality, such as hyper-pararmeter
    search, model training and predicting
    '''
    def __init__(self):
        self.model = {}
        self.X_train, self.X_test, self.y_train = None, None, None
        self.best_params = None
        self.best_model = None

    def hyper_param_search(self, model_name, hyper_params, verbose=1, n_jobs=-1,
                           scoring='neg_mean_absolute_error', cv=5,
                           return_train_score=False):
        estimator = self.model[model_name]
        reg_search = GridSearchCV(
            estimator, hyper_params, verbose=verbose, n_jobs=n_jobs,
            scoring=scoring, cv=cv, return_train_score=return_train_score
        )
        reg_search.fit(self.X_train, self.y_train)
        self.best_params = reg_search.best_params_
        self.best_model = reg_search

        return reg_search

    def model_train(self, model_name, hyper_params):
        self.model[model_name].set_params(**hyper_params).fit(self.X_train, self.y_train)
        return self.model[model_name]

    def model_predict(self, model_name, X_test=None):
        if X_test is None:
            return self.model[model_name].predict(self.X_test)
        else:
            return self.model[model_name].predict(X_test)