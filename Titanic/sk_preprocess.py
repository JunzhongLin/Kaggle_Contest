import os.path

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
import pickle, json
from joblib import load, dump


class DataframeSelector(BaseEstimator, TransformerMixin):
    '''
    # select sepecific feature columns
    '''
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return X[self.feature_names].values


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    ''' Encoding the categorical features

    '''
    def __init__(self, feature_names, encoder_params):
        self.feature_names = feature_names
        self.encoder_params = encoder_params
        self.le_dict = defaultdict(OrdinalEncoder)

    def fit(self, X, y=None):
        for col in self.feature_names:
            self.le_dict[col].set_params(**self.encoder_params).fit(X[col])
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X[self.feature_names] = X[self.feature_names].apply(
            lambda x: self.le_dict[x.name].transform(x)[0]
        )

        return X


class OutlierExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, kwargs, train=True):
        """
        Create a transformer to remove outliers. A threshold is set for selection
        criteria, and further arguments are passed to the LocalOutlierFactor class

        Keyword Args:
            neg_conf_val (float): The threshold for excluding samples with a lower
               negative outlier factor.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """

        self.threshold = kwargs.pop('neg_conf_val', -10.0)

        self.kwargs = kwargs
        self.train = train

    def transform(self, X, y=None):
        """
        Uses LocalOutlierFactor class to subselect data based on some threshold
        Returns:
            ndarray: subsampled data

        Notes:
            X should be of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if self.train:
            return (X[self.lcf.negative_outlier_factor_ > self.threshold, :],
                    y[self.lcf.negative_outlier_factor_ > self.threshold])
        else:
            return X

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lcf = LocalOutlierFactor(**self.kwargs)
        self.lcf.fit(X)
        return self


def main(output_folder, remove_outlier=False, outlier_detect_params=None, outlier_threshold=-10.0,
         **kwargs):
    # initiate the pipelines
    pipelines = []

    # load data from csv file

    train_data = pd.read_csv('./Titanic/data/train.csv')
    test_data = pd.read_csv('./Titanic/data/test.csv')

    # build pipelines
    for idx, key in enumerate(kwargs.keys()):
        sel_cols = key.split(',')
        pipeline = Pipeline([
            ('selector', DataframeSelector(sel_cols))
        ] + [(transformer_name, globals()[transformer_name](**kwargs[key][transformer_name]))
             for transformer_name in kwargs[key].keys()]
        )
        pipelines.append((key, pipeline))

    full_pipeline = FeatureUnion(pipelines)

    # transform data

    # Generate train, test set
    X_train = full_pipeline.fit_transform(train_data)
    y_train = train_data['Survived'].values
    X_test = full_pipeline.transform(test_data)

    # Remove outlier if needed
    if remove_outlier:
        lcf = LocalOutlierFactor(**outlier_detect_params)
        lcf.fit(X_train)
        X_train = X_train[lcf.negative_outlier_factor_ > outlier_threshold, :]
        y_train = y_train[lcf.negative_outlier_factor_ > outlier_threshold]

    # save the data
    saved_path = os.path.join('./Titanic/transformed_data', output_folder)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    names = ['X_train.pkl', 'X_test.pkl', 'y_train.pkl']
    data_sets = [X_train, X_test, y_train]
    for name, data in zip(names, data_sets):
        with open(os.path.join(saved_path, name), 'wb') as f:
            dump(data, f)

    return None


if __name__ == '__main__':

    outlier_setting = {
        'remove_outlier': False,
        'outlier_detect_params': None,
        'outlier_threshold': -10.0
    }

    config_kwargs = {
        'Age,SibSp,Parch,Fare': {
            'SimpleImputer': {
                'strategy': 'mean'
            }
        },
        'Pclass,Sex,Embarked': {
            'SimpleImputer': {
                'strategy': 'most_frequent'
            },
            'OrdinalEncoder': {
                'handle_unknown': 'error'
            }
        }



    }
    # save the preprocessed data
    output_folder = 'second_try'
    main(output_folder, **outlier_setting, **config_kwargs)

    # save the config json file for documentation
    saved_config_file = os.path.join(
        './Titanic/transformed_data/',
        output_folder,
        'configs.json'
    )

    # add outlier setting for documentation
    config_kwargs['outlier_setting'] = outlier_setting
    with open(saved_config_file, 'w') as f:
        json.dump(config_kwargs, f, indent=4)





