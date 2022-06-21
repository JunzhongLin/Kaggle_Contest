from typing import List
import os.path

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import LocalOutlierFactor
import pickle, json
from joblib import load, dump
from src.base_preprocess import BaseProcessor


class DataframeSelector(BaseEstimator, TransformerMixin):
    '''
    # select sepecific feature columns
    '''
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        return X[self.feature_names].copy()


class ArraySelector(BaseEstimator, TransformerMixin):
    '''
    # select sepecific feature columns from  np.array
    '''

    def __init__(self, col_idx: List[int]):
        self.col_idx = col_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, self.col_idx]


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


class TitleExtractor(BaseEstimator, TransformerMixin):
    '''
    input only accept pandas dataframe
    '''

    def __init__(self, titles_dict: dict, output: List[str]):
        self.titles_dict = titles_dict
        self.output = output

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        # Parse the title from name
        X['title_cat'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        # convert rare title into 'others' and perform ordinal encoding
        X['title'] = X['title_cat'].apply(
            lambda x: self.titles_dict[x] if x in self.titles_dict else self.titles_dict['Others']
        )
        X.drop(columns=['Name'])

        return X[self.output]


class AloneChecker(BaseEstimator, TransformerMixin):
    def __init__(self, output=List[str]):
        self.output = output

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X[self.output[0]] = X.apply(lambda x: 1 if x[0]+x[1] != 0 else 0, axis=1)
        return X[self.output].values


class DataProcessor(BaseProcessor):
    def __init__(self, raw_data_train: str, raw_data_test: str, target: str,
                 project_transformed_data_path: str):
        super(DataProcessor, self).__init__()
        self.train_data = pd.read_csv(raw_data_train)
        self.test_data = pd.read_csv(raw_data_test)
        self.target = target
        self.project_folder = project_transformed_data_path
        self.global_dict = globals()

'''
def main(output_folder, remove_outlier=False, outlier_detect_params=None, outlier_threshold=-10.0,
         **kwargs):
    # initiate the pipelines
    pipelines = []

    # load data from csv file

    train_data = pd.read_csv('./Titanic/data/train.csv')
    test_data = pd.read_csv('./Titanic/data/test.csv')

    # prepare the definination of final features

    final_features = {}

    # build pipelines
    for idx, key in enumerate(kwargs.keys()):
        sel_cols = key.split(',')
        pipeline = Pipeline([
            ('selector', DataframeSelector(sel_cols))
        ] + [(transformer_name, globals()[transformer_name](**kwargs[key][transformer_name]))
             for transformer_name in list(kwargs[key].keys())[:-1]]
        )
        pipelines.append((key, pipeline))

        for feature, value in kwargs[key]['output_features'].items():
            final_features[feature] = value

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

    # Save the Config information for documentation purpose:
    saved_config_file = os.path.join(
        './Titanic/transformed_data/',
        output_folder,
        'configs.json'
    )

    # add outlier setting for documentation
    config_kwargs['outlier_setting'] = {
        'remove_outlier': remove_outlier,
        'outlier_detect_params': outlier_detect_params,
        'outlier_threshold': outlier_threshold
    }
    # add final output features
    config_kwargs['final_features'] = final_features

    with open(saved_config_file, 'w') as f:
        json.dump(config_kwargs, f, indent=4)

    return None
'''



if __name__ == '__main__':

    outlier_setting = {
        'remove_outlier': False,
        'outlier_detect_params': None,
        'outlier_threshold': -10.0
    }

    config_kwargs = {
        'Pclass,Sex,Embarked': {
            'DataframeSelector': {
                'feature_names': ['Pclass', 'Sex', 'Embarked']
            },
            'SimpleImputer': {
                'strategy': 'most_frequent'
            },
            'OrdinalEncoder': {
                'handle_unknown': 'error'
            },
            'output_features': {
                'Pclass': 'A proxy for socio-economic status (SES)',
                'Sex': 'Male or Female',
                'Embarked': 'C = Cherbourg, Q = Queenstown, S = Southampton'
            }
        },
        'Age,Name,SibSp,Parch': {
            'DataframeSelector': {
                'feature_names': ['Age', 'Name', 'SibSp', 'Parch']
            },
            'TitleExtractor': {
                'titles_dict': {'Master': 0, 'Miss': 1, 'Mrs': 2, 'Mr': 3, 'Others': 4},
                'output': ['Age', 'title', 'SibSp', 'Parch']
            },
            'KNNImputer': {
                'n_neighbors': 5,
            },
            'ArraySelector': {
                'col_idx': [0, 1]
            },
            'output_features': {
                'Age': 'Part is Imputed using KNN method based on title, SibSp, Parch',
                'Title': 'Master=0, Miss=1, Mrs=2, Mr=3, Others=4',
                # 'SibSp': 'number of sister/brother/spouse',
                # 'Parch': 'number of children/parent'
            }
        },
        'Fare': {
            'DataframeSelector': {
                'feature_names': ['Fare']
            },
            'SimpleImputer': {
                'strategy': 'mean'
            },
            'output_features': {
                'Fare': 'Passenger fare'
            }
        },
        'IsAlone': {
            'DataframeSelector': {
                'feature_names': ['SibSp', 'Parch']
            },
            'AloneChecker': {
                'output': ['IsAlone']
            },
            'output_features': {
                'IsAlone': '0 for alone, 1 for not alone'
            }
        }
    }

    # save the preprocessed data
    processor = DataProcessor(
        raw_data_train='./Titanic/data/train.csv',
        raw_data_test='./Titanic/data/test.csv',
        target='Survived',
        project_transformed_data_path='./Titanic/transformed_data/'
    )

    processor.process(
        output_folder='fourth_try',
        **outlier_setting,
        **config_kwargs
    )








