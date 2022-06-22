import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from typing import List
from sklearn.neighbors import LocalOutlierFactor
import pickle, json
from joblib import load, dump
import os
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


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

class BaseProcessor:
    def __init__(self,):
        self.train_data = None
        self.test_data = None
        self.final_features = {}
        self.target = None
        self.project_folder = None
        self.global_dict = {}
        self.output_folder = None
        self.kwargs = None

    def process(self, remove_outlier=False,
                outlier_detect_params=None, outlier_threshold=-10.0,
                **kwargs):

        # build pipelines
        pipelines = []

        for idx, key in enumerate(kwargs.keys()):

            pipeline = Pipeline(
                [(transformer_name, self.global_dict[transformer_name](**kwargs[key][transformer_name]))
                 for transformer_name in list(kwargs[key].keys())[:-1]]
                                )
            pipelines.append((key, pipeline))

            for feature, value in kwargs[key]['output_features'].items():
                self.final_features[feature] = value

        full_pipeline = FeatureUnion(pipelines)

        # Generate train, test set
        # X_data = self.train_data.append(self.test_data) ## possible data leakage !!
        # full_pipeline.fit(X_data)
        self.X_train = full_pipeline.fit_transform(self.train_data)
        self.y_train = self.train_data[self.target].values
        self.X_test = full_pipeline.transform(self.test_data)

        # Remove outlier if needed
        if remove_outlier:
            lcf = LocalOutlierFactor(**outlier_detect_params)
            lcf.fit(self.X_train)
            self.X_train = self.X_train[lcf.negative_outlier_factor_ > outlier_threshold, :]
            self.y_train = self.y_train[lcf.negative_outlier_factor_ > outlier_threshold]

        # add outlier setting for documentation
        self.kwargs = kwargs
        self.kwargs['outlier_setting'] = {
            'remove_outlier': remove_outlier,
            'outlier_detect_params': outlier_detect_params,
            'outlier_threshold': outlier_threshold
        }
        # add final output features
        self.kwargs['final_features'] = self.final_features

        return None

    def save_data(self,):
        # save the data
        saved_path = os.path.join(self.project_folder, self.output_folder)
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        names = ['X_train.pkl', 'X_test.pkl', 'y_train.pkl']
        data_sets = [self.X_train, self.X_test, self.y_train]
        for name, data in zip(names, data_sets):
            with open(os.path.join(saved_path, name), 'wb') as f:
                dump(data, f)

        print('Preprocessed data saved in {}'.format(
            saved_path
        ))
        return None

    def save_log(self):
        # Save the Config information for documentation purpose:
        saved_config_file = os.path.join(
            './Titanic/transformed_data/',
            self.output_folder,
            'configs.json'
        )
        with open(saved_config_file, 'w') as f:
            json.dump(self.kwargs, f, indent=4)

        print('Preprocess configs saved in {}'.format(
            saved_config_file
        ))

        return None


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

    def get_feature_names_out(self, input_features=None):
        out_features = []
        for i in self.col_idx:
            out_features.append(input_features[i])
        return out_features


class BaseSkProcessor:

    def __init__(self, output_folder, raw_data_train, raw_data_test,
                 target, project_folder):
        self.train_data = pd.read_csv(raw_data_train)
        self.test_data = pd.read_csv(raw_data_test)
        self.target = target
        self.project_folder = project_folder
        self.output_folder = output_folder
        self.best_pipe, self.default_pipe = None, None
        self.train_X, self.train_y, self.test_X = None, None, None

    def sk_preprocess(self, pipe=None, use_best=False):

        if use_best:
            pipe = self.best_pipe
        else:
            if not pipe:
                pipe = self.default_pipe

        pipe.fit(self.train_X.append(self.test_X))

        self.transformed_train_X = pipe.transform(self.train_X)
        self.transformed_train_y = self.train_y
        self.transformed_test_X = pipe.transform(self.test_X)

        return None

    def search_param(self,  model, hyper_params, pipe=None):
        if not pipe:
            pipe = self.default_pipe

        full_pipe = Pipeline(
            steps=[
                ('preprocess', pipe),
                ('model', model)
            ]
        )

        searchCV = GridSearchCV(full_pipe, hyper_params, n_jobs=-1)
        searchCV.fit(self.train_X, self.train_y)

        print("Best parameter (CV score=%0.3f):" % searchCV.best_score_)
        print(searchCV.best_params_)

        self.best_pipe = searchCV.best_estimator_[:-1]

        return None

    def save_data(self):
        # save the data
        saved_path = os.path.join(self.project_folder, self.output_folder)
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        names = ['X_train.pkl', 'X_test.pkl', 'y_train.pkl']
        data_sets = [self.transformed_train_X, self.transformed_test_X,
                     self.transformed_train_y]
        for name, data in zip(names, data_sets):
            with open(os.path.join(saved_path, name), 'wb') as f:
                dump(data, f)

        print('Preprocessed data saved in {}'.format(
            saved_path
        ))
        return None

    def save_log(self, best_pipe=False):
        saved_path = os.path.join(self.project_folder, self.output_folder)
        if best_pipe:
            pipe = self.best_pipe
        else:
            pipe = self.default_pipe

        with open(os.path.join(saved_path, 'pipe.pkl'), 'wb') as f:
            dump(pipe, f)

        print('Pipe saved in {}'.format(
            saved_path
        ))

        return None


