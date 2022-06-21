import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from typing import List
from sklearn.neighbors import LocalOutlierFactor
import pickle, json
from joblib import load, dump
import os


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
        X_data = self.train_data.append(self.test_data)
        full_pipeline.fit(X_data)
        self.X_train = full_pipeline.transform(self.train_data)
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
