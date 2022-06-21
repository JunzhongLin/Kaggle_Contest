import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from typing import List
from sklearn.neighbors import LocalOutlierFactor
import pickle, json
from joblib import load, dump
import os


class BaseProcessor:
    def __init__(self, raw_data_train: str, raw_data_test: str, target: str,
                 project_transformed_data_path: str):
        self.train_data = pd.read_csv(raw_data_train)
        self.test_data = pd.read_csv(raw_data_test)
        self.final_features = {}
        self.target = target
        self.project_folder = project_transformed_data_path

    def process(self, output_folder, remove_outlier=False,
                outlier_detect_params=None, outlier_threshold=-10.0,
                **kwargs):

        # build pipelines
        pipelines = []

        for idx, key in enumerate(kwargs.keys()):

            pipeline = Pipeline(
                [(transformer_name, globals()[transformer_name](**kwargs[key][transformer_name]))
                 for transformer_name in list(kwargs[key].keys())[:-1]]
                                )
            pipelines.append((key, pipeline))

            for feature, value in kwargs[key]['output_features'].items():
                self.final_features[feature] = value

        full_pipeline = FeatureUnion(pipelines)

        # Generate train, test set
        X_train = full_pipeline.fit_transform(self.train_data)
        y_train = self.train_data[self.target].values
        X_test = full_pipeline.transform(self.test_data)

        # Remove outlier if needed
        if remove_outlier:
            lcf = LocalOutlierFactor(**outlier_detect_params)
            lcf.fit(X_train)
            X_train = X_train[lcf.negative_outlier_factor_ > outlier_threshold, :]
            y_train = y_train[lcf.negative_outlier_factor_ > outlier_threshold]

        # save the data
        saved_path = os.path.join(self.project_folder, output_folder)
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
        kwargs['outlier_setting'] = {
            'remove_outlier': remove_outlier,
            'outlier_detect_params': outlier_detect_params,
            'outlier_threshold': outlier_threshold
        }
        # add final output features
        kwargs['final_features'] = self.final_features

        with open(saved_config_file, 'w') as f:
            json.dump(kwargs, f, indent=4)

        return None

    def save_configs(self):

        pass