# basic lib
import numpy as np
import pandas as pd
from typing import List

# sklearn packages
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

# sklearn ml model
from sklearn.ensemble import RandomForestClassifier

# data save/load
from joblib import load, dump
import os

from src.base_preprocess import BaseSkProcessor, ArraySelector


class SkPreprocessor(BaseSkProcessor):

    def __init__(self, **kwargs):
        super(SkPreprocessor, self).__init__(**kwargs)
        self.titles_dict = {'Master': 0, 'Miss': 1, 'Mrs': 2, 'Mr': 3, 'Others': 4}
        self.train_X, self.train_y, self.test_X = self._pd_preprocess()
        self.default_pipe = self._build_pipe()
        self.best_pipe = None

    def _pd_preprocess(self):
        train_y = self.train_data[self.target].copy()
        train_X = self.train_data.drop(self.target, axis=1).copy()
        test_X = self.test_data.copy()

        for dataset in (train_X, test_X):
            # extract title
            dataset['Title'] = dataset['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            dataset['Title'] = dataset['Title'].apply(
                lambda x: self.titles_dict[x] if x in self.titles_dict else self.titles_dict['Others']
            )
            # add isAlone feature
            dataset['IsAlone'] = dataset[['SibSp', 'Parch']].apply(
                lambda x: 1 if x[0]+x[1] != 0 else 0, axis=1
            )

        return train_X, train_y, test_X

    def _build_pipe(self):

        # pipe1 handles Pclass, Sex, Embarked
        pipe1 = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='error'))
                   ]
        )

        # pipe2_age will bucketize age column

        pipe2_age = ColumnTransformer(
            transformers=[
                ('bucket', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'), [0])
            ],
            remainder='passthrough'
        )

        # pipe2 handles age, title
        pipe2 = Pipeline(
            steps=[
                ('KNNimputer', KNNImputer(n_neighbors=5)),
                ('selector', ArraySelector(col_idx=[0, 1])),
                ('pipe2_age', pipe2_age)
            ]
        )

        # pipe 3 handles Fare
        pipe3 = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('bucket', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform'))
            ]
        )
        # pipe 4 handles IsAlone
        pipe4 = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ]
        )
        full_pipe = ColumnTransformer(
            transformers=[
                ('pipe1', pipe1, ['Pclass', 'Sex', 'Embarked']),
                ('pipe2', pipe2, ['Age', 'Title', 'SibSp', 'Parch']),
                ('pipe3', pipe3, ['Fare']),
                ('pipe4', pipe4, ['IsAlone'])
            ]
        )

        return full_pipe


if __name__ == '__main__':
    config_kwargs= {
        'output_folder': 'fourth_try',
        'raw_data_train': './Titanic/data/train.csv',
        'raw_data_test': './Titanic/data/test.csv',
        'target': 'Survived',
        'project_folder': './Titanic/transformed_data'
    }

    param_grid ={
        'preprocess__pipe2__KNNimputer__n_neighbors': [3, 5, 10],
        'preprocess__pipe2__pipe2_age__bucket__n_bins': [3, 5, 7],
        'preprocess__pipe3__bucket__n_bins': [3, 5, 7]
    }

    dummy_model = RandomForestClassifier()
    p = SkPreprocessor(**config_kwargs)
    p.sk_preprocess()

