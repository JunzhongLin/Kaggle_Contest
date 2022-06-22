import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


class NumPlusOne(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X+1

    def get_feature_names_out(self, input_features=None):
        return input_features


df = pd.DataFrame({'brand': ['aaaa', 'asdfasdf', 'sadfds', 'NaN'],
                   'category': ['asdf', 'asfa', 'asdfas', 'as'],
                   'num1': [1, 1, 0, 0],
                   'target': [0.2, 0.11, 1.34, 1.123]})

preprocessor2 = ColumnTransformer(
    transformers=[
        ('test', NumPlusOne(), [0])
    ]
)

numeric_features = ['num1']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('test', NumPlusOne()),
    ('Pre', preprocessor2)
])

categorical_features = ['brand', 'category']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])



clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor',  LinearRegression())])
clf.fit(df.drop('target', 1), df['target'])

# clf.named_steps['preprocessor'].transformers_[0][1].named_steps['scaler'].get_feature_names_out(categorical_features)
clf[:-1].get_feature_names_out()