import pandas as pd
import numpy as np
import os
import sys
from src.correlation_eval import theils_u
from matplotlib import pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew #for some statistics
#sklearn package
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, PowerTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from joblib import load, dump
import os
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from src.base_preprocess import BaseSkProcessor, ArraySelector


color = sns.color_palette()
sns.set_style('darkgrid')


class PlusOne(BaseEstimator, TransformerMixin):

    def __init__(self, ):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X + 1

    def get_feature_names_out(self, input_features=None):
        return input_features


class HousePriceProcess(BaseSkProcessor):

    def __init__(self, **kwargs):
        super(HousePriceProcess, self).__init__(**kwargs)

        self._pd_process()
        self.default_pipe = self._build_pipeline()
        self.best_pipe = None

    def _check_outlier(self, df, len_train):

        full_data = df
        train_set_cond = full_data['Id'] <= len_train

        index_to_drop = full_data[
            (full_data['SalePrice'] < 300000) & (full_data['GrLivArea'] > 4000) & train_set_cond
            ].index

        LotFrontAge_idx = full_data[
            (full_data['SalePrice'] < 300000) & (full_data['LotFrontage'] > 300) & train_set_cond
            ].index

        index_to_drop = index_to_drop.union(LotFrontAge_idx, )

        BsmtSF1_idx = full_data[
            (full_data['SalePrice'] < 200000) & (full_data['BsmtFinSF1'] > 4000) & train_set_cond
            ].index
        index_to_drop = index_to_drop.union(BsmtSF1_idx)

        TotalBsmtSF_idx = full_data[
            (full_data['SalePrice'] < 200000) & (full_data['TotalBsmtSF'] > 6000) & train_set_cond
            ].index
        index_to_drop = index_to_drop.union(TotalBsmtSF_idx)

        LotArea_idx = full_data[
            full_data['LotArea'] > 100000
            ].index

        index_to_drop = index_to_drop.union(LotArea_idx)
        return index_to_drop

    def _impute_missing_value(self, df,):
        X_full = df

        na_columns = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageQual', 'GarageFinish', 'GarageCond', 'GarageType',
            'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2',
            'BsmtFinType1', 'MasVnrType'
        ]

        zero_columns = [
            'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath',
            'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF',
            'GarageArea'
        ]

        num_cols, cat_cols = [], []
        for col in X_full:
            if X_full[col].dtype == 'float64':
                num_cols.append(col)
            else:
                cat_cols.append(col)

        median_columns = list(set(num_cols) - set(na_columns) - set(zero_columns))
        mode_columns = list(set(cat_cols) - set(na_columns) - set(zero_columns))

        for col in na_columns:
            X_full[col].fillna('None', inplace=True)
        for col in zero_columns:
            X_full[col].fillna(0, inplace=True)
        for col in median_columns:
            X_full[col].fillna(X_full[col].median(), inplace=True)
        for col in mode_columns:
            X_full[col].fillna(X_full[col].mode()[0], inplace=True)

        return X_full

    def _construct_features(self, df):
        X_full = df
        # for avoiding divided by zero
        eps = 1

        # Area regarding to level
        # ---------------------------------
        # Total above grade square feet
        X_full['TotalGrFlrSF'] = X_full['1stFlrSF'] + X_full['2ndFlrSF']
        X_full['TotalGrFlrSF'] = X_full['TotalGrFlrSF'].astype('Float64')

        # Total square feet.
        X_full['TotalSF'] = X_full['TotalGrFlrSF'] + X_full['TotalBsmtSF']
        X_full['TotalSF'] = X_full['TotalSF'].astype('Float64')

        # Bedroom related
        # ----------------------------------
        # average area per bedroom
        X_full['BedroomAvgAbGr'] = X_full['GrLivArea'] / (X_full['BedroomAbvGr'] + eps)

        # Bathroom related
        # -----------------------------------
        # total number of bathrooms above grade

        X_full['TotalBathAbGr'] = 2 * X_full['FullBath'] + X_full['HalfBath']
        X_full['TotalBathAbGr'] = X_full['TotalBathAbGr'].astype('Int64')

        # total number of bathrooms in basement
        X_full['TotalBathBsmt'] = 2* X_full['BsmtFullBath'] + X_full['BsmtHalfBath']
        X_full['TotalBathBsmt'] = X_full['TotalBathBsmt'].astype('Int64')

        # total number of bathrooms
        X_full['TotalBath'] = X_full['TotalBathAbGr'] + X_full['TotalBathBsmt']
        X_full['TotalBath'] = X_full['TotalBath'].astype('Int64')

        # ratio between bedrooms and bathrooms above ground
        X_full['BedroomBathRatioAbGr'] = X_full['BedroomAbvGr'] / (X_full['TotalBathAbGr'] + eps)

        # ratio between bedrooms and bathrooms

        X_full['BedroomBathRatio'] = X_full['BedroomAbvGr'] / (X_full['TotalBath'] + eps)

        # garage related
        # ----------------------
        # average garage size per car

        X_full['GarageAvgCarSF'] = X_full['GarageArea'] / (X_full['GarageCars'] + eps)

        # Porch related
        # ------------------
        # total area of porch
        X_full['TotalPorchSF'] = X_full['OpenPorchSF'] + X_full['EnclosedPorch'] + X_full['3SsnPorch'] + X_full[
            'ScreenPorch']


        # Overall state
        X_full['OverallState'] = X_full['OverallQual'] + X_full['OverallCond']
        X_full['OverallState'] = X_full['OverallState'].astype('Int64')

        # Age
        X_full['Age'] = X_full['YrSold'] - X_full['YearBuilt']
        X_full['Age'] = X_full['Age'].apply(
            lambda x: x if x >= 0 else 0
        )
        X_full['Age'] = X_full['Age'].astype('Float64')

        # RemodAge:
        X_full['RemodAge'] = X_full['YrSold'] - X_full['YearRemodAdd']
        X_full['RemodAge'] = X_full['RemodAge'].apply(
            lambda x: x if x >= 0 else 0
        )
        X_full['RemodAge'] = X_full['RemodAge'].astype('Float64')

        # RemodStart:
        X_full['RemodStart'] = X_full['YearRemodAdd'] - X_full['YearBuilt']
        X_full['RemodStart'] = X_full['RemodStart'].apply(
            lambda x: x if x >= 0 else 0
        )
        X_full['RemodStart'] = X_full['RemodStart'].astype('Float64')

        # GrAvgRoomSF:
        X_full['GrAvgRoomSF'] = X_full['GrLivArea'] / (
                X_full['BedroomAbvGr'] + X_full['KitchenAbvGr'] + X_full['TotalBathAbGr'] + eps
        )

        # TotalBath:
        X_full['TotalBath'] = 2 * X_full['TotalBathAbGr'] + X_full['TotalBathBsmt']
        X_full['TotalBath'] = X_full['TotalBath'].astype('Int64')

        # HighQualSF:
        X_full['HighQualSF'] = X_full['GrLivArea'] + X_full['1stFlrSF'] + X_full['2ndFlrSF'] + \
                               0.5 * X_full['GarageArea'] + 0.5 * X_full['TotalBsmtSF'] + X_full['MasVnrArea']

        return X_full

    def _check_skewness(self, df: pd.DataFrame, check_skewness_cols):
        X_full = df
        skewness = (X_full[check_skewness_cols].apply(lambda x: skew(x.dropna()))
                    .sort_values(ascending=False))

        skew_df = pd.DataFrame({'skewness': skewness}).reset_index()
        tf_skew_df = skew_df[abs(skew_df['skewness'] > 0.7)]

        skew_cols = tf_skew_df['index'].tolist()

        return skew_cols

    def _pd_process(self):
        # initialize the datasets

        y_train = self.train_data[self.target].copy()
        train_data = self.train_data.drop(self.target, axis=1).copy()
        test_data = self.test_data.copy()
        full_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        X_full = full_data.copy()
        y_train = self.train_data[['Id', 'SalePrice']].copy()

        len_train, len_test = train_data.shape[0], test_data.shape[0]

        # Modify miss matched dtypes:

        change_to_float = [
            'LotArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
            'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
            'PoolArea', 'MiscVal'
        ]
        change_to_int = ['GarageYrBlt']

        X_full['MSSubClass'] = X_full['MSSubClass'].astype('str')
        for col in X_full.columns:
            if X_full[col].dtype == 'float':
                X_full[col] = X_full[col].astype('Float64')
            elif X_full[col].dtype == 'int':
                X_full[col] = X_full[col].astype('Int64')

        for col in change_to_float:
            X_full[col] = X_full[col].astype('Float64')

        for col in change_to_int:
            X_full[col] = X_full[col].astype('Int64')


        # drop outlier
        index_to_drop = self._check_outlier(self.train_data, len_train)
        # print('index_dropped: ', index_to_drop)
        X_full.drop(index=index_to_drop, inplace=True)
        y_train.drop(index=index_to_drop, inplace=True)

        # train_data.drop(index=index_to_drop, inplace=True)
        len_train = len_train - len(list(index_to_drop))   # update the size of train set

        # impute the missing values:
        X_full = self._impute_missing_value(X_full)

        # build new features
        X_full = self._construct_features(X_full)

        # identify zero-inflated feature
        zero_inflated_cols = []
        for col in X_full:
            if X_full[col].dtype == 'Float64':
                if np.sum(X_full[col] == 0) >= 0.4 * X_full.shape[0]:
                    zero_inflated_cols.append(col)

        # create binary features for those zero-inflated_cols
        for col in zero_inflated_cols:
            X_full['has_' + col] = X_full[col].apply(lambda x: 0 if x == 0 else 1)
            X_full['has_' + col] = X_full['has_' + col].astype('Int64')

        # Transform skewed data
        y_train['SalePrice_log'] = np.log1p(y_train['SalePrice'])

        # identify the skewed cols which will be transformed in the pipeline and the cols to be encoded

        check_skew_cols, cols_to_encode = [], []
        for col in X_full.columns:
            if X_full[col].dtype == 'Float64':
                check_skew_cols.append(col)
            elif X_full[col].dtype == 'object':
                cols_to_encode.append(col)
        for col in zero_inflated_cols:
            check_skew_cols.remove(col)

        self.skew_cols = self._check_skewness(X_full, check_skew_cols)
        self.cols_to_encode = cols_to_encode
        self.X_full = X_full.drop(['Id', 'MiscVal'], axis=1)   # remove the Id column
        self.train_X = self.X_full.iloc[:len_train, :]
        self.test_X = self.X_full.iloc[len_train:, :]
        self.train_y = y_train['SalePrice_log']
        self.len_train, self.len_test = len_train, len_test

        return None

    def _build_pipeline(self):
        pt_pipe = Pipeline(
            steps=[
                ('p1', PlusOne()),
                ('pt', PowerTransformer(method='box-cox'))
            ]
        )

        pipe = ColumnTransformer(
            transformers=[
                ('power_trans', pt_pipe, self.skew_cols),
                ('ordinal_encode', OrdinalEncoder(handle_unknown='error'), self.cols_to_encode),
            ], remainder='passthrough'
        )

        return pipe


if __name__ == '__main__':
    config_kwargs= {
        'output_folder': '2022-06-29',
        'raw_data_train': './house_price/data/train.csv',
        'raw_data_test': './house_price/data/test.csv',
        'target': 'SalePrice',
        'project_folder': './house_price/transformed_data'
    }

    p = HousePriceProcess(**config_kwargs)
    p.sk_preprocess()


