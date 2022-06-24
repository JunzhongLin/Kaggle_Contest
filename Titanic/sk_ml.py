from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
from joblib import dump, load
import os
from src.base_models import SuperviseModel
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class TitanicXgbClassifier(SuperviseModel):

    def __init__(self, data_path):
        super().__init__()
        self.model = {}
        self.data_path = data_path
        self._init_data()
        self._init_classifier()
        self.searchCV: GridSearchCV = None

    def _init_data(self):
        self.X_train = load(os.path.join(self.data_path, 'X_train.pkl'))
        self.X_test = load(os.path.join(self.data_path, 'X_test.pkl'))
        self.y_train = load(os.path.join(self.data_path, 'y_train.pkl'))

    def _init_classifier(self):
        xgb_cls = xgb.XGBClassifier(
            objective='binary:logistic', colsample_bytree=0.5, learning_rate=0.05,
            max_depth=15, alpha=5, n_estimators=500, min_child_weight=7, gamma=0,
            eval_metric=accuracy_score
        )
        self.model['xgb_cls'] = xgb_cls

    def res_visualize(self):
        pass

    def search_hyper_params(self, model_name, hyper_params):
        searchCV = GridSearchCV(self.model[model_name], hyper_params, n_jobs=-1, refit=True)
        searchCV.fit(self.X_train, self.y_train)
        self.searchCV = searchCV

        return None

    def prepare_submission(self):
        y_pred = self.searchCV.predict(self.X_test)
        test_df = pd.read_csv('/Titanic/data/test.csv')
        submission_df = pd.DataFrame(
            {
                'PassengerId': test_df['PassengerId'],
                'Survived': y_pred
            }
        )
        submission_df.to_csv('./Titanic/submission/submission.csv', index=False)

        return None


class TitanicRfClassifier(SuperviseModel):
    '''

    '''

    def __init__(self, data_path):

        super(TitanicRfClassifier, self).__init__()
        self.data_path = data_path
        self.model = {}
        self._init_data()
        self._init_classifier()

    def _init_data(self):

        self.X_train = load(os.path.join(self.data_path, 'X_train.pkl'))
        self.X_test = load(os.path.join(self.data_path, 'X_test.pkl'))
        self.y_train = load(os.path.join(self.data_path, 'y_train.pkl'))

    def _init_classifier(self):

        rf_cls = RandomForestClassifier(

        )
        self.model['rf_cls'] = rf_cls


if __name__=='__main__':
    xgb_cls = TitanicXgbClassifier('./Titanic/transformed_data/fourth_try')
    xgb_cls_hyperparams = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [300, 500, 800],
        'max_depth': [10, 15, 20],
        'colsample_bytree': [0.5, 0.6, 0.7],
        'alpha': [0, 5, 10, 20],
        'min_child_weight': [1, 7, 15]

    }
    searchCV = GridSearchCV(xgb_cls.model['xgb_cls'], xgb_cls_hyperparams, refit=True, n_jobs=-1)
    searchCV.fit(xgb_cls.X_train, xgb_cls.y_train)

    print("Best parameter (CV score=%0.3f):" % searchCV.best_score_)
    print(searchCV.best_params_)

    # '{'alpha': 0, 'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 15, 'n_estimators': 800}'

    '''
        score = cross_validate(xgb_cls.model['xgb_cls'], xgb_cls.X_train, xgb_cls.y_train,
                           return_train_score=True)
    print('cv_score from xgb: \n')
    print(
        'train_score: ', score['train_score']
    )
    print(' ')
    print(
        'test_score: ', score['test_score']
    )
    rf_cls = TitanicRfClassifier('./Titanic/transformed_data/fourth_try')
    
    '''





