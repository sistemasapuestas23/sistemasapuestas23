import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from config.settings import CATBOOST_PARAMS


class CatBoostMulticlass(BaseEstimator, ClassifierMixin):
    def __init__(self, params: dict = None):
        self.params = params or CATBOOST_PARAMS.copy()
        self.model_ = None
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostMulticlass":
        self.model_ = CatBoostClassifier(**self.params)
        self.model_.fit(X, y)
        self.feature_importances_ = self.model_.get_feature_importance()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)
