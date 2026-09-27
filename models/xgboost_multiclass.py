import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from config.settings import XGBOOST_PARAMS


class XGBoostMulticlass(BaseEstimator, ClassifierMixin):
    def __init__(self, params: dict = None):
        self.params = params or XGBOOST_PARAMS.copy()
        self.model_ = None
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostMulticlass":
        dtrain = xgb.DMatrix(X, label=y)
        self.model_ = xgb.train(
            self.params, dtrain, num_boost_round=self.params.get("n_estimators", 400)
        )
        self.feature_importances_ = self.model_.get_score(importance_type="gain")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        return self.model_.predict(dtest)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
