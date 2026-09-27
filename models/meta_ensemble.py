import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from lightgbm import LGBMClassifier
from config.settings import META_LEARNER, RANDOM_SEED


class StackingMetaEnsemble(BaseEstimator, ClassifierMixin):
    """Out-of-fold stacking meta-learner. Strictly temporal OOF probabilities only."""

    def __init__(self, meta_learner: str = META_LEARNER):
        self.meta_learner = meta_learner
        self.meta_model_ = None

    def fit(self, oof_probs: np.ndarray, y: np.ndarray) -> "StackingMetaEnsemble":
        if self.meta_learner == "logistic":
            self.meta_model_ = LogisticRegression(
                multi_class="multinomial",
                max_iter=1000,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        else:
            self.meta_model_ = LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=200,
                learning_rate=0.05,
                random_state=RANDOM_SEED,
                verbosity=-1,
            )
        self.meta_model_.fit(oof_probs, y)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        return self.meta_model_.predict_proba(probs)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self.meta_model_.predict(probs)
