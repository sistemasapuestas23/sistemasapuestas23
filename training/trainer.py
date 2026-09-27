import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any
import logging
from models.poisson_dixon_coles import DixonColesPoisson
from models.xgboost_multiclass import XGBoostMulticlass
from models.catboost_multiclass import CatBoostMulticlass
from models.dnn_pytorch import DNNClassifier
from models.lstm_pytorch import LSTMClassifier
from models.lstm_momentum_pytorch import LSTMMomentumClassifier
from models.meta_ensemble import StackingMetaEnsemble
from training.time_series_cv import temporal_split_generator
from config.settings import MODELS_DIR
from utils.metrics import multiclass_log_loss, accuracy

logger = logging.getLogger(__name__)


class FullTrainer:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.meta_ensemble = StackingMetaEnsemble()
        self.oof_probs: np.ndarray = None
        self.feature_columns: list = []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        home_team: np.ndarray,
        away_team: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
    ) -> Dict[str, float]:
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.int64)
        n = len(y_np)
        n_models = 6
        oof = np.zeros((n, n_models * 3))

        poisson = DixonColesPoisson()
        for train_idx, val_idx in temporal_split_generator(n):
            poisson.fit(
                home_team[train_idx], away_team[train_idx],
                home_goals[train_idx], away_goals[train_idx],
            )
            oof[val_idx, 0:3] = poisson.predict_proba(home_team[val_idx], away_team[val_idx])
        poisson.fit(home_team, away_team, home_goals, away_goals)
        self.models["poisson"] = poisson

        for m_idx, (name, Cls) in enumerate(
            [
                ("xgboost", XGBoostMulticlass),
                ("catboost", CatBoostMulticlass),
                ("dnn", DNNClassifier),
                ("lstm", LSTMClassifier),
                ("lstm_momentum", LSTMMomentumClassifier),
            ],
            start=1,
        ):
            model = Cls()
            for train_idx, val_idx in temporal_split_generator(n):
                model.fit(X_np[train_idx], y_np[train_idx])
                oof[val_idx, m_idx * 3 : (m_idx + 1) * 3] = model.predict_proba(X_np[val_idx])
            model.fit(X_np, y_np)
            self.models[name] = model

        self.oof_probs = oof
        self.meta_ensemble.fit(oof, y_np)
        self.models["meta"] = self.meta_ensemble

        meta_pred = self.meta_ensemble.predict_proba(oof)
        metrics = {
            "oof_log_loss": multiclass_log_loss(y_np, meta_pred),
            "oof_accuracy": accuracy(y_np, np.argmax(meta_pred, axis=1)),
        }
        logger.info("OOF Metrics: %s", metrics)
        return metrics

    def save(self, path: Path = None) -> None:
        path = path or MODELS_DIR
        path.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, path / f"{name}.joblib")
        logger.info("All models saved to %s", path)
