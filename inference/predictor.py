import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any
from config.settings import MODELS_DIR
from feature_engineering.feature_builder import FeatureBuilder


class Predictor:
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models = {}
        self.feature_builder = FeatureBuilder()
        self._load_models()

    def _load_models(self) -> None:
        for name in ["poisson", "xgboost", "catboost", "dnn", "lstm", "lstm_momentum", "meta"]:
            path = self.models_dir / f"{name}.joblib"
            if path.exists():
                self.models[name] = joblib.load(path)

    def predict(self, match_features: pd.DataFrame, home_team: np.ndarray, away_team: np.ndarray) -> Dict[str, Any]:
        X, _ = self.feature_builder.build(match_features)
        X_np = X.values.astype(np.float32)
        probs_list = []
        if "poisson" in self.models:
            probs_list.append(self.models["poisson"].predict_proba(home_team, away_team))
        for name in ["xgboost", "catboost", "dnn", "lstm", "lstm_momentum"]:
            if name in self.models:
                probs_list.append(self.models[name].predict_proba(X_np))
        stacked = np.hstack(probs_list)
        if "meta" in self.models:
            final_probs = self.models["meta"].predict_proba(stacked)
        else:
            final_probs = np.mean(probs_list, axis=0)
        return {
            "prob_away": float(final_probs[0, 0]),
            "prob_draw": float(final_probs[0, 1]),
            "prob_home": float(final_probs[0, 2]),
            "prediction": int(np.argmax(final_probs[0])),
        }
