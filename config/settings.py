import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

THESTATSAPI_API_KEY = os.environ.get("THESTATSAPI_API_KEY")
if not THESTATSAPI_API_KEY:
    raise EnvironmentError("THESTATSAPI_API_KEY environment variable is required")

THESTATSAPI_BASE_URL = os.environ.get(
    "THESTATSAPI_BASE_URL", "https://api.thestatsapi.com/api"
).rstrip("/")
MAX_QPS = float(os.environ.get("MAX_QPS", "2.0"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

POISSON_MAX_ITER = 500
XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}
CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
    "verbose": False,
}
DNN_HIDDEN = [256, 128, 64]
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_MOMENTUM_WINDOW = 5
META_LEARNER = "logistic"
TIME_SERIES_N_SPLITS = 5
RANDOM_SEED = 42
