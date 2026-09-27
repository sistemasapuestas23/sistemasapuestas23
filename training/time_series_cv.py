import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Generator
from config.settings import TIME_SERIES_N_SPLITS


def temporal_split_generator(
    n_samples: int, n_splits: int = TIME_SERIES_N_SPLITS
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Strict expanding-window TimeSeriesSplit. Random K-Fold is forbidden."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(n_samples)
    for train_idx, val_idx in tscv.split(indices):
        yield train_idx, val_idx
