import numpy as np
import pandas as pd
from typing import List


def compute_rolling_stats(
    df: pd.DataFrame,
    group_col: str,
    value_cols: List[str],
    windows: List[int] = [3, 5, 10],
) -> pd.DataFrame:
    """Strictly causal rolling statistics — no future leakage."""
    df = df.sort_values(["date", group_col]).copy()
    for col in value_cols:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            df[f"{col}_roll_std_{w}"] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
            )
    return df


def compute_elo(
    df: pd.DataFrame,
    home_col: str = "home_team_id",
    away_col: str = "away_team_id",
    result_col: str = "result",
    k: float = 20.0,
    initial: float = 1500.0,
) -> pd.DataFrame:
    """Causal Elo rating computation."""
    teams = pd.unique(df[[home_col, away_col]].values.ravel())
    elo = {t: initial for t in teams}
    home_elo = []
    away_elo = []
    for _, row in df.iterrows():
        h, a = row[home_col], row[away_col]
        home_elo.append(elo[h])
        away_elo.append(elo[a])
        expected_h = 1.0 / (1.0 + 10 ** ((elo[a] - elo[h]) / 400.0))
        if row[result_col] == 2:
            score_h = 1.0
        elif row[result_col] == 1:
            score_h = 0.5
        else:
            score_h = 0.0
        elo[h] += k * (score_h - expected_h)
        elo[a] += k * ((1.0 - score_h) - (1.0 - expected_h))
    df = df.copy()
    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    return df
