import pandas as pd
import numpy as np
from typing import Tuple
from feature_engineering.temporal_features import compute_rolling_stats, compute_elo


class FeatureBuilder:
    """Builds strictly temporal feature matrices including player aggregates, xG and odds-derived features with zero data leakage."""

    def __init__(self):
        self.feature_columns: list = []

    def build(self, matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Expected columns (from TheStatsAPI enrichment):
        date, home_team_id, away_team_id, home_goals, away_goals,
        home_xg, away_xg, home_shots, away_shots, home_possession,
        away_possession, odds_home, odds_draw, odds_away,
        home_player_goals_avg, away_player_goals_avg, ...
        """
        df = matches.sort_values("date").reset_index(drop=True).copy()
        df["result"] = np.where(
            df["home_goals"] > df["away_goals"],
            2,
            np.where(df["home_goals"] < df["away_goals"], 0, 1),
        )
        df = compute_elo(df)

        value_cols = [
            "home_goals", "away_goals", "home_xg", "away_xg",
            "home_shots", "away_shots", "home_possession", "away_possession",
        ]
        for col in value_cols:
            if col not in df.columns:
                df[col] = 0.0

        df = compute_rolling_stats(df, "home_team_id", value_cols)
        df = compute_rolling_stats(df, "away_team_id", value_cols)

        df["goal_diff_home"] = df["home_goals"] - df["away_goals"]
        df["xg_diff"] = df.get("home_xg", 0) - df.get("away_xg", 0)
        df["home_form_3"] = df.groupby("home_team_id")["result"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        df["away_form_3"] = df.groupby("away_team_id")["result"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

        if "odds_home" in df.columns:
            df["implied_home"] = 1.0 / df["odds_home"].clip(lower=1.01)
            df["implied_draw"] = 1.0 / df["odds_draw"].clip(lower=1.01)
            df["implied_away"] = 1.0 / df["odds_away"].clip(lower=1.01)
            total_imp = df["implied_home"] + df["implied_draw"] + df["implied_away"]
            df["implied_home"] /= total_imp
            df["implied_draw"] /= total_imp
            df["implied_away"] /= total_imp

        feature_cols = [
            c for c in df.columns
            if any(k in c for k in [
                "elo", "roll_mean", "roll_std", "form", "goal_diff",
                "xg", "shots", "possession", "implied", "player"
            ])
        ]
        self.feature_columns = feature_cols
        X = df[feature_cols].fillna(0.0)
        y = df["result"].astype(int)
        return X, y
