import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_loader.thestatsapi_client import TheStatsAPIClient
from feature_engineering.feature_builder import FeatureBuilder
from training.trainer import FullTrainer
from inference.predictor import Predictor
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class FullPipeline:
    def __init__(self):
        setup_logging()
        self.client = TheStatsAPIClient()
        self.feature_builder = FeatureBuilder()
        self.trainer = FullTrainer()
        self.predictor = None

    def fetch_historical_matches(
        self,
        competition_id: str,
        season_ids: list,
        max_pages: int = 50,
    ) -> pd.DataFrame:
        rows = []
        for season_id in season_ids:
            page = 1
            while page <= max_pages:
                data = self.client.get_matches(
                    competition_id=competition_id,
                    season_id=season_id,
                    status="finished",
                    page=page,
                    per_page=100,
                )
                matches = data.get("data", [])
                if not matches:
                    break
                for m in matches:
                    home = m.get("home_team", {})
                    away = m.get("away_team", {})
                    match_id = m.get("id")
                    stats = {}
                    odds = {}
                    try:
                        stats_resp = self.client.get_match_stats(match_id)
                        stats = stats_resp.get("data", {})
                    except Exception:
                        pass
                    try:
                        odds_resp = self.client.get_match_odds(match_id)
                        odds = odds_resp.get("data", {})
                    except Exception:
                        pass
                    row = {
                        "match_id": match_id,
                        "date": pd.to_datetime(m.get("utc_date")),
                        "home_team_id": home.get("id"),
                        "away_team_id": away.get("id"),
                        "home_goals": home.get("score", 0) or 0,
                        "away_goals": away.get("score", 0) or 0,
                        "home_xg": stats.get("home", {}).get("xg", 0.0),
                        "away_xg": stats.get("away", {}).get("xg", 0.0),
                        "home_shots": stats.get("home", {}).get("shots", 0),
                        "away_shots": stats.get("away", {}).get("shots", 0),
                        "home_possession": stats.get("home", {}).get(
                            "possession", 50.0
                        ),
                        "away_possession": stats.get("away", {}).get(
                            "possession", 50.0
                        ),
                    }
                    bookmakers = odds.get("bookmakers", [])
                    if bookmakers:
                        markets = bookmakers[0].get("markets", {}).get("match_odds", {})
                        row["odds_home"] = float(
                            markets.get("home", {}).get("last_seen", 2.5) or 2.5
                        )
                        row["odds_draw"] = float(
                            markets.get("draw", {}).get("last_seen", 3.2) or 3.2
                        )
                        row["odds_away"] = float(
                            markets.get("away", {}).get("last_seen", 3.0) or 3.0
                        )
                    else:
                        row["odds_home"] = 2.5
                        row["odds_draw"] = 3.2
                        row["odds_away"] = 3.0
                    rows.append(row)
                page += 1
                if page > data.get("meta", {}).get("total_pages", 1):
                    break
        df = pd.DataFrame(rows)
        df = df.dropna(subset=["date", "home_team_id", "away_team_id"]).sort_values(
            "date"
        )
        return df

    def run_training(self, competition_id: str, season_ids: list) -> dict:
        logger.info("Fetching historical data from TheStatsAPI for %s", competition_id)
        matches = self.fetch_historical_matches(competition_id, season_ids)
        if matches.empty:
            raise ValueError("No historical matches returned from TheStatsAPI")
        X, y = self.feature_builder.build(matches)
        home_team = matches["home_team_id"].values
        away_team = matches["away_team_id"].values
        home_goals = matches["home_goals"].values.astype(int)
        away_goals = matches["away_goals"].values.astype(int)
        metrics = self.trainer.train(X, y, home_team, away_team, home_goals, away_goals)
        self.trainer.save()
        self.predictor = Predictor()
        return metrics

    def predict_upcoming(
        self, competition_id: str, date_from: str, date_to: str
    ) -> pd.DataFrame:
        if self.predictor is None:
            self.predictor = Predictor()
        data = self.client.get_matches(
            competition_id=competition_id,
            date_from=date_from,
            date_to=date_to,
            status="scheduled",
            per_page=50,
        )
        results = []
        for m in data.get("data", []):
            home = m.get("home_team", {})
            away = m.get("away_team", {})
            match_id = m.get("id")
            feat = pd.DataFrame(
                [
                    {
                        "date": pd.to_datetime(m.get("utc_date")),
                        "home_team_id": home.get("id"),
                        "away_team_id": away.get("id"),
                        "home_goals": 0,
                        "away_goals": 0,
                        "home_xg": 0.0,
                        "away_xg": 0.0,
                        "home_shots": 0,
                        "away_shots": 0,
                        "home_possession": 50.0,
                        "away_possession": 50.0,
                        "odds_home": 2.5,
                        "odds_draw": 3.2,
                        "odds_away": 3.0,
                    }
                ]
            )
            try:
                odds_resp = self.client.get_match_odds(match_id)
                bookmakers = odds_resp.get("data", {}).get("bookmakers", [])
                if bookmakers:
                    markets = bookmakers[0].get("markets", {}).get("match_odds", {})
                    feat["odds_home"] = float(
                        markets.get("home", {}).get("last_seen", 2.5) or 2.5
                    )
                    feat["odds_draw"] = float(
                        markets.get("draw", {}).get("last_seen", 3.2) or 3.2
                    )
                    feat["odds_away"] = float(
                        markets.get("away", {}).get("last_seen", 3.0) or 3.0
                    )
            except Exception:
                pass
            pred = self.predictor.predict(
                feat,
                np.array([home.get("id")]),
                np.array([away.get("id")]),
            )
            results.append(
                {
                    "match_id": match_id,
                    "home": home.get("name"),
                    "away": away.get("name"),
                    "kickoff": m.get("utc_date"),
                    **pred,
                }
            )
        return pd.DataFrame(results)

    def close(self) -> None:
        self.client.close()
