import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class DixonColesPoisson:
    """Independent Poisson + Dixon-Coles low-score adjustment via MLE."""

    def __init__(self, max_iter: int = 500):
        self.max_iter = max_iter
        self.params_: Dict = {}
        self.teams_: list = []

    def _log_likelihood(
        self,
        theta: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        n_teams: int,
    ) -> float:
        attack = theta[:n_teams]
        defence = theta[n_teams : 2 * n_teams]
        home_adv = theta[2 * n_teams]
        rho = theta[2 * n_teams + 1]
        mu_h = np.exp(attack[home_idx] + defence[away_idx] + home_adv)
        mu_a = np.exp(attack[away_idx] + defence[home_idx])
        ll = poisson.logpmf(home_goals, mu_h) + poisson.logpmf(away_goals, mu_a)
        tau = np.ones_like(ll)
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_11 = (home_goals == 1) & (away_goals == 1)
        tau[mask_00] = 1 - mu_h[mask_00] * mu_a[mask_00] * rho
        tau[mask_10] = 1 + mu_a[mask_10] * rho
        tau[mask_01] = 1 + mu_h[mask_01] * rho
        tau[mask_11] = 1 - rho
        ll = ll + np.log(np.clip(tau, 1e-10, None))
        return -np.sum(ll)

    def fit(
        self,
        home_team: np.ndarray,
        away_team: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
    ) -> "DixonColesPoisson":
        teams = np.unique(np.concatenate([home_team, away_team]))
        self.teams_ = list(teams)
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        home_idx = np.array([team_to_idx[t] for t in home_team])
        away_idx = np.array([team_to_idx[t] for t in away_team])
        theta0 = np.zeros(2 * n + 2)
        theta0[:n] = 0.1
        theta0[n : 2 * n] = -0.1
        theta0[2 * n] = 0.3
        theta0[2 * n + 1] = -0.1
        res = minimize(
            self._log_likelihood,
            theta0,
            args=(home_goals, away_goals, home_idx, away_idx, n),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        if not res.success:
            logger.warning(
                "Dixon-Coles optimization did not fully converge: %s", res.message
            )
        self.params_ = {
            "attack": res.x[:n],
            "defence": res.x[n : 2 * n],
            "home_adv": res.x[2 * n],
            "rho": res.x[2 * n + 1],
            "team_to_idx": team_to_idx,
        }
        return self

    def predict_proba(self, home_team: np.ndarray, away_team: np.ndarray) -> np.ndarray:
        n_matches = len(home_team)
        probs = np.zeros((n_matches, 3))
        attack = self.params_["attack"]
        defence = self.params_["defence"]
        home_adv = self.params_["home_adv"]
        team_to_idx = self.params_["team_to_idx"]
        for i in range(n_matches):
            h = team_to_idx.get(home_team[i])
            a = team_to_idx.get(away_team[i])
            if h is None or a is None:
                probs[i] = [1 / 3, 1 / 3, 1 / 3]
                continue
            mu_h = np.exp(attack[h] + defence[a] + home_adv)
            mu_a = np.exp(attack[a] + defence[h])
            max_g = 10
            p_matrix = np.outer(
                poisson.pmf(np.arange(max_g + 1), mu_h),
                poisson.pmf(np.arange(max_g + 1), mu_a),
            )
            p_home = np.sum(np.tril(p_matrix, -1))
            p_draw = np.sum(np.diag(p_matrix))
            p_away = np.sum(np.triu(p_matrix, 1))
            total = p_home + p_draw + p_away + 1e-12
            probs[i] = [p_away / total, p_draw / total, p_home / total]
        return probs
