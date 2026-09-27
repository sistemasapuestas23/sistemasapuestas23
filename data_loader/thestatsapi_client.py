import time
import logging
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from data_loader.rate_limiter import TokenBucketRateLimiter
from config.settings import THESTATSAPI_API_KEY, THESTATSAPI_BASE_URL, MAX_QPS

logger = logging.getLogger(__name__)


class TheStatsAPIClient:
    """Production TheStatsAPI client with rate limiting, exponential backoff and full coverage of player stats, odds and historicals."""

    def __init__(self):
        self.api_key = THESTATSAPI_API_KEY
        self.base_url = THESTATSAPI_BASE_URL
        self.limiter = TokenBucketRateLimiter(max_qps=MAX_QPS)
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "FootballPredictionEngine/1.0-TheStatsAPI",
            }
        )

    def _request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self.limiter.acquire()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=45)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "5"))
            logger.warning(
                "Rate limited by TheStatsAPI. Sleeping %s seconds", retry_after
            )
            time.sleep(retry_after)
            return self._request(endpoint, params)
        response.raise_for_status()
        return response.json()

    def get_competitions(
        self, page: int = 1, per_page: int = 100, country: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {"page": page, "per_page": per_page}
        if country:
            params["country"] = country
        return self._request("football/competitions", params)

    def get_competition_seasons(self, competition_id: str) -> Dict[str, Any]:
        return self._request(f"football/competitions/{competition_id}/seasons")

    def get_matches(
        self,
        competition_id: Optional[str] = None,
        season_id: Optional[str] = None,
        team_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 100,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"page": page, "per_page": per_page}
        if competition_id:
            params["competition_id"] = competition_id
        if season_id:
            params["season_id"] = season_id
        if team_id:
            params["team_id"] = team_id
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        if status:
            params["status"] = status
        return self._request("football/matches", params)

    def get_match(self, match_id: str) -> Dict[str, Any]:
        return self._request(f"football/matches/{match_id}")

    def get_match_stats(self, match_id: str) -> Dict[str, Any]:
        return self._request(f"football/matches/{match_id}/stats")

    def get_match_player_stats(self, match_id: str) -> Dict[str, Any]:
        return self._request(f"football/matches/{match_id}/player-stats")

    def get_match_odds(self, match_id: str) -> Dict[str, Any]:
        return self._request(f"football/matches/{match_id}/odds")

    def get_match_live_odds(self, match_id: str) -> Dict[str, Any]:
        return self._request(f"football/matches/{match_id}/odds/live")

    def get_match_lineups(self, match_id: str) -> Dict[str, Any]:
        return self._request(f"football/matches/{match_id}/lineups")

    def get_match_timeline(self, match_id: str) -> Dict[str, Any]:
        return self._request(f"football/matches/{match_id}/timeline")

    def get_match_shotmap(self, match_id: str) -> Dict[str, Any]:
        return self._request(f"football/matches/{match_id}/shotmap")

    def get_player(self, player_id: str) -> Dict[str, Any]:
        return self._request(f"football/players/{player_id}")

    def get_player_stats(
        self, player_id: str, season_id: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {}
        if season_id:
            params["season_id"] = season_id
        return self._request(f"football/players/{player_id}/stats", params)

    def get_team_stats(
        self, team_id: str, season_id: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {}
        if season_id:
            params["season_id"] = season_id
        return self._request(f"football/teams/{team_id}/stats", params)

    def get_team_players(self, team_id: str) -> Dict[str, Any]:
        return self._request(f"football/teams/{team_id}/players")

    def get_standings(
        self, competition_id: str, season_id: str, group: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {}
        if group:
            params["group"] = group
        return self._request(
            f"football/competitions/{competition_id}/seasons/{season_id}/standings",
            params,
        )

    def close(self) -> None:
        self.session.close()
