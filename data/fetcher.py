"""MLB Stats API — schedule, lineups, probable pitchers."""

import statsapi
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def get_games_for_date(date_str: str) -> list[dict]:
    """
    Returns a list of game dicts for the given date (MM/DD/YYYY or YYYY-MM-DD).
    Each dict contains teams, probable pitchers, and lineup if available.
    """
    # statsapi accepts YYYY-MM-DD
    if "/" in date_str:
        parts = date_str.split("/")
        date_str = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"

    schedule = statsapi.schedule(
        sportId=1,
        date=date_str,
        hydrate="probablePitcher,lineups,team",
    )

    games = []
    for g in schedule:
        if g.get("status") in ("Final", "Game Over"):
            continue

        game = {
            "game_id": g["game_id"],
            "date": date_str,
            "home_team": g["home_name"],
            "away_team": g["away_name"],
            "home_team_id": g["home_id"],
            "away_team_id": g["away_id"],
            "home_pitcher": _extract_pitcher(g, "home"),
            "away_pitcher": _extract_pitcher(g, "away"),
            "home_lineup": _extract_lineup(g, "home"),
            "away_lineup": _extract_lineup(g, "away"),
            "venue": g.get("venue_name", ""),
        }
        games.append(game)
        logger.info(
            "Game: %s @ %s | SP: %s vs %s | Lineup posted: %s/%s",
            game["away_team"],
            game["home_team"],
            game["away_pitcher"]["name"] if game["away_pitcher"] else "TBD",
            game["home_pitcher"]["name"] if game["home_pitcher"] else "TBD",
            bool(game["away_lineup"]),
            bool(game["home_lineup"]),
        )

    return games


def _extract_pitcher(game: dict, side: str) -> dict | None:
    key = f"{side}_probable_pitcher"
    if not game.get(key):
        return None
    return {
        "name": game[key],
        "id": game.get(f"{side}_probable_pitcher_id"),
    }


def _extract_lineup(game: dict, side: str) -> list[dict]:
    """Return ordered lineup list, empty if not yet posted."""
    batters = game.get(f"{side}_lineup", [])
    if not batters:
        return []
    return [{"name": b, "id": None} for b in batters]


@lru_cache(maxsize=32)
def get_roster(team_id: int) -> list[dict]:
    """Fetch 40-man roster for a team (cached)."""
    roster_data = statsapi.roster(team_id, rosterType="40Man")
    players = []
    for line in roster_data.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            players.append({"name": " ".join(parts[2:]), "number": parts[0], "position": parts[1]})
    return players
