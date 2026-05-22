"""MLB Stats API — schedule, lineups, probable pitchers."""

import statsapi
import logging
from datetime import date as _date, timedelta
from functools import lru_cache

logger = logging.getLogger(__name__)

# Statuses that allow lineup extraction from boxscore
_BOXSCORE_STATUSES = {"Final", "Game Over", "In Progress", "Manager challenge",
                      "Delayed", "Delayed: Rain"}


def get_games_for_date(date_str: str) -> list[dict]:
    """
    Returns a list of game dicts for the given date (YYYY-MM-DD).
    Includes all regular-season games regardless of status so historical
    dates work for testing and post-game analysis.
    """
    if "/" in date_str:
        parts = date_str.split("/")
        date_str = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"

    schedule = statsapi.schedule(sportId=1, date=date_str)

    games = []
    for g in schedule:
        if g.get("game_type") != "R":
            continue

        game_id = g["game_id"]
        status = g.get("status", "")

        home_pitcher = _build_pitcher(g, "home")
        away_pitcher = _build_pitcher(g, "away")

        # Fetch lineup from boxscore when game data is available
        if status in _BOXSCORE_STATUSES:
            home_lineup, away_lineup = _lineup_from_boxscore(game_id)
        else:
            home_lineup, away_lineup = [], []

        game = {
            "game_id": game_id,
            "date": date_str,
            "status": status,
            "home_team": g["home_name"],
            "away_team": g["away_name"],
            "home_team_id": g["home_id"],
            "away_team_id": g["away_id"],
            "home_pitcher": home_pitcher,
            "away_pitcher": away_pitcher,
            "home_lineup": home_lineup,
            "away_lineup": away_lineup,
            "venue": g.get("venue_name", ""),
            "home_score": g.get("home_score"),
            "away_score": g.get("away_score"),
        }
        games.append(game)
        logger.info(
            "Game: %s @ %s  [%s]  SP: %s vs %s  Lineup: %s/%s",
            game["away_team"],
            game["home_team"],
            status,
            away_pitcher["name"] if away_pitcher else "TBD",
            home_pitcher["name"] if home_pitcher else "TBD",
            bool(away_lineup),
            bool(home_lineup),
        )

    return games


def _build_pitcher(game: dict, side: str) -> dict | None:
    name = game.get(f"{side}_probable_pitcher", "").strip()
    if not name:
        return None
    player_id = _lookup_player_id(name)
    return {"name": name, "id": player_id}


@lru_cache(maxsize=256)
def _lookup_player_id(name: str) -> int | None:
    try:
        results = statsapi.lookup_player(name)
        if results:
            return results[0]["id"]
    except Exception:
        pass
    return None


def _lineup_from_boxscore(game_id: int) -> tuple[list[dict], list[dict]]:
    """Extract ordered batting lineups from boxscore data."""
    try:
        data = statsapi.get("game", {
            "gamePk": game_id,
            "fields": "liveData,boxscore,teams,batters,battingOrder,players,person,id,fullName",
        })
        if not data:
            return [], []
        box = data.get("liveData", {}).get("boxscore", {}).get("teams", {})

        home_lineup = _parse_side_lineup(box.get("home", {}))
        away_lineup = _parse_side_lineup(box.get("away", {}))
        return home_lineup, away_lineup
    except Exception as exc:
        logger.debug("Could not fetch boxscore for game %s: %s", game_id, exc)
        return [], []


def get_projected_lineup(team_id: int, before_date_str: str) -> list[dict]:
    """
    Return the most recent starting lineup for team_id by scanning back
    through completed games before before_date_str (up to 7 days).
    Returns an empty list if nothing is found.
    """
    d = _date.fromisoformat(before_date_str) - timedelta(days=1)
    for _ in range(7):
        try:
            schedule = statsapi.schedule(sportId=1, date=d.isoformat(), team=team_id)
        except Exception:
            d -= timedelta(days=1)
            continue
        for g in schedule:
            if g.get("game_type") != "R":
                continue
            if g.get("status") not in {"Final", "Game Over"}:
                continue
            game_id = g["game_id"]
            home_lineup, away_lineup = _lineup_from_boxscore(game_id)
            lineup = home_lineup if g["home_id"] == team_id else away_lineup
            if lineup:
                logger.debug("Projected lineup for team %s from %s game %s",
                             team_id, d.isoformat(), game_id)
                return lineup
        d -= timedelta(days=1)
    return []


def _parse_side_lineup(side: dict) -> list[dict]:
    """Return starters sorted by batting order (100, 200, ..., 900)."""
    players = side.get("players", {})
    batters = side.get("batters", [])

    ordered = []
    for pid in batters:
        key = f"ID{pid}"
        p = players.get(key, {})
        order_str = p.get("battingOrder", "")
        if not order_str or not order_str.isdigit():
            continue
        order = int(order_str)
        # Only include starters (batting order ends in 00: 100, 200, ..., 900)
        if order % 100 == 0:
            ordered.append({
                "name": p.get("person", {}).get("fullName", ""),
                "id": pid,
                "batting_order": order // 100,
            })

    ordered.sort(key=lambda x: x["batting_order"])
    return ordered


