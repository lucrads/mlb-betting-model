"""MLB Stats API — schedule, lineups, probable pitchers."""

import statsapi
import logging
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
        box = data.get("liveData", {}).get("boxscore", {}).get("teams", {})

        home_lineup = _parse_side_lineup(box.get("home", {}))
        away_lineup = _parse_side_lineup(box.get("away", {}))
        return home_lineup, away_lineup
    except Exception as exc:
        logger.debug("Could not fetch boxscore for game %s: %s", game_id, exc)
        return [], []


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


@lru_cache(maxsize=32)
def get_roster(team_id: int) -> list[dict]:
    """Fetch 40-man roster for a team (cached)."""
    roster_data = statsapi.roster(team_id, rosterType="40Man")
    players = []
    for line in roster_data.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            players.append({
                "name": " ".join(parts[2:]),
                "number": parts[0],
                "position": parts[1],
            })
    return players
