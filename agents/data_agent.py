"""
Data Agent — fetches and stores all daily inputs for the model.

Responsibilities:
  - Game schedule + lineups (MLB Stats API)
  - Player profiles: pitcher stats, batter stats, bullpen (Statcast + MLB API)
  - Pre-game sportsbook odds (ESPN / Odds API) — saved ONCE, never overwritten
  - Wind / weather context (Open-Meteo)

Run:
  python3 agents/data_agent.py --date 2026-05-22
  python3 agents/data_agent.py --date 2026-05-22 --odds-key YOUR_KEY

The agent is idempotent: re-running for the same date skips already-stored data
EXCEPT odds, which are always skipped once written (backtest accuracy guarantee).
"""

import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetcher import get_games_for_date
from data.player_stats import get_batter_profile, get_pitcher_profile, get_bullpen_profile
from data.odds import fetch_odds, resolve_game_odds
from data.weather import get_wind_context
from model.simulator import build_bullpen_profile
import store
import config
from config import PARK_HR_FACTORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [data]  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_LINEUP_SIZE = 9


def parse_args():
    p = argparse.ArgumentParser(description="MLB Data Agent")
    p.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: today)")
    p.add_argument("--odds-key", default=None)
    return p.parse_args()


def _default_lineup(team_name: str) -> list:
    return [{"name": f"{team_name} Batter {i+1}", "id": None} for i in range(_DEFAULT_LINEUP_SIZE)]


def run(date: str, odds_key: str | None = None) -> dict:
    """
    Fetch and store all data for the given date.
    Returns: {"games": [...], "profiles": {...}, "odds": {...}, "weather": {...}}
    """
    from datetime import date as date_cls
    if not date:
        date = date_cls.today().isoformat()

    odds_api_key = odds_key or config.ODDS_API_KEY

    logger.info("=== Data Agent | %s ===", date)

    # ── 1. Schedule ──────────────────────────────────────────────────────────
    logger.info("Fetching schedule...")
    games = get_games_for_date(date)
    if not games:
        logger.warning("No games found for %s", date)
        return {"games": [], "profiles": {}, "odds": {}, "weather": {}}

    store.write_schedule(date, _serialize_games(games))
    logger.info("Stored schedule: %d games", len(games))

    # ── 2. Pre-game odds (written once, never overwritten) ───────────────────
    if store.odds_saved(date):
        logger.info("Odds already stored for %s — skipping fetch (backtest integrity)", date)
        odds_data = {}
    else:
        logger.info("Fetching pre-game odds...")
        odds_data = fetch_odds(api_key=odds_api_key, date_str=date)
        odds_by_game_id = _odds_by_game_id(games, odds_data)
        written = store.write_odds(date, odds_by_game_id)
        logger.info("Odds stored: %d games (%s)", len(odds_by_game_id),
                    "written" if written else "already existed")

    # ── 3. Player profiles + weather ─────────────────────────────────────────
    logger.info("Enriching %d games with player profiles + weather...", len(games))
    profiles = {}
    weather_map = {}

    for game in games:
        gid = str(game["game_id"])
        logger.info("  %s @ %s", game["away_team"], game["home_team"])

        # Pitchers
        hp = game["home_pitcher"]
        ap = game["away_pitcher"]
        home_pitcher_profile = get_pitcher_profile(hp["name"], hp.get("id")) if hp else get_pitcher_profile("Unknown", None)
        away_pitcher_profile = get_pitcher_profile(ap["name"], ap.get("id")) if ap else get_pitcher_profile("Unknown", None)

        # Bullpens
        home_bp_stats = get_bullpen_profile(game["home_team"])
        away_bp_stats = get_bullpen_profile(game["away_team"])
        home_bullpen_profile = build_bullpen_profile(home_bp_stats)
        away_bullpen_profile = build_bullpen_profile(away_bp_stats)

        # Lineups
        home_lineup = game["home_lineup"] or _default_lineup(game["home_team"])
        away_lineup = game["away_lineup"] or _default_lineup(game["away_team"])
        home_lineup_profiles = [get_batter_profile(b["name"], b.get("id")) for b in home_lineup]
        away_lineup_profiles = [get_batter_profile(b["name"], b.get("id")) for b in away_lineup]

        # Weather
        wind = get_wind_context(game.get("venue", ""), date)
        logger.info("    Wind: %s", wind["description"])

        park_hr_factor = PARK_HR_FACTORS.get(game["home_team"], 1.0)
        profiles[gid] = {
            "home_pitcher_profile":  home_pitcher_profile,
            "away_pitcher_profile":  away_pitcher_profile,
            "home_bullpen_profile":  home_bullpen_profile,
            "away_bullpen_profile":  away_bullpen_profile,
            "home_lineup":           home_lineup,
            "away_lineup":           away_lineup,
            "home_lineup_profiles":  home_lineup_profiles,
            "away_lineup_profiles":  away_lineup_profiles,
            "outward_wind_mph":      wind["outward_wind_mph"],
            "wind":                  wind,
            "park_hr_factor":        park_hr_factor,
        }
        weather_map[gid] = wind

    store.write_player_profiles(date, profiles)
    store.write_weather(date, weather_map)
    logger.info("Player profiles + weather stored.")

    return {"games": games, "profiles": profiles, "odds": odds_data, "weather": weather_map}


def _serialize_games(games: list) -> list:
    """Keep only JSON-serializable fields from each game dict."""
    _safe = (str, int, float, bool, list, dict, type(None))
    return [{k: v for k, v in g.items() if isinstance(v, _safe)} for g in games]


def _odds_by_game_id(games: list, odds_data: dict) -> dict:
    """Map game_id → odds entry by matching team names."""
    result = {}
    for game in games:
        entry = resolve_game_odds(odds_data, game["home_team"], game["away_team"])
        if entry:
            result[str(game["game_id"])] = entry
    return result


if __name__ == "__main__":
    args = parse_args()
    run(date=args.date, odds_key=args.odds_key)
