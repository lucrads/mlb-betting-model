"""
Backtest runner — compares model predictions against actual MLB results.

Usage:
  python3 backtest.py                          # full season up to yesterday
  python3 backtest.py --start 2026-04-01       # from a specific date
  python3 backtest.py --sims 200               # override sim count (default 200)
  python3 backtest.py --refresh                # rerun all games, overwriting cache

Results are saved to backtest_results.json (incremental — skips already-processed games).
"""

import sys
import os
import json
import logging
import argparse
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from data.fetcher import get_games_for_date
from data.odds import fetch_odds, resolve_game_odds
from data.player_stats import get_batter_profile, get_pitcher_profile, get_bullpen_profile
from data.weather import get_wind_context
from model.simulator import build_bullpen_profile
from model.monte_carlo import run_simulations
from output.edge_calc import compute_edge
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "backtest_results.json")
SEASON_START = f"{config.CURRENT_SEASON}-03-20"
_FINAL_STATUSES = {"Final", "Game Over"}


def parse_args():
    p = argparse.ArgumentParser(description="MLB Model Backtest Runner")
    p.add_argument("--start", default=SEASON_START, help="Start date YYYY-MM-DD")
    p.add_argument("--end",   default=None,         help="End date YYYY-MM-DD (default: yesterday)")
    p.add_argument("--sims",  type=int, default=200, help="Simulations per game (default 200)")
    p.add_argument("--refresh", action="store_true", help="Reprocess all games (ignore cache)")
    return p.parse_args()


def load_results() -> list[dict]:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results: list[dict]) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _enrich_game(game: dict) -> dict:
    """Load player profiles and wind data into a game dict (mirrors main.py)."""
    from main import build_default_lineup

    if game["home_pitcher"]:
        game["home_pitcher_profile"] = get_pitcher_profile(
            game["home_pitcher"]["name"], game["home_pitcher"].get("id")
        )
    else:
        game["home_pitcher_profile"] = get_pitcher_profile("Unknown", None)

    if game["away_pitcher"]:
        game["away_pitcher_profile"] = get_pitcher_profile(
            game["away_pitcher"]["name"], game["away_pitcher"].get("id")
        )
    else:
        game["away_pitcher_profile"] = get_pitcher_profile("Unknown", None)

    home_bullpen_stats = get_bullpen_profile(game["home_team"])
    away_bullpen_stats = get_bullpen_profile(game["away_team"])
    game["home_bullpen_profile"] = build_bullpen_profile(home_bullpen_stats)
    game["away_bullpen_profile"] = build_bullpen_profile(away_bullpen_stats)

    if not game["home_lineup"]:
        game["home_lineup"] = build_default_lineup(game["home_team"])
    if not game["away_lineup"]:
        game["away_lineup"] = build_default_lineup(game["away_team"])

    game["home_lineup_profiles"] = [
        get_batter_profile(b["name"], b.get("id")) for b in game["home_lineup"]
    ]
    game["away_lineup_profiles"] = [
        get_batter_profile(b["name"], b.get("id")) for b in game["away_lineup"]
    ]

    wind = get_wind_context(game.get("venue", ""), game["date"])
    game["wind"] = wind
    game["outward_wind_mph"] = wind["outward_wind_mph"]
    return game


def _process_game(game: dict, odds_data: dict) -> dict:
    """Run model on one completed game and return a result record."""
    game = _enrich_game(game)
    sim = run_simulations(game)

    game_odds = resolve_game_odds(odds_data, game["home_team"], game["away_team"])
    edge = compute_edge(sim, game_odds)

    actual_home = int(game.get("home_score") or 0)
    actual_away = int(game.get("away_score") or 0)
    home_won = actual_home > actual_away

    model_favors_home = sim["home_win_pct"] >= sim["away_win_pct"]
    model_correct_ml = model_favors_home == home_won

    # Confidence-based classification (used when no live odds exist)
    confidence = max(sim["home_win_pct"], sim["away_win_pct"])
    if confidence >= 0.70:
        confidence_rec = "HIGH"
    elif confidence >= 0.60:
        confidence_rec = "MED"
    elif confidence >= 0.55:
        confidence_rec = "LEAN"
    else:
        confidence_rec = "PASS"

    # Track per-bet outcomes for edge-range analysis
    bet_results = []
    for bet in edge.get("best_bets", []):
        if bet["side_key"] == "home_team":
            won = home_won
        elif bet["side_key"] == "away_team":
            won = not home_won
        elif bet["market"] == "over" and edge.get("book_total") is not None:
            won = (actual_home + actual_away) > edge["book_total"]
        elif bet["market"] == "under" and edge.get("book_total") is not None:
            won = (actual_home + actual_away) < edge["book_total"]
        else:
            won = None
        if won is not None:
            bet_results.append({"market": bet["market"], "edge": bet["edge"], "won": won})

    return {
        "date":             game["date"],
        "game_id":          game["game_id"],
        "home_team":        game["home_team"],
        "away_team":        game["away_team"],
        "home_win_pct":     sim["home_win_pct"],
        "away_win_pct":     sim["away_win_pct"],
        "model_total":      sim["avg_total_runs"],
        "actual_home_runs": actual_home,
        "actual_away_runs": actual_away,
        "home_won":         home_won,
        "actual_total":     actual_home + actual_away,
        "model_favors_home": model_favors_home,
        "model_correct_ml":  model_correct_ml,
        "confidence":        round(confidence, 4),
        "confidence_rec":    confidence_rec,
        "recommendation":    edge["recommendation"],
        "home_edge":        edge.get("home_edge"),
        "away_edge":        edge.get("away_edge"),
        "total_edge_over":  edge.get("total_edge_over"),
        "total_edge_under": edge.get("total_edge_under"),
        "book_total":       edge.get("book_total"),
        "had_odds":         edge.get("has_odds", False),
        "bet_results":      bet_results,
    }


def run_backtest(start_date: str, end_date: str, sims: int, refresh: bool) -> list[dict]:
    config.NUM_SIMULATIONS = sims

    results = [] if refresh else load_results()
    processed_ids = {r["game_id"] for r in results}

    current = date.fromisoformat(start_date)
    end     = date.fromisoformat(end_date)

    total_new = 0
    while current <= end:
        date_str = current.isoformat()
        current += timedelta(days=1)

        try:
            games = get_games_for_date(date_str)
        except Exception as e:
            logger.warning("Could not fetch schedule for %s: %s", date_str, e)
            continue

        final_games = [
            g for g in games
            if g["status"] in _FINAL_STATUSES
            and g["home_score"] is not None
            and g["game_id"] not in processed_ids
        ]
        if not final_games:
            continue

        # ESPN historical odds are unavailable for Final games — pass empty dict
        odds_data: dict = {}

        for game in final_games:
            try:
                rec = _process_game(game, odds_data)
                results.append(rec)
                processed_ids.add(game["game_id"])
                total_new += 1
                logger.info(
                    "  ✓ %s @ %s  [%s-%s]  model=%s  correct=%s",
                    game["away_team"], game["home_team"],
                    rec["actual_away_runs"], rec["actual_home_runs"],
                    "HOME" if rec["model_favors_home"] else "AWAY",
                    rec["model_correct_ml"],
                )
            except Exception as e:
                logger.error("  ✗ %s @ %s: %s", game["away_team"], game["home_team"], e)

        save_results(results)
        logger.info("Saved %d total results after %s", len(results), date_str)

    logger.info("=== Backtest complete — %d new games processed ===", total_new)
    return results


def main():
    args = parse_args()
    end_date = args.end or (date.today() - timedelta(days=1)).isoformat()

    logger.info(
        "=== Backtest | %s → %s | %d sims ===",
        args.start, end_date, args.sims,
    )
    results = run_backtest(args.start, end_date, args.sims, args.refresh)

    if results:
        ml_correct = sum(1 for r in results if r["model_correct_ml"])
        logger.info(
            "ML accuracy: %d / %d = %.1f%%",
            ml_correct, len(results), 100 * ml_correct / len(results),
        )


if __name__ == "__main__":
    main()
