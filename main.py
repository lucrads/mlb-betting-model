"""
MLB Predictive Betting Model

Usage:
  python main.py --date 2025-05-21 --odds-key YOUR_KEY

  # Use ODDS_API_KEY env var instead of --odds-key flag:
  ODDS_API_KEY=xxx python main.py --date 2025-05-21

  # Skip odds comparison (just model output):
  python main.py --date 2025-05-21
"""

import sys
import os
import logging
import argparse
from datetime import date

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from data.fetcher import get_games_for_date
from data.player_stats import (
    get_batter_profile,
    get_pitcher_profile,
    get_bullpen_profile,
)
from data.odds import fetch_odds, resolve_game_odds
from data.weather import get_wind_context
from model.simulator import build_bullpen_profile
from model.monte_carlo import run_simulations
from output.edge_calc import compute_edge, prob_to_american
from output.matchup_details import compute_matchup_details
from output.report import generate_report
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default lineup to use when official lineup not yet posted
_DEFAULT_LINEUP_SIZE = 9


def parse_args():
    parser = argparse.ArgumentParser(description="MLB Predictive Betting Model")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date to simulate (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--odds-key",
        default=None,
        help="The Odds API key. Can also be set via ODDS_API_KEY env var.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write the HTML report.",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=None,
        help=f"Number of simulations per game (default: {config.NUM_SIMULATIONS}).",
    )
    return parser.parse_args()


def build_default_lineup(team_name: str) -> list[dict]:
    """Placeholder lineup using league-average batter profiles when lineup not posted."""
    return [
        {"name": f"{team_name} Batter {i+1}", "id": None}
        for i in range(_DEFAULT_LINEUP_SIZE)
    ]


def enrich_game(game: dict) -> dict:
    """Load all player profiles needed for simulation into the game dict."""
    # Pitchers
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

    # Bullpens
    home_bullpen_stats = get_bullpen_profile(game["home_team"])
    away_bullpen_stats = get_bullpen_profile(game["away_team"])
    game["home_bullpen_profile"] = build_bullpen_profile(home_bullpen_stats)
    game["away_bullpen_profile"] = build_bullpen_profile(away_bullpen_stats)

    # Lineups
    if not game["home_lineup"]:
        logger.info("No lineup posted for %s — using default.", game["home_team"])
        game["home_lineup"] = build_default_lineup(game["home_team"])

    if not game["away_lineup"]:
        logger.info("No lineup posted for %s — using default.", game["away_team"])
        game["away_lineup"] = build_default_lineup(game["away_team"])

    game["home_lineup_profiles"] = [
        get_batter_profile(b["name"], b.get("id")) for b in game["home_lineup"]
    ]
    game["away_lineup_profiles"] = [
        get_batter_profile(b["name"], b.get("id")) for b in game["away_lineup"]
    ]

    # Wind context
    wind = get_wind_context(game.get("venue", ""), game["date"])
    game["wind"] = wind
    game["outward_wind_mph"] = wind["outward_wind_mph"]
    logger.info("Wind @ %s: %s", game.get("venue", "?"), wind["description"])

    return game


def main():
    args = parse_args()

    # Override sim count if provided
    if args.sims:
        config.NUM_SIMULATIONS = args.sims

    # API key: CLI flag > env var
    odds_api_key = args.odds_key or config.ODDS_API_KEY

    logger.info("=== MLB Model  |  Date: %s  |  Sims: %d ===", args.date, config.NUM_SIMULATIONS)

    # 1. Fetch today's games
    logger.info("Fetching schedule...")
    games = get_games_for_date(args.date)
    if not games:
        logger.warning("No games found for %s.", args.date)
        return

    logger.info("Found %d game(s).", len(games))

    # 2. Fetch sportsbook odds
    logger.info("Fetching sportsbook odds...")
    odds_data = fetch_odds(api_key=odds_api_key, date_str=args.date)

    # 3. Load player stats and simulate each game
    game_results = []
    for game in games:
        logger.info("--- %s @ %s ---", game["away_team"], game["home_team"])

        # Enrich with player profiles
        game = enrich_game(game)

        # Run simulations
        simulation = run_simulations(game)

        # Resolve sportsbook odds for this game
        game_odds = resolve_game_odds(odds_data, game["home_team"], game["away_team"])
        if not game_odds:
            logger.info("No odds found for %s @ %s", game["away_team"], game["home_team"])

        # Compute edge
        edge = compute_edge(simulation, game_odds)

        # Compute per-batter matchup details for the detail view
        details = compute_matchup_details(game)

        game_results.append({
            "game": game,
            "simulation": simulation,
            "edge": edge,
            "details": details,
        })

        _print_game_summary(game, simulation, edge)

    # 4. Generate HTML report
    report_path = generate_report(args.date, game_results, output_dir=args.output_dir)
    logger.info("=== Report written: %s ===", report_path)


def _fmt_ml(odds: int | None) -> str:
    if odds is None:
        return "N/A"
    return f"+{odds}" if odds > 0 else str(odds)


def _print_game_summary(game: dict, sim: dict, edge: dict) -> None:
    home = game["home_team"]
    away = game["away_team"]
    rec = edge["recommendation"]
    wind = game.get("wind", {})

    # Model-implied American odds
    model_away_ml = prob_to_american(sim["away_win_pct"])
    model_home_ml = prob_to_american(sim["home_win_pct"])

    print(f"\n  {away} @ {home}")
    print(f"  Model:   {away} {_fmt_ml(model_away_ml)}  |  {home} {_fmt_ml(model_home_ml)}")

    if edge["has_odds"]:
        bh = edge.get("best_home_ml")
        ba = edge.get("best_away_ml")
        bh_bk = edge.get("best_home_book", "")
        ba_bk = edge.get("best_away_book", "")
        print(f"  Best:    {away} {_fmt_ml(ba)} ({ba_bk})  |  {home} {_fmt_ml(bh)} ({bh_bk})")

        if edge["home_edge"] is not None:
            # Show edge as implied-odds difference in plain language
            better_side = home if (edge["home_edge"] or 0) >= (edge["away_edge"] or 0) else away
            better_edge = max(edge["home_edge"] or 0, edge["away_edge"] or 0)
            print(f"  ML Edge: {better_side} {better_edge*100:+.1f}%")

        if edge["total_edge_over"] is not None:
            total = edge["book_total"]
            ov = edge["total_edge_over"] or 0
            un = edge["total_edge_under"] or 0
            print(f"  O/U:     Model {sim['avg_total_runs']:.1f} | Line {total} → "
                  f"Over {ov*100:+.1f}% / Under {un*100:+.1f}%")
    else:
        print(f"  Runs:    {sim['avg_away_runs']:.1f} – {sim['avg_home_runs']:.1f}  (total: {sim['avg_total_runs']:.1f})")

    if wind.get("description") and not wind.get("dome"):
        wdesc = wind["description"]
        print(f"  Wind:    {wdesc}")

    print(f"  --> {rec}")


if __name__ == "__main__":
    main()
