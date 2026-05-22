"""
Display Agent — assembles all stored data into the HTML report.

Reads from store:
  store/{date}/schedule.json        (game list)
  store/{date}/player_profiles.json (pitcher/lineup/weather info)
  store/{date}/simulations.json     (Monte Carlo results)
  store/{date}/odds.json            (pre-game sportsbook lines)
  store/backtest/game_records.json  (historical accuracy log)

Produces: report_{date}.html

Run:
  python3 agents/display_agent.py --date 2026-05-22
  python3 agents/display_agent.py --date 2026-05-22 --output-dir /tmp
"""

import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import store
from output.edge_calc import compute_edge
from output.matchup_details import compute_matchup_details
from output.report import generate_report_from_store
from output.backtest_analyzer import analyze_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [display]  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="MLB Display Agent")
    p.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: today)")
    p.add_argument("--output-dir", default=".", help="Directory for HTML report")
    return p.parse_args()


def run(date: str, output_dir: str = ".") -> str:
    """
    Assemble stored data and generate the HTML report.
    Returns path to the written report file.
    """
    from datetime import date as date_cls
    if not date:
        date = date_cls.today().isoformat()

    logger.info("=== Display Agent | %s ===", date)

    # Load all stored data
    schedule = store.read_schedule(date)
    profiles = store.read_player_profiles(date)
    sims = store.read_simulations(date)
    stored_odds = store.read_odds(date)
    bt_records = store.read_backtest_records()

    if not schedule:
        logger.error("No schedule found for %s — run data_agent first.", date)
        return ""

    if not sims:
        logger.error("No simulations found for %s — run engine_agent first.", date)
        return ""

    logger.info("Assembling %d games...", len(schedule))

    # Build game_results list in the same format report.py expects
    game_results = []
    for game in schedule:
        gid = str(game["game_id"])
        sim = sims.get(gid)
        if not sim:
            logger.warning("  No simulation for game %s (%s @ %s) — skipping",
                           gid, game.get("away_team"), game.get("home_team"))
            continue

        p = profiles.get(gid, {})

        # Merge profile data back into the game dict for report rendering
        enriched_game = dict(game)
        enriched_game["home_pitcher_profile"] = p.get("home_pitcher_profile", {})
        enriched_game["away_pitcher_profile"] = p.get("away_pitcher_profile", {})
        enriched_game["home_bullpen_profile"] = p.get("home_bullpen_profile", {})
        enriched_game["away_bullpen_profile"] = p.get("away_bullpen_profile", {})
        enriched_game["home_lineup"] = p.get("home_lineup", game.get("home_lineup", []))
        enriched_game["away_lineup"] = p.get("away_lineup", game.get("away_lineup", []))
        enriched_game["home_lineup_profiles"] = p.get("home_lineup_profiles", [])
        enriched_game["away_lineup_profiles"] = p.get("away_lineup_profiles", [])
        enriched_game["outward_wind_mph"] = p.get("outward_wind_mph", 0.0)
        enriched_game["wind"] = p.get("wind", {"description": "", "dome": False})

        # Resolve odds for this game (from immutable stored pre-game lines)
        game_odds = stored_odds.get(gid)

        # Compute edge vs stored odds
        edge = compute_edge(sim, game_odds)

        # Per-batter matchup details for the lineup detail view
        details = compute_matchup_details(enriched_game)

        game_results.append({
            "game": enriched_game,
            "simulation": sim,
            "edge": edge,
            "details": details,
        })

        logger.info(
            "  %s @ %s  →  Home %.1f%%  Away %.1f%%  | %s",
            game.get("away_team"), game.get("home_team"),
            sim["home_win_pct"] * 100,
            sim["away_win_pct"] * 100,
            edge["recommendation"],
        )

    # Analyze backtest records
    bt_stats = analyze_backtest(bt_records) if bt_records else None
    if bt_stats:
        logger.info("Backtest: %d games, ML accuracy %.1f%%",
                    bt_stats["total_games"], bt_stats["ml_rate"] * 100)

    # Generate report
    report_path = generate_report_from_store(
        date_str=date,
        game_results=game_results,
        bt_records=bt_records,
        bt_stats=bt_stats,
        output_dir=output_dir,
    )
    logger.info("Report written: %s", report_path)
    return report_path


if __name__ == "__main__":
    args = parse_args()
    run(date=args.date, output_dir=args.output_dir)
