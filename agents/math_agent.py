"""
Math Agent — pre-computes per at-bat probability distributions.

For every game on the given date, computes outcome distributions for:
  - Each batter in both lineups vs the opposing starter
  - Each batter in both lineups vs the opposing bullpen profile

Saves results to store/{date}/ab_probs.json so the engine agent can run
simulations without calling compute_at_bat_probs() inside the hot loop.

Run:
  python3 agents/math_agent.py --date 2026-05-22

Schema of ab_probs.json:
  {
    "<game_id>": {
      "home": {
        "vs_starter": [  # one dict per batter (lineup order)
          {"HR": 0.03, "BB": 0.09, "K": 0.21, "1B": 0.15, "2B": 0.05, "3B": 0.005, "OUT": 0.455},
          ...
        ],
        "vs_bullpen": [ ... ]
      },
      "away": {
        "vs_starter": [ ... ],
        "vs_bullpen": [ ... ]
      }
    },
    ...
  }
"""

import sys
import os
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.matchup import compute_at_bat_probs
import store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [math]  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="MLB Math Agent")
    p.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: today)")
    return p.parse_args()


def run(date: str) -> dict:
    """
    Pre-compute per at-bat probability distributions for all games on date.
    Returns the ab_probs dict (also written to store).
    """
    from datetime import date as date_cls
    if not date:
        date = date_cls.today().isoformat()

    logger.info("=== Math Agent | %s ===", date)

    profiles = store.read_player_profiles(date)
    if not profiles:
        logger.error("No player profiles found for %s — run data_agent first.", date)
        return {}

    ab_probs = {}

    for gid, p in profiles.items():
        logger.info("  Computing AB probs for game %s...", gid)

        home_starter = p["home_pitcher_profile"]
        away_starter = p["away_pitcher_profile"]
        home_bullpen = p["home_bullpen_profile"]
        away_bullpen = p["away_bullpen_profile"]
        home_lineup = p["home_lineup_profiles"]
        away_lineup = p["away_lineup_profiles"]
        wind_mph = p.get("outward_wind_mph", 0.0)
        park_hr_factor = p.get("park_hr_factor", 1.0)

        # Away batters face home pitcher (starter then bullpen); both play in home park
        away_vs_starter = [
            compute_at_bat_probs(b, home_starter, outward_wind_mph=wind_mph, park_hr_factor=park_hr_factor)
            for b in away_lineup
        ]
        away_vs_bullpen = [
            compute_at_bat_probs(b, home_bullpen, outward_wind_mph=wind_mph, park_hr_factor=park_hr_factor)
            for b in away_lineup
        ]

        # Home batters face away pitcher (starter then bullpen); all in home park
        home_vs_starter = [
            compute_at_bat_probs(b, away_starter, outward_wind_mph=wind_mph, park_hr_factor=park_hr_factor)
            for b in home_lineup
        ]
        home_vs_bullpen = [
            compute_at_bat_probs(b, away_bullpen, outward_wind_mph=wind_mph, park_hr_factor=park_hr_factor)
            for b in home_lineup
        ]

        ab_probs[gid] = {
            "home": {
                "vs_starter": home_vs_starter,
                "vs_bullpen": home_vs_bullpen,
            },
            "away": {
                "vs_starter": away_vs_starter,
                "vs_bullpen": away_vs_bullpen,
            },
        }

        logger.info(
            "    %d home batters × 2, %d away batters × 2 computed.",
            len(home_lineup), len(away_lineup),
        )

    store.write_ab_probs(date, ab_probs)
    logger.info("AB probability distributions stored for %d games.", len(ab_probs))

    return ab_probs


if __name__ == "__main__":
    args = parse_args()
    run(date=args.date)
