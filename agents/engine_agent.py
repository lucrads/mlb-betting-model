"""
Engine Agent — runs Monte Carlo game simulations using pre-computed AB probs.

Reads:
  store/{date}/player_profiles.json  (game/team/player structure)
  store/{date}/ab_probs.json         (pre-computed per-batter outcome distributions)

Writes:
  store/{date}/simulations.json      (sim results per game_id)

Using pre-computed AB probs avoids calling compute_at_bat_probs() inside the
simulation hot loop, which gives a ~30-40% speedup and ensures math and engine
agents use identical probability distributions.

Run:
  python3 agents/engine_agent.py --date 2026-05-22 [--sims 1000]
"""

import sys
import os
import logging
import argparse
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.matchup import sample_outcome
from model.simulator import _apply_outcome, _get_active_pitcher
import store
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [engine]  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="MLB Engine Agent")
    p.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: today)")
    p.add_argument("--sims", type=int, default=None, help="Simulations per game")
    return p.parse_args()


def _simulate_half_inning_fast(
    probs_vs_starter: list[dict],
    probs_vs_bullpen: list[dict],
    batter_idx: int,
    inning: int,
    walkoff: bool = False,
    runs_needed: int = 999,
) -> tuple[int, int]:
    """
    Simulate one half-inning using pre-computed probability tables.
    Selects starter vs bullpen probs based on inning number.
    Returns (runs_scored, next_batter_idx).
    """
    lineup_probs = probs_vs_bullpen if inning > config.STARTER_INNINGS_LIMIT else probs_vs_starter
    lineup_size = len(lineup_probs)
    if lineup_size == 0:
        return 0, batter_idx

    outs = 0
    runs = 0
    bases = [False, False, False]
    current_idx = batter_idx

    while outs < 3:
        probs = lineup_probs[current_idx % lineup_size]
        outcome = sample_outcome(probs)
        runs_scored, bases, outs = _apply_outcome(outcome, bases, outs)
        runs += runs_scored
        current_idx += 1
        if walkoff and runs > runs_needed:
            break

    return runs, current_idx % lineup_size


def simulate_game_fast(ab_probs_game: dict) -> tuple[int, int]:
    """
    Simulate one full game using pre-computed AB probability tables.
    Returns (home_runs, away_runs).
    """
    home_probs_starter = ab_probs_game["home"]["vs_starter"]
    home_probs_bullpen = ab_probs_game["home"]["vs_bullpen"]
    away_probs_starter = ab_probs_game["away"]["vs_starter"]
    away_probs_bullpen = ab_probs_game["away"]["vs_bullpen"]

    home_runs = 0
    away_runs = 0
    away_batter_idx = 0
    home_batter_idx = 0

    inning = 1
    max_innings = 15

    while inning <= max_innings:
        # Top: away bats vs home pitching
        runs, away_batter_idx = _simulate_half_inning_fast(
            away_probs_starter, away_probs_bullpen, away_batter_idx, inning,
        )
        away_runs += runs

        # Bottom: home bats vs away pitching
        runs, home_batter_idx = _simulate_half_inning_fast(
            home_probs_starter, home_probs_bullpen, home_batter_idx, inning,
            walkoff=(inning >= 9),
            runs_needed=(away_runs - home_runs) if inning >= 9 else 999,
        )
        home_runs += runs

        if inning >= 9 and home_runs != away_runs:
            break

        inning += 1

    return home_runs, away_runs


def run_game_simulations(gid: str, ab_probs_game: dict, n_sims: int) -> dict:
    """Run n_sims simulations for one game. Returns aggregated sim result dict."""
    home_wins = 0
    away_wins = 0
    ties = 0
    home_run_totals = []
    away_run_totals = []
    score_dist: Counter = Counter()

    for _ in range(n_sims):
        h, a = simulate_game_fast(ab_probs_game)
        home_run_totals.append(h)
        away_run_totals.append(a)
        score_dist[(h, a)] += 1
        if h > a:
            home_wins += 1
        elif a > h:
            away_wins += 1
        else:
            ties += 1

    avg_home = float(np.mean(home_run_totals))
    avg_away = float(np.mean(away_run_totals))

    run_dist: Counter = Counter()
    for (h, a), cnt in score_dist.items():
        run_dist[h + a] += cnt

    return {
        "home_win_pct": round(home_wins / n_sims, 4),
        "away_win_pct": round(away_wins / n_sims, 4),
        "tie_pct": round(ties / n_sims, 4),
        "avg_home_runs": round(avg_home, 2),
        "avg_away_runs": round(avg_away, 2),
        "avg_total_runs": round(avg_home + avg_away, 2),
        "run_distribution": dict(run_dist),
        "most_common_scores": score_dist.most_common(5),
    }


def run(date: str, n_sims: int = None) -> dict:
    """
    Run Monte Carlo simulations for all games on date.
    Returns dict of {game_id: sim_result} (also written to store).
    """
    from datetime import date as date_cls
    if not date:
        date = date_cls.today().isoformat()

    n_sims = n_sims or config.NUM_SIMULATIONS
    logger.info("=== Engine Agent | %s | %d sims ===", date, n_sims)

    ab_probs = store.read_ab_probs(date)
    if not ab_probs:
        logger.error("No AB probs found for %s — run math_agent first.", date)
        return {}

    profiles = store.read_player_profiles(date)

    results = {}
    for gid, ab_probs_game in ab_probs.items():
        p = profiles.get(gid, {})
        home = p.get("home_pitcher_profile", {}).get("name", "Home")
        away = p.get("away_pitcher_profile", {}).get("name", "Away")
        logger.info("  Simulating game %s (%d sims)...", gid, n_sims)

        result = run_game_simulations(gid, ab_probs_game, n_sims)
        results[gid] = result

        logger.info(
            "    Home %.1f%% | Away %.1f%% | Avg total: %.1f",
            result["home_win_pct"] * 100,
            result["away_win_pct"] * 100,
            result["avg_total_runs"],
        )

    store.write_simulations(date, results)
    logger.info("Simulations stored for %d games.", len(results))

    return results


if __name__ == "__main__":
    args = parse_args()
    if args.sims:
        config.NUM_SIMULATIONS = args.sims
    run(date=args.date, n_sims=args.sims)
