"""
Single-game simulation.

Threads an `rng` object — either a numpy Generator-backed antithetic
facade from monte_carlo.py or, for stand-alone use, falls back to the
global numpy random state. All stochastic choices in this module flow
through `rng.choice(items, p)` and `rng.random()`, which makes the
simulator deterministic given the rng state and enables antithetic
variates / common-random-numbers variance reduction at the aggregator.
"""

import numpy as np
from model.matchup import compute_at_bat_probs
from config import BASE_RUNNING, STARTER_INNINGS_LIMIT, STARTER_BATTERS_FACED_LIMIT, FIP_WOBA_INTERCEPT, FIP_WOBA_SLOPE  # noqa: F401


class _GlobalRng:
    """Fallback RNG facade that calls numpy's global random state."""

    @staticmethod
    def random() -> float:
        return float(np.random.random())

    @staticmethod
    def choice(items, p):
        return np.random.choice(items, p=p)


_GLOBAL_RNG = _GlobalRng()


def simulate_game(game: dict, rng=None) -> tuple[int, int]:
    """
    Simulate one full game and return (home_runs, away_runs).

    `rng`: optional object exposing `random()` and `choice(items, p)`.
           When None, falls back to numpy's global random state.
    """
    if rng is None:
        rng = _GLOBAL_RNG

    away_lineup = game["away_lineup_profiles"]
    home_lineup = game["home_lineup_profiles"]
    away_starter = game["away_pitcher_profile"]
    home_starter = game["home_pitcher_profile"]
    away_bullpen = game["away_bullpen_profile"]
    home_bullpen = game["home_bullpen_profile"]

    home_runs = 0
    away_runs = 0
    outward_wind_mph = game.get("outward_wind_mph", 0.0)
    park_hr_factor   = game.get("park_hr_factor", 1.0)

    away_batter_idx = 0
    home_batter_idx = 0
    away_tbf = 0  # total batters faced by away starter (for TBF limit)
    home_tbf = 0

    inning = 1
    max_innings = 15  # safety cap for extra innings

    while inning <= max_innings:
        away_pitcher = _get_active_pitcher(away_starter, away_bullpen, inning, away_tbf)
        runs, away_batter_idx, tbf = _simulate_half_inning(
            away_lineup, away_batter_idx, away_pitcher,
            rng=rng,
            outward_wind_mph=outward_wind_mph,
            park_hr_factor=park_hr_factor,
        )
        away_runs += runs
        if away_pitcher is away_starter:
            away_tbf += tbf

        home_pitcher = _get_active_pitcher(home_starter, home_bullpen, inning, home_tbf)
        runs, home_batter_idx, tbf = _simulate_half_inning(
            home_lineup, home_batter_idx, home_pitcher,
            rng=rng,
            walkoff=(inning >= 9),
            runs_needed=(away_runs - home_runs) if inning >= 9 else 999,
            outward_wind_mph=outward_wind_mph,
            park_hr_factor=park_hr_factor,
        )
        home_runs += runs
        if home_pitcher is home_starter:
            home_tbf += tbf

        if inning >= 9 and home_runs != away_runs:
            break

        inning += 1

    return home_runs, away_runs


def _get_active_pitcher(starter: dict, bullpen: dict, inning: int, batters_faced: int) -> dict:
    if inning > STARTER_INNINGS_LIMIT or batters_faced >= STARTER_BATTERS_FACED_LIMIT:
        return bullpen
    return starter


def _simulate_half_inning(
    lineup: list[dict],
    batter_idx: int,
    pitcher: dict,
    rng,
    walkoff: bool = False,
    runs_needed: int = 999,
    outward_wind_mph: float = 0.0,
    park_hr_factor: float = 1.0,
) -> tuple[int, int, int]:
    """Simulate one half-inning. Returns (runs_scored, next_batter_idx, batters_faced)."""
    outs = 0
    runs = 0
    batters_faced = 0
    bases = [False, False, False]  # 1B, 2B, 3B

    lineup_size = len(lineup)
    if lineup_size == 0:
        return 0, batter_idx, 0

    current_idx = batter_idx
    while outs < 3:
        batter = lineup[current_idx % lineup_size]
        probs = compute_at_bat_probs(
            batter, pitcher,
            outward_wind_mph=outward_wind_mph,
            park_hr_factor=park_hr_factor,
        )
        outcome = _sample_outcome(probs, rng)
        runs_scored, bases, outs = _apply_outcome(outcome, bases, outs, rng)
        runs += runs_scored
        current_idx += 1
        batters_faced += 1
        if walkoff and runs > runs_needed:
            break

    return runs, current_idx % lineup_size, batters_faced


def _sample_outcome(probs: dict[str, float], rng) -> str:
    outcomes = list(probs.keys())
    weights = [probs[o] for o in outcomes]
    return rng.choice(outcomes, p=weights)


def _apply_outcome(
    outcome: str,
    bases: list[bool],
    outs: int,
    rng,
) -> tuple[int, list[bool], int]:
    """Apply a batting outcome to the base state. Returns (runs_scored, new_bases, new_outs)."""
    b1, b2, b3 = bases
    runs = 0
    new_outs = outs
    pr = BASE_RUNNING

    if outcome == "HR":
        runs = 1 + sum([b1, b2, b3])
        return runs, [False, False, False], new_outs

    if outcome == "3B":
        runs = sum([b1, b2, b3])
        return runs, [False, False, True], new_outs

    if outcome == "2B":
        runs += int(b2) + int(b3)
        runner_1b_scored = False
        if b1 and rng.random() < pr["double_runner_1b_scores_prob"]:
            runs += 1
            runner_1b_scored = True
        new_b3 = b1 and not runner_1b_scored
        return runs, [False, True, new_b3], new_outs

    if outcome == "1B":
        if b3:
            runs += 1
        new_b3 = False
        if b2:
            if rng.random() < pr["single_runner_2b_scores_prob"]:
                runs += 1
            else:
                new_b3 = True  # runner from 2B stops at 3B
        new_b2 = False
        if b1:
            if new_b3:
                # 3B already occupied — runner from 1B can only reach 2B
                new_b2 = True
            elif rng.random() < pr["single_runner_1b_to_3b_prob"]:
                new_b3 = True   # runner from 1B takes extra base to 3B
            else:
                new_b2 = True   # runner from 1B stops at 2B
        return runs, [True, new_b2, new_b3], new_outs

    if outcome == "BB":
        if b1 and b2 and b3:
            runs += 1
            return runs, [True, True, True], new_outs
        if b1 and b2:
            # Runner on 2B forced to 3B, runner on 1B forced to 2B
            return runs, [True, True, True], new_outs
        if b1:
            # Runner on 1B forced to 2B
            return runs, [True, True, b3], new_outs
        return runs, [True, b2, b3], new_outs

    if outcome == "K":
        new_outs += 1
        return 0, bases[:], new_outs

    # OUT — groundout/flyout
    new_outs += 1
    if new_outs < 3 and b3 and rng.random() < pr["groundout_runner_3b_scores_prob"]:
        runs += 1
        return runs, [b1, b2, False], new_outs
    return 0, bases[:], new_outs


def build_bullpen_profile(bullpen_stats: dict) -> dict:
    """Convert team bullpen ERA/FIP into a pitcher profile for simulation."""
    era = bullpen_stats.get("era", 4.50)
    fip = bullpen_stats.get("fip", 4.20)
    woba_allowed = max(0.180, min(0.420, FIP_WOBA_INTERCEPT + fip * FIP_WOBA_SLOPE))

    pitch_types = ["FF", "SI", "SL", "FC", "ST", "CH", "CU", "FS"]
    return {
        "name": "Bullpen",
        "era": era,
        "fip": fip,
        "hand": "R",
        "pitch_mix": {"FF": 0.45, "SL": 0.28, "FC": 0.10, "CH": 0.10, "CU": 0.07},
        "pitch_woba_allowed": {pt: round(woba_allowed, 3) for pt in pitch_types},
        "woba_allowed_overall": round(woba_allowed, 3),
    }
