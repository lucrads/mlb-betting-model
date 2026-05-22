"""
Single game simulation.

Models a full 9-inning game (plus extra innings if tied) by:
  - Cycling through lineups batter by batter
  - Computing per at-bat probabilities via the matchup engine
  - Applying base running rules after each event
  - Switching from starter to bullpen profile after starter limit
"""

import numpy as np
from model.matchup import compute_at_bat_probs, sample_outcome, _pitcher_avg_woba_allowed
from config import BASE_RUNNING, STARTER_INNINGS_LIMIT, STARTER_BATTERS_FACED_LIMIT, FIP_WOBA_INTERCEPT, FIP_WOBA_SLOPE


def simulate_game(game: dict, batter_profiles: dict, pitcher_profiles: dict) -> tuple[int, int]:
    """
    Simulate one full game.

    Args:
        game: game dict from fetcher (teams, lineups, pitchers)
        batter_profiles: {player_name: profile_dict}
        pitcher_profiles: {player_name: profile_dict}

    Returns:
        (home_runs, away_runs)
    """
    away_lineup = game["away_lineup_profiles"]
    home_lineup = game["home_lineup_profiles"]
    away_starter = game["away_pitcher_profile"]
    home_starter = game["home_pitcher_profile"]
    away_bullpen = game["away_bullpen_profile"]
    home_bullpen = game["home_bullpen_profile"]

    home_runs = 0
    away_runs = 0
    outward_wind_mph = game.get("outward_wind_mph", 0.0)

    # Lineup position state persists across innings
    away_batter_idx = 0
    home_batter_idx = 0

    inning = 1
    max_innings = 15  # safety cap for extra innings

    while inning <= max_innings:
        # Top of inning: away team bats vs home pitcher
        away_pitcher = _get_active_pitcher(away_starter, away_bullpen, inning)
        runs, away_batter_idx = _simulate_half_inning(
            away_lineup, away_batter_idx, away_pitcher, outward_wind_mph=outward_wind_mph,
        )
        away_runs += runs

        # Bottom of inning: home team bats vs away pitcher
        home_pitcher = _get_active_pitcher(home_starter, home_bullpen, inning)
        runs, home_batter_idx = _simulate_half_inning(
            home_lineup, home_batter_idx, home_pitcher,
            walkoff=(inning >= 9),
            runs_needed=(away_runs - home_runs) if inning >= 9 else 999,
            outward_wind_mph=outward_wind_mph,
        )
        home_runs += runs

        # Check for game end after 9+ complete innings
        if inning >= 9 and home_runs != away_runs:
            break

        inning += 1

    return home_runs, away_runs


def _get_active_pitcher(starter: dict, bullpen: dict, inning: int) -> dict:
    if inning > STARTER_INNINGS_LIMIT:
        return bullpen
    return starter


def _simulate_half_inning(
    lineup: list[dict],
    batter_idx: int,
    pitcher: dict,
    walkoff: bool = False,
    runs_needed: int = 999,
    outward_wind_mph: float = 0.0,
) -> tuple[int, int]:
    """
    Simulate one half-inning. Returns (runs_scored, next_batter_idx).

    runs_needed: in walkoff situations, stop once cumulative runs exceed this
                 (i.e. home team takes the lead).
    """
    outs = 0
    runs = 0
    bases = [False, False, False]  # 1B, 2B, 3B

    lineup_size = len(lineup)
    if lineup_size == 0:
        return 0, batter_idx

    current_idx = batter_idx

    while outs < 3:
        batter = lineup[current_idx % lineup_size]
        probs = compute_at_bat_probs(batter, pitcher, outward_wind_mph=outward_wind_mph)
        outcome = sample_outcome(probs)

        runs_scored, bases, outs = _apply_outcome(outcome, bases, outs)
        runs += runs_scored

        current_idx += 1

        if walkoff and runs > runs_needed:
            break

    return runs, current_idx % lineup_size


def _apply_outcome(
    outcome: str,
    bases: list[bool],
    outs: int,
) -> tuple[int, list[bool], int]:
    """
    Apply a batting outcome to the base state.

    Returns: (runs_scored, new_bases, new_outs)
    Bases: [1B, 2B, 3B] as booleans
    """
    b1, b2, b3 = bases
    runs = 0
    new_outs = outs
    rng = BASE_RUNNING

    if outcome == "HR":
        runs = 1 + sum([b1, b2, b3])
        return runs, [False, False, False], new_outs

    elif outcome == "3B":
        runs = sum([b1, b2, b3])
        return runs, [False, False, True], new_outs

    elif outcome == "2B":
        # Runners on 2nd/3rd score; runner on 1st scores ~30% or stops at 3rd
        runs += int(b2) + int(b3)
        runner_1b_scored = False
        if b1:
            if np.random.random() < rng["double_runner_1b_scores_prob"]:
                runs += 1
                runner_1b_scored = True
        # Batter on 2nd; runner from 1st on 3rd if didn't score
        new_b3 = b1 and not runner_1b_scored
        return runs, [False, True, new_b3], new_outs

    elif outcome == "1B":
        # Runner on 3rd scores
        if b3:
            runs += 1
        # Runner on 2nd: scores ~75%, else stops at 3rd
        new_b3 = False
        if b2:
            if np.random.random() < rng["single_runner_2b_scores_prob"]:
                runs += 1
            else:
                new_b3 = True
        # Runner on 1st: takes extra base ~30%
        new_b2 = bool(b1)
        new_b1_extra = False
        if b1:
            if np.random.random() < rng["single_runner_1b_to_3b_prob"]:
                new_b3 = True
                new_b2 = False
        # Batter is on 1st
        return runs, [True, new_b2, new_b3], new_outs

    elif outcome == "BB":
        # Force advances only
        if b1 and b2 and b3:
            runs += 1
            return runs, [True, True, True], new_outs
        elif b1 and b2:
            return runs, [True, True, b3], new_outs
        elif b1:
            return runs, [True, True, b3], new_outs
        else:
            return runs, [True, b2, b3], new_outs

    elif outcome == "K":
        new_outs += 1
        return 0, bases[:], new_outs

    else:  # OUT (groundout / flyout)
        new_outs += 1
        if new_outs < 3:
            # Runner on 3rd tags up on flyout / scores on groundout
            if b3:
                if np.random.random() < rng["groundout_runner_3b_scores_prob"]:
                    runs += 1
                    return runs, [b1, b2, False], new_outs
        return 0, bases[:], new_outs


def build_bullpen_profile(bullpen_stats: dict) -> dict:
    """Convert team bullpen ERA/FIP into a pitcher profile for simulation.

    woba_allowed is derived from this team's own FIP — no league avg substitution.
    Pitch mix is a reasonable bullpen aggregate (heavy fastball/slider usage);
    since woba_allowed is uniform across types, the mix only weights the batter's
    pitch-type splits and does not introduce league avg into the denominator.
    """
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
