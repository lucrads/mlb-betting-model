"""
Single game simulation.

Models a full 9-inning game (plus extra innings if tied) by:
  - Cycling through lineups batter by batter
  - Computing per at-bat probabilities via the matchup engine
  - Applying base running rules after each event
  - Switching from starter to bullpen profile after starter limit
"""

import numpy as np
from model.matchup import compute_at_bat_probs, sample_outcome
from config import BASE_RUNNING, STARTER_INNINGS_LIMIT, STARTER_BATTERS_FACED_LIMIT


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

    # Lineup position state persists across innings
    away_batter_idx = 0
    home_batter_idx = 0

    inning = 1
    max_innings = 15  # safety cap for extra innings

    while inning <= max_innings:
        # Top of inning: away team bats vs home pitcher
        away_pitcher = _get_active_pitcher(away_starter, away_bullpen, inning)
        runs, away_batter_idx = _simulate_half_inning(away_lineup, away_batter_idx, away_pitcher)
        away_runs += runs

        # Bottom of inning: home team bats vs away pitcher
        # Walk-off: home wins in bottom of 9th+
        home_pitcher = _get_active_pitcher(home_starter, home_bullpen, inning)
        runs, home_batter_idx = _simulate_half_inning(
            home_lineup, home_batter_idx, home_pitcher,
            walkoff=(inning >= 9 and home_runs + 1 > away_runs or inning >= 9)
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
) -> tuple[int, int]:
    """
    Simulate one half-inning. Returns (runs_scored, next_batter_idx).

    lineup: list of batter profile dicts
    batter_idx: index of the leadoff batter for this half-inning
    walkoff: if True, end immediately when home team takes the lead
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
        probs = compute_at_bat_probs(batter, pitcher)
        outcome = sample_outcome(probs)

        runs_scored, bases, new_outs = _apply_outcome(outcome, bases, outs)
        runs += runs_scored
        outs = new_outs

        current_idx += 1

        if walkoff and runs > 0:
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
        # All runners on 2nd and 3rd score; runner on 1st reaches 3rd (~70%) or scores (~30%)
        runs += int(b2) + int(b3)
        if b1:
            if np.random.random() < rng["double_runner_1b_scores_prob"]:
                runs += 1
                new_b1 = False
            else:
                new_b1 = False  # runner goes to 3rd
                return runs, [False, False, True], new_outs
        new_bases = [False, False, False]
        new_bases[1] = True  # batter on 2nd
        if b1 and runs == int(b2) + int(b3):  # runner on 1st didn't score
            new_bases[2] = True
        return runs, new_bases, new_outs

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
    """Convert bullpen ERA/FIP to a generic pitcher profile for simulation."""
    era = bullpen_stats.get("era", 4.20)
    fip = bullpen_stats.get("fip", 4.10)
    # Estimate wOBA from FIP: rough linear mapping
    # League avg FIP ~4.10 → wOBA allowed ~0.320
    woba_allowed = 0.320 + (fip - 4.10) * 0.020

    return {
        "name": "Bullpen",
        "era": era,
        "fip": fip,
        "hand": "R",
        "pitch_mix": {"FF": 0.50, "SL": 0.25, "CH": 0.15, "CU": 0.10},
        "pitch_woba_allowed": {pt: woba_allowed for pt in ["FF", "SI", "SL", "CH", "CU", "FC", "ST", "FS"]},
    }
