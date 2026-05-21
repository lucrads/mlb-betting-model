"""
Per at-bat probability engine.

Takes a batter profile and pitcher profile and returns a probability
distribution over at-bat outcomes: HR, BB, K, 1B, 2B, 3B, OUT.

Steps:
  1. Start with batter's base outcome rates (season stats)
  2. Adjust for L/R split vs pitcher handedness
  3. Adjust for pitcher's pitch mix vs batter's pitch-type effectiveness
  4. Normalize to sum to 1.0
"""

import numpy as np
from config import LEAGUE_AVG_WOBA_BY_PITCH

OUTCOMES = ["HR", "BB", "K", "1B", "2B", "3B", "OUT"]
POSITIVE_OUTCOMES = {"HR", "BB", "1B", "2B", "3B"}
NEGATIVE_OUTCOMES = {"K", "OUT"}


def compute_at_bat_probs(batter: dict, pitcher: dict) -> dict[str, float]:
    """
    Returns a dict {outcome: probability} for a single plate appearance.
    """
    rates = dict(batter["outcome_rates"])

    # Step 2: L/R split adjustment
    rates = _apply_split_adjustment(rates, batter, pitcher)

    # Step 3: Pitch mix adjustment
    rates = _apply_pitch_mix_adjustment(rates, batter, pitcher)

    # Step 4: Normalize
    rates = _normalize(rates)

    return rates


def _apply_split_adjustment(rates: dict, batter: dict, pitcher: dict) -> dict:
    pitcher_hand = pitcher.get("hand", "R")
    batter_woba = batter.get("woba", 0.320) or 0.320
    woba_vs_hand = batter.get("woba_vs_hand", {})
    woba_vs_this_hand = woba_vs_hand.get(pitcher_hand, batter_woba)

    if batter_woba <= 0:
        return rates

    split_multiplier = woba_vs_this_hand / batter_woba

    # Clamp to reasonable range to avoid extreme adjustments
    split_multiplier = float(np.clip(split_multiplier, 0.60, 1.60))

    adjusted = {}
    for outcome, prob in rates.items():
        if outcome in POSITIVE_OUTCOMES:
            adjusted[outcome] = prob * split_multiplier
        elif outcome in NEGATIVE_OUTCOMES:
            # Worse splits → more negative outcomes
            adjusted[outcome] = prob * (2.0 - split_multiplier)
        else:
            adjusted[outcome] = prob

    return adjusted


def _apply_pitch_mix_adjustment(rates: dict, batter: dict, pitcher: dict) -> dict:
    pitch_mix = pitcher.get("pitch_mix", {})
    batter_woba_vs_pitch = batter.get("woba_vs_pitch", {})

    if not pitch_mix:
        return rates

    matchup_factor = 0.0
    total_weight = 0.0

    for pitch_type, pct in pitch_mix.items():
        if pct <= 0:
            continue
        league_avg = LEAGUE_AVG_WOBA_BY_PITCH.get(pitch_type, LEAGUE_AVG_WOBA_BY_PITCH["OTHER"])
        batter_val = batter_woba_vs_pitch.get(pitch_type, league_avg)
        if league_avg > 0:
            matchup_factor += pct * (batter_val / league_avg)
            total_weight += pct

    if total_weight <= 0:
        return rates

    matchup_factor /= total_weight
    matchup_factor = float(np.clip(matchup_factor, 0.65, 1.50))

    adjusted = {}
    for outcome, prob in rates.items():
        if outcome in POSITIVE_OUTCOMES:
            adjusted[outcome] = prob * matchup_factor
        elif outcome in NEGATIVE_OUTCOMES:
            adjusted[outcome] = prob * (2.0 - matchup_factor)
        else:
            adjusted[outcome] = prob

    return adjusted


def _normalize(rates: dict) -> dict[str, float]:
    total = sum(rates.values())
    if total <= 0:
        # Fallback to equal distribution
        n = len(rates)
        return {k: 1.0 / n for k in rates}
    return {k: max(v / total, 0.0) for k, v in rates.items()}


def sample_outcome(probs: dict[str, float]) -> str:
    """Sample a single at-bat outcome from the probability distribution."""
    outcomes = list(probs.keys())
    weights = [probs[o] for o in outcomes]
    return np.random.choice(outcomes, p=weights)
