"""
Per at-bat probability engine.

Takes a batter profile and pitcher profile and returns a probability
distribution over at-bat outcomes: HR, BB, K, 1B, 2B, 3B, OUT.

Steps:
  1. Start with batter's base outcome rates (season/career stats)
  2. Adjust for L/R split vs pitcher handedness (batter's wOBA vs this hand / overall wOBA)
  3. Adjust for pitcher's pitch mix vs batter's pitch-type effectiveness
     — denominator is what THIS pitcher actually allows per pitch type, not a league avg
  4. Normalize to sum to 1.0
"""

import numpy as np
from config import FIP_WOBA_INTERCEPT, FIP_WOBA_SLOPE

OUTCOMES = ["HR", "BB", "K", "1B", "2B", "3B", "OUT"]
POSITIVE_OUTCOMES = {"HR", "BB", "1B", "2B", "3B"}
NEGATIVE_OUTCOMES = {"K", "OUT"}

# Neutral outcome rates used only when a batter has zero data
_NEUTRAL_RATES = {
    "HR": 0.030, "BB": 0.085, "K": 0.225,
    "1B": 0.145, "2B": 0.050, "3B": 0.005, "OUT": 0.460,
}


def compute_at_bat_probs(batter: dict, pitcher: dict) -> dict[str, float]:
    """
    Returns a dict {outcome: probability} for a single plate appearance.
    All adjustments use player-specific data only; empty data fields produce
    no adjustment (multiplier stays at 1.0).
    """
    rates = dict(batter["outcome_rates"]) if batter.get("outcome_rates") else dict(_NEUTRAL_RATES)

    rates = _apply_split_adjustment(rates, batter, pitcher)
    rates = _apply_pitch_mix_adjustment(rates, batter, pitcher)
    rates = _normalize(rates)

    return rates


def _apply_split_adjustment(rates: dict, batter: dict, pitcher: dict) -> dict:
    """
    Scale outcomes by batter's wOBA vs this pitcher's hand relative to
    batter's overall wOBA. Skips adjustment if batter has no wOBA data.
    """
    batter_woba = batter.get("woba") or 0.0
    if batter_woba <= 0:
        return rates  # No batter data — no adjustment

    pitcher_hand = pitcher.get("hand", "R")
    woba_vs_hand = batter.get("woba_vs_hand", {})
    # Fall back to overall wOBA if no split data for this hand (ratio = 1.0)
    woba_vs_this_hand = woba_vs_hand.get(pitcher_hand, batter_woba)

    split_multiplier = woba_vs_this_hand / batter_woba
    split_multiplier = float(np.clip(split_multiplier, 0.60, 1.60))

    adjusted = {}
    for outcome, prob in rates.items():
        if outcome in POSITIVE_OUTCOMES:
            adjusted[outcome] = prob * split_multiplier
        elif outcome in NEGATIVE_OUTCOMES:
            adjusted[outcome] = prob * (2.0 - split_multiplier)
        else:
            adjusted[outcome] = prob

    return adjusted


def _apply_pitch_mix_adjustment(rates: dict, batter: dict, pitcher: dict) -> dict:
    """
    Scale outcomes using the geometric mean of batter and pitcher Statcast values
    per pitch type, normalized by the batter's overall wOBA.

    matchup_factor = Σ(pitch_pct × sqrt(batter_woba_vs_PT × pitcher_woba_allowed_PT))
                     ────────────────────────────────────────────────────────────────
                                      batter_overall_woba

    Semantics:
      - factor = 1.0 when the geometric mean of the matchup equals batter's average
        production (neutral — neither batter nor pitcher has an edge)
      - factor < 1.0 when pitcher dominates (allows little; their pitch suppresses batter)
      - factor > 1.0 when batter has an edge on this pitcher's primary pitches

    Both batter and pitcher values come entirely from player Statcast data.
    Falls back to player-level averages (never league constants) when a
    specific pitch type is missing from either player's Statcast data.
    """
    pitch_mix = pitcher.get("pitch_mix", {})
    if not pitch_mix:
        return rates

    batter_woba_vs_pitch = batter.get("woba_vs_pitch", {})
    pitcher_woba_allowed = pitcher.get("pitch_woba_allowed", {})
    batter_overall_woba = batter.get("woba") or 0.0
    pitcher_avg_woba = _pitcher_avg_woba_allowed(pitcher)

    if pitcher_avg_woba <= 0:
        return rates

    # Normalizer: batter's overall wOBA (or pitcher avg if batter has no data)
    normalizer = batter_overall_woba if batter_overall_woba > 0 else pitcher_avg_woba

    geo_sum = 0.0
    total_weight = 0.0

    for pitch_type, pct in pitch_mix.items():
        if pct <= 0:
            continue
        pitcher_woba_pt = pitcher_woba_allowed.get(pitch_type, pitcher_avg_woba)
        if pitcher_woba_pt <= 0:
            continue
        # Batter's wOBA vs this pitch type; fall back to overall wOBA if missing
        batter_woba_pt = batter_woba_vs_pitch.get(pitch_type) or batter_overall_woba or pitcher_avg_woba

        # Geometric mean of both players' performance on this pitch type
        geo_mean_pt = np.sqrt(batter_woba_pt * pitcher_woba_pt)
        geo_sum += pct * geo_mean_pt
        total_weight += pct

    if total_weight <= 0:
        return rates

    matchup_factor = (geo_sum / total_weight) / normalizer
    matchup_factor = float(np.clip(matchup_factor, 0.60, 1.60))

    adjusted = {}
    for outcome, prob in rates.items():
        if outcome in POSITIVE_OUTCOMES:
            adjusted[outcome] = prob * matchup_factor
        elif outcome in NEGATIVE_OUTCOMES:
            adjusted[outcome] = prob * (2.0 - matchup_factor)
        else:
            adjusted[outcome] = prob

    return adjusted


def _pitcher_avg_woba_allowed(pitcher: dict) -> float:
    """
    Pitcher's average wOBA allowed across all pitch types with Statcast data.
    Falls back to FIP-based estimate if no pitch-level data available.
    Uses only this pitcher's own data — no league constants.
    """
    woba_by_pitch = pitcher.get("pitch_woba_allowed", {})
    if woba_by_pitch:
        vals = [v for v in woba_by_pitch.values() if v > 0]
        if vals:
            return sum(vals) / len(vals)
    # Use pre-computed overall if available (set during profile build)
    overall = pitcher.get("woba_allowed_overall")
    if overall:
        return overall
    # Last resort: derive from pitcher's own FIP
    fip = pitcher.get("fip", 4.20)
    return max(0.180, min(0.420, FIP_WOBA_INTERCEPT + fip * FIP_WOBA_SLOPE))


def _normalize(rates: dict) -> dict[str, float]:
    total = sum(rates.values())
    if total <= 0:
        n = len(rates)
        return {k: 1.0 / n for k in rates}
    return {k: max(v / total, 0.0) for k, v in rates.items()}


def sample_outcome(probs: dict[str, float]) -> str:
    """Sample a single at-bat outcome from the probability distribution."""
    outcomes = list(probs.keys())
    weights = [probs[o] for o in outcomes]
    return np.random.choice(outcomes, p=weights)
