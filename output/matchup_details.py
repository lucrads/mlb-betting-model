"""
Compute per-batter and per-pitcher matchup details for the detail view.
These numbers explain WHERE the model's edge comes from.
All ratios use pitcher-specific wOBA allowed as the denominator — no league constants.
"""

import numpy as np
from model.matchup import _pitcher_avg_woba_allowed


def compute_matchup_details(game: dict) -> dict:
    """
    Returns structured matchup detail data for a game:
      - away_batters / home_batters: per-batter matchup vs opposing SP
      - away_sp / home_sp: pitcher summary cards
      - away_lineup_score / home_lineup_score: how good is this lineup vs this SP
    """
    away_profiles = game.get("away_lineup_profiles", [])
    home_profiles = game.get("home_lineup_profiles", [])
    away_sp = game.get("away_pitcher_profile", {})
    home_sp = game.get("home_pitcher_profile", {})

    return {
        "away_batters": [_batter_vs_pitcher(b, home_sp) for b in away_profiles],
        "home_batters": [_batter_vs_pitcher(b, away_sp) for b in home_profiles],
        "away_sp": _pitcher_summary(away_sp),
        "home_sp": _pitcher_summary(home_sp),
        "away_lineup_score": _lineup_score(away_profiles, home_sp),
        "home_lineup_score": _lineup_score(home_profiles, away_sp),
    }


def _batter_vs_pitcher(batter: dict, pitcher: dict) -> dict:
    """Compute how a specific batter matches up against a specific pitcher."""
    pitcher_hand = pitcher.get("hand", "R")
    batter_woba = batter.get("woba") or 0.0
    woba_vs_hand = batter.get("woba_vs_hand", {}).get(pitcher_hand, batter_woba)

    # Split factor: batter's wOBA vs this pitcher hand / batter's overall wOBA
    if batter_woba > 0:
        split_mult = float(np.clip(woba_vs_hand / batter_woba, 0.60, 1.60))
    else:
        split_mult = 1.0  # No batter data — neutral

    pitch_mix = pitcher.get("pitch_mix", {})
    batter_woba_vs_pitch = batter.get("woba_vs_pitch", {})
    matchup_factor = _pitch_mix_factor(pitch_mix, batter_woba_vs_pitch, batter_woba, pitcher)

    combined = split_mult * matchup_factor
    combined = float(np.clip(combined, 0.50, 1.80))

    return {
        "name": batter.get("name", ""),
        "hand": batter.get("hand", "R"),
        "woba": round(batter_woba, 3),
        "woba_vs_hand": round(float(woba_vs_hand), 3),
        "split_mult": round(split_mult, 3),
        "matchup_factor": round(matchup_factor, 3),
        "combined": round(combined, 3),
        "vs_pitcher_hand": pitcher_hand,
        "pitch_edges": _pitch_edges(pitch_mix, batter_woba_vs_pitch, batter_woba, pitcher),
    }


def _pitcher_summary(profile: dict) -> dict:
    """Summarise a pitcher for the detail view."""
    pitch_mix = profile.get("pitch_mix", {})
    top_pitches = sorted(pitch_mix.items(), key=lambda x: x[1], reverse=True)[:4]
    return {
        "name": profile.get("name", "TBD"),
        "hand": profile.get("hand", "R"),
        "era": profile.get("era", 4.50),
        "fip": profile.get("fip", 4.20),
        "whip": profile.get("whip", 1.35),
        "woba_allowed": profile.get("woba_allowed_overall"),
        "top_pitches": [{"type": pt, "pct": round(p * 100, 1)} for pt, p in top_pitches],
        "pitch_mix": pitch_mix,
    }


def _pitch_mix_factor(
    pitch_mix: dict,
    batter_woba_vs_pitch: dict,
    batter_overall_woba: float,
    pitcher: dict,
) -> float:
    """
    Geometric-mean matchup factor — mirrors model/matchup.py exactly.
    = Σ(pitch_pct × sqrt(batter_vs_PT × pitcher_allows_PT)) / normalizer
    """
    if not pitch_mix:
        return 1.0

    pitcher_woba_allowed = pitcher.get("pitch_woba_allowed", {})
    pitcher_avg_woba = _pitcher_avg_woba_allowed(pitcher)
    if pitcher_avg_woba <= 0:
        return 1.0

    normalizer = batter_overall_woba if batter_overall_woba > 0 else pitcher_avg_woba
    geo_sum = 0.0
    total_w = 0.0

    for pt, pct in pitch_mix.items():
        if pct <= 0:
            continue
        pitcher_woba_pt = pitcher_woba_allowed.get(pt, pitcher_avg_woba)
        if pitcher_woba_pt <= 0:
            continue
        batter_woba_pt = batter_woba_vs_pitch.get(pt) or batter_overall_woba or pitcher_avg_woba
        geo_sum += pct * np.sqrt(batter_woba_pt * pitcher_woba_pt)
        total_w += pct

    if total_w <= 0:
        return 1.0
    return float(np.clip((geo_sum / total_w) / normalizer, 0.60, 1.60))


def _pitch_edges(
    pitch_mix: dict,
    batter_woba_vs_pitch: dict,
    batter_overall_woba: float,
    pitcher: dict,
) -> list[dict]:
    """
    Return pitches sorted by how much they favor/hurt the batter vs this pitcher.
    relative > 1.0 → batter advantage on this pitch type.
    relative < 1.0 → pitcher advantage.
    Uses geometric mean normalized by batter overall wOBA — same scale as the model.
    """
    pitcher_woba_allowed = pitcher.get("pitch_woba_allowed", {})
    pitcher_avg_woba = _pitcher_avg_woba_allowed(pitcher)
    normalizer = batter_overall_woba if batter_overall_woba > 0 else pitcher_avg_woba
    edges = []

    if normalizer <= 0:
        return []

    for pt, pct in pitch_mix.items():
        if pct < 0.05:
            continue
        pitcher_woba_pt = pitcher_woba_allowed.get(pt, pitcher_avg_woba)
        if pitcher_woba_pt <= 0:
            continue
        batter_woba_pt = batter_woba_vs_pitch.get(pt) or batter_overall_woba or pitcher_avg_woba
        geo_pt = float(np.sqrt(batter_woba_pt * pitcher_woba_pt))
        relative = geo_pt / normalizer
        edges.append({
            "type": pt,
            "pct": round(pct * 100, 1),
            "relative": round(relative, 3),
            "batter_woba": round(float(batter_woba_pt), 3),
            "pitcher_woba_allowed": round(float(pitcher_woba_pt), 3),
        })

    edges.sort(key=lambda x: abs(x["relative"] - 1.0), reverse=True)
    return edges[:3]


def _lineup_score(batter_profiles: list[dict], opposing_sp: dict) -> float:
    """Average combined matchup score for a lineup vs a pitcher. 1.0 = neutral."""
    if not batter_profiles:
        return 1.0
    scores = [_batter_vs_pitcher(b, opposing_sp)["combined"] for b in batter_profiles]
    return round(float(np.mean(scores)), 3)
