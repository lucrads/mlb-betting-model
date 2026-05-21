"""
Compute per-batter and per-pitcher matchup details for the detail view.
These numbers explain WHERE the model's edge comes from.
"""

import numpy as np
from config import LEAGUE_AVG_WOBA_BY_PITCH


def compute_matchup_details(game: dict) -> dict:
    """
    Returns structured matchup detail data for a game:
      - away_batters / home_batters: per-batter matchup vs opposing SP
      - away_sp / home_sp: pitcher summary cards
      - away_lineup_edge / home_lineup_edge: how good is this lineup vs this SP
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
    batter_woba = batter.get("woba", 0.320) or 0.320
    woba_vs_hand = batter.get("woba_vs_hand", {}).get(pitcher_hand, batter_woba)

    split_mult = float(np.clip(woba_vs_hand / max(batter_woba, 0.01), 0.60, 1.60))

    pitch_mix = pitcher.get("pitch_mix", {})
    batter_woba_vs_pitch = batter.get("woba_vs_pitch", {})
    matchup_factor = _pitch_mix_factor(pitch_mix, batter_woba_vs_pitch)

    # Combined quality score relative to league average batter (1.0 = average)
    combined = split_mult * matchup_factor
    combined = float(np.clip(combined, 0.50, 1.80))

    return {
        "name": batter.get("name", ""),
        "hand": batter.get("hand", "R"),
        "woba": round(batter_woba, 3),
        "woba_vs_hand": round(woba_vs_hand, 3),
        "split_mult": round(split_mult, 3),
        "matchup_factor": round(matchup_factor, 3),
        "combined": round(combined, 3),
        "vs_pitcher_hand": pitcher_hand,
        # Best/worst pitch types for this batter vs this pitcher's mix
        "pitch_edges": _pitch_edges(pitch_mix, batter_woba_vs_pitch),
    }


def _pitcher_summary(profile: dict) -> dict:
    """Summarise a pitcher for the detail view."""
    pitch_mix = profile.get("pitch_mix", {})
    # Top-3 pitches by usage
    top_pitches = sorted(pitch_mix.items(), key=lambda x: x[1], reverse=True)[:4]
    return {
        "name": profile.get("name", "TBD"),
        "hand": profile.get("hand", "R"),
        "era": profile.get("era", 4.20),
        "fip": profile.get("fip", 4.10),
        "whip": profile.get("whip", 1.30),
        "top_pitches": [{"type": pt, "pct": round(p * 100, 1)} for pt, p in top_pitches],
        "pitch_mix": pitch_mix,
    }


def _pitch_mix_factor(pitch_mix: dict, batter_woba_vs_pitch: dict) -> float:
    """How well does this batter match up to this pitcher's pitch mix? 1.0 = average."""
    if not pitch_mix:
        return 1.0
    total_w = 0.0
    factor = 0.0
    for pt, pct in pitch_mix.items():
        if pct <= 0:
            continue
        league_avg = LEAGUE_AVG_WOBA_BY_PITCH.get(pt, LEAGUE_AVG_WOBA_BY_PITCH.get("OTHER", 0.310))
        batter_val = batter_woba_vs_pitch.get(pt, league_avg)
        if league_avg > 0:
            factor += pct * (batter_val / league_avg)
            total_w += pct
    if total_w <= 0:
        return 1.0
    return float(np.clip(factor / total_w, 0.65, 1.50))


def _pitch_edges(pitch_mix: dict, batter_woba_vs_pitch: dict) -> list[dict]:
    """Return pitches sorted by how much they favor/hurt the batter."""
    edges = []
    for pt, pct in pitch_mix.items():
        if pct < 0.05:
            continue
        league_avg = LEAGUE_AVG_WOBA_BY_PITCH.get(pt, LEAGUE_AVG_WOBA_BY_PITCH.get("OTHER", 0.310))
        batter_val = batter_woba_vs_pitch.get(pt, league_avg)
        relative = batter_val / league_avg if league_avg > 0 else 1.0
        edges.append({
            "type": pt,
            "pct": round(pct * 100, 1),
            "relative": round(relative, 3),
        })
    edges.sort(key=lambda x: abs(x["relative"] - 1.0), reverse=True)
    return edges[:3]


def _lineup_score(batter_profiles: list[dict], opposing_sp: dict) -> float:
    """Average combined matchup score for a lineup vs a pitcher. 1.0 = league average."""
    if not batter_profiles:
        return 1.0
    scores = [_batter_vs_pitcher(b, opposing_sp)["combined"] for b in batter_profiles]
    return round(float(np.mean(scores)), 3)
