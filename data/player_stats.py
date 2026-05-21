"""
pybaseball-based stats loader.

Fetches and caches:
  - Season batting/pitching leaderboards (FanGraphs)
  - Statcast pitch mix and pitch-type wOBA per pitcher
  - Statcast pitch-type wOBA per batter
  - L/R splits via FanGraphs splits API
"""

import logging
import requests
import pandas as pd
import numpy as np
import pybaseball
from datetime import date
from functools import lru_cache
from config import CURRENT_SEASON, LEAGUE_AVG_WOBA_BY_PITCH

logger = logging.getLogger(__name__)

pybaseball.cache.enable()

_batting_df: pd.DataFrame | None = None
_pitching_df: pd.DataFrame | None = None


def _season_dates(year: int) -> tuple[str, str]:
    start = f"{year}-03-20"
    end = date.today().isoformat() if year == date.today().year else f"{year}-11-01"
    return start, end


# ---------------------------------------------------------------------------
# Season leaderboards
# ---------------------------------------------------------------------------

def get_batting_stats() -> pd.DataFrame:
    global _batting_df
    if _batting_df is None:
        logger.info("Loading FanGraphs batting stats for %d...", CURRENT_SEASON)
        _batting_df = pybaseball.batting_stats(CURRENT_SEASON, qual=50)
        _batting_df.columns = [c.strip() for c in _batting_df.columns]
    return _batting_df


def get_pitching_stats() -> pd.DataFrame:
    global _pitching_df
    if _pitching_df is None:
        logger.info("Loading FanGraphs pitching stats for %d...", CURRENT_SEASON)
        _pitching_df = pybaseball.pitching_stats(CURRENT_SEASON, qual=1)
        _pitching_df.columns = [c.strip() for c in _pitching_df.columns]
    return _pitching_df


# ---------------------------------------------------------------------------
# Player ID lookup
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def lookup_player_id(last: str, first: str) -> int | None:
    try:
        result = pybaseball.playerid_lookup(last, first)
        if result.empty:
            return None
        row = result.iloc[0]
        return int(row.get("key_mlbam") or row.get("mlbam_id") or 0) or None
    except Exception as e:
        logger.warning("ID lookup failed for %s %s: %s", first, last, e)
        return None


def name_to_id(full_name: str) -> int | None:
    parts = full_name.strip().split()
    if len(parts) < 2:
        return None
    last = parts[-1]
    first = parts[0]
    return lookup_player_id(last, first)


# ---------------------------------------------------------------------------
# Batter profile
# ---------------------------------------------------------------------------

def get_batter_profile(player_name: str, player_id: int | None = None) -> dict:
    """
    Returns a dict with:
      - outcome_rates: {HR, BB, K, 1B, 2B, 3B, OUT}
      - woba_vs_hand: {L: float, R: float}
      - woba_vs_pitch: {pitch_type: float, ...}
      - hand: 'L' | 'R' | 'S'
    Falls back to league averages when data is missing.
    """
    profile = {
        "name": player_name,
        "outcome_rates": _league_avg_outcome_rates(),
        "woba": 0.320,
        "woba_vs_hand": {"L": 0.320, "R": 0.320},
        "woba_vs_pitch": dict(LEAGUE_AVG_WOBA_BY_PITCH),
        "hand": "R",
    }

    batting = get_batting_stats()
    row = _find_player_row(batting, player_name)
    if row is not None:
        profile.update(_extract_batter_rates(row))

    mlbam_id = player_id or name_to_id(player_name)
    if mlbam_id:
        _enrich_batter_statcast(profile, mlbam_id)
        _enrich_batter_splits(profile, mlbam_id)

    return profile


def _find_player_row(df: pd.DataFrame, name: str):
    name_col = "Name" if "Name" in df.columns else df.columns[0]
    matches = df[df[name_col].str.contains(name.split()[-1], case=False, na=False)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _league_avg_outcome_rates() -> dict:
    return {"HR": 0.030, "BB": 0.085, "K": 0.225, "1B": 0.145, "2B": 0.050, "3B": 0.005, "OUT": 0.460}


def _extract_batter_rates(row) -> dict:
    pa = float(row.get("PA", 1) or 1)
    if pa < 10:
        return {}

    hr = float(row.get("HR", 0) or 0) / pa
    bb = float(row.get("BB", 0) or 0) / pa
    k  = float(row.get("SO", row.get("K", 0)) or 0) / pa
    h  = float(row.get("H", 0) or 0) / pa
    doubles = float(row.get("2B", 0) or 0) / pa
    triples = float(row.get("3B", 0) or 0) / pa
    singles = max(h - doubles - triples - hr, 0)

    total_event = hr + bb + k + singles + doubles + triples
    out = max(1.0 - total_event, 0.05)

    woba_val = float(row.get("wOBA", 0.320) or 0.320)

    return {
        "outcome_rates": {
            "HR": hr, "BB": bb, "K": k,
            "1B": singles, "2B": doubles, "3B": triples,
            "OUT": out,
        },
        "woba": woba_val,
    }


def _enrich_batter_statcast(profile: dict, mlbam_id: int) -> None:
    start, end = _season_dates(CURRENT_SEASON)
    try:
        sc = pybaseball.statcast_batter(start, end, player_id=mlbam_id)
        if sc.empty:
            return
        # wOBA by pitch type
        woba_by_pitch = {}
        for pt, grp in sc.groupby("pitch_type"):
            if pt and not pd.isna(pt):
                woba_col = grp["estimated_woba_using_speedangle"] if "estimated_woba_using_speedangle" in grp.columns else grp.get("woba_value")
                if woba_col is not None:
                    val = woba_col.mean()
                    if not pd.isna(val):
                        woba_by_pitch[str(pt).upper()] = round(float(val), 3)
        if woba_by_pitch:
            profile["woba_vs_pitch"].update(woba_by_pitch)

        # handedness from stand column
        if "stand" in sc.columns:
            stands = sc["stand"].dropna()
            if not stands.empty:
                profile["hand"] = stands.mode()[0]
    except Exception as e:
        logger.warning("Statcast batter fetch failed for id=%s: %s", mlbam_id, e)


def _enrich_batter_splits(profile: dict, mlbam_id: int) -> None:
    """Fetch L/R split wOBA from FanGraphs splits API."""
    try:
        url = (
            f"https://www.fangraphs.com/api/leaders/splits/splits-leaders"
            f"?columngroup=&stat=bat&startseason={CURRENT_SEASON}&endseason={CURRENT_SEASON}"
            f"&splitArr=&splitArrPitches=&autoPt=false&splitTeams=false&statgroup=1"
            f"&startinnings=1&endinnings=9&numteams=0&active=1&postseason=0&players={mlbam_id}&type=0"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return
        data = resp.json()
        splits_data = data.get("data", data) if isinstance(data, dict) else data
        woba_vs = {}
        for entry in splits_data:
            split = str(entry.get("Split", entry.get("split", "")))
            woba = entry.get("wOBA")
            if split == "vs LHP" and woba:
                woba_vs["L"] = float(woba)
            elif split == "vs RHP" and woba:
                woba_vs["R"] = float(woba)
        if woba_vs:
            profile["woba_vs_hand"].update(woba_vs)
    except Exception as e:
        logger.debug("Splits fetch failed for id=%s: %s", mlbam_id, e)


# ---------------------------------------------------------------------------
# Pitcher profile
# ---------------------------------------------------------------------------

def get_pitcher_profile(player_name: str, player_id: int | None = None) -> dict:
    """
    Returns a dict with:
      - era, fip, whip
      - pitch_mix: {pitch_type: pct, ...}  (sums to 1.0)
      - pitch_woba_allowed: {pitch_type: float, ...}
      - hand: 'L' | 'R'
      - bullpen_era (fallback)
    """
    profile = {
        "name": player_name,
        "era": 4.20,
        "fip": 4.10,
        "whip": 1.30,
        "hand": "R",
        "pitch_mix": {"FF": 0.55, "SL": 0.20, "CH": 0.15, "CU": 0.10},
        "pitch_woba_allowed": dict(LEAGUE_AVG_WOBA_BY_PITCH),
    }

    pitching = get_pitching_stats()
    row = _find_player_row(pitching, player_name)
    if row is not None:
        profile.update(_extract_pitcher_rates(row))

    mlbam_id = player_id or name_to_id(player_name)
    if mlbam_id:
        _enrich_pitcher_statcast(profile, mlbam_id)

    return profile


def _extract_pitcher_rates(row) -> dict:
    return {
        "era": float(row.get("ERA", 4.20) or 4.20),
        "fip": float(row.get("FIP", 4.10) or 4.10),
        "whip": float(row.get("WHIP", 1.30) or 1.30),
    }


def _enrich_pitcher_statcast(profile: dict, mlbam_id: int) -> None:
    start, end = _season_dates(CURRENT_SEASON)
    try:
        sc = pybaseball.statcast_pitcher(start, end, player_id=mlbam_id)
        if sc.empty:
            return

        # Pitch mix
        total = len(sc)
        mix = {}
        for pt, grp in sc.groupby("pitch_type"):
            if pt and not pd.isna(pt):
                mix[str(pt).upper()] = round(len(grp) / total, 3)
        if mix:
            profile["pitch_mix"] = mix

        # Pitcher handedness
        if "p_throws" in sc.columns:
            throws = sc["p_throws"].dropna()
            if not throws.empty:
                profile["hand"] = throws.mode()[0]

        # wOBA allowed by pitch type
        woba_allowed = {}
        for pt, grp in sc.groupby("pitch_type"):
            if pt and not pd.isna(pt):
                woba_col = grp.get("woba_value") if "woba_value" in grp.columns else None
                if woba_col is not None:
                    val = woba_col.mean()
                    if not pd.isna(val):
                        woba_allowed[str(pt).upper()] = round(float(val), 3)
        if woba_allowed:
            profile["pitch_woba_allowed"].update(woba_allowed)

    except Exception as e:
        logger.warning("Statcast pitcher fetch failed for id=%s: %s", mlbam_id, e)


# ---------------------------------------------------------------------------
# Bullpen profile (team aggregate)
# ---------------------------------------------------------------------------

def get_bullpen_profile(team_name: str) -> dict:
    """Aggregate bullpen ERA/FIP from pitching leaderboard for a given team."""
    pitching = get_pitching_stats()
    team_col = "Team" if "Team" in pitching.columns else None
    if team_col is None:
        return {"era": 4.20, "fip": 4.10}

    # Filter relievers: IP < 40 as a rough heuristic
    team_df = pitching[pitching[team_col].str.contains(team_name.split()[-1], case=False, na=False)]
    if team_df.empty:
        return {"era": 4.20, "fip": 4.10}

    relievers = team_df[pd.to_numeric(team_df.get("IP", pd.Series([])), errors="coerce").fillna(0) < 40]
    if relievers.empty:
        relievers = team_df

    era = float(pd.to_numeric(relievers["ERA"], errors="coerce").mean() or 4.20)
    fip = float(pd.to_numeric(relievers.get("FIP", pd.Series([])), errors="coerce").mean() or 4.10)
    return {"era": era, "fip": fip}
