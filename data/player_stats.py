"""
Stats loader using MLB Stats API + Baseball Savant (pybaseball Statcast).
No FanGraphs dependency — uses official MLB API for season totals and
Baseball Savant for pitch-level data.
"""

import logging
import os
import sys
import statsapi
import pandas as pd
import pybaseball
from datetime import date
from functools import lru_cache
from config import CURRENT_SEASON, LEAGUE_AVG_WOBA_BY_PITCH

logger = logging.getLogger(__name__)

pybaseball.cache.enable()


def _quiet(func, *args, **kwargs):
    """Call func suppressing any stdout (e.g. pybaseball progress prints)."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout


def _season_dates(year: int) -> tuple[str, str]:
    start = f"{year}-03-20"
    end = date.today().isoformat() if year == date.today().year else f"{year}-11-01"
    return start, end


# ---------------------------------------------------------------------------
# Player ID helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _mlbam_id_for_name(full_name: str) -> int | None:
    """Use statsapi to resolve a player name to an MLBAM ID."""
    try:
        results = statsapi.lookup_player(full_name)
        if results:
            return results[0]["id"]
        # Try last-name only for partial matches
        last = full_name.split()[-1]
        results = statsapi.lookup_player(last)
        if results:
            return results[0]["id"]
    except Exception as exc:
        logger.debug("ID lookup failed for %s: %s", full_name, exc)
    return None


# ---------------------------------------------------------------------------
# Season stats from MLB Stats API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _get_batter_season_stats(mlbam_id: int, season: int) -> dict:
    try:
        result = statsapi.player_stat_data(mlbam_id, type="season", group="hitting")
        player_info = {
            "hand": result.get("bat_side", "R"),
        }
        for s in result.get("stats", []):
            if str(s.get("season")) == str(season):
                player_info["stats"] = s["stats"]
                return player_info
        # Fall back to first entry if season not explicitly matched
        if result.get("stats"):
            player_info["stats"] = result["stats"][0]["stats"]
        return player_info
    except Exception as exc:
        logger.debug("Batter season stats failed id=%s: %s", mlbam_id, exc)
        return {}


@lru_cache(maxsize=512)
def _get_pitcher_season_stats(mlbam_id: int, season: int) -> dict:
    try:
        result = statsapi.player_stat_data(mlbam_id, type="season", group="pitching")
        player_info = {
            "hand": result.get("pitch_hand", "R"),
        }
        for s in result.get("stats", []):
            if str(s.get("season")) == str(season):
                player_info["stats"] = s["stats"]
                return player_info
        if result.get("stats"):
            player_info["stats"] = result["stats"][0]["stats"]
        return player_info
    except Exception as exc:
        logger.debug("Pitcher season stats failed id=%s: %s", mlbam_id, exc)
        return {}


# ---------------------------------------------------------------------------
# Statcast helpers (Baseball Savant)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _statcast_batter_cached(mlbam_id: int, start: str, end: str) -> pd.DataFrame:
    try:
        logger.debug("Fetching Statcast batter data id=%s", mlbam_id)
        df = _quiet(pybaseball.statcast_batter, start, end, player_id=mlbam_id)
        return df if not df.empty else pd.DataFrame()
    except Exception as exc:
        logger.debug("Statcast batter failed id=%s: %s", mlbam_id, exc)
        return pd.DataFrame()


@lru_cache(maxsize=256)
def _statcast_pitcher_cached(mlbam_id: int, start: str, end: str) -> pd.DataFrame:
    try:
        logger.debug("Fetching Statcast pitcher data id=%s", mlbam_id)
        df = _quiet(pybaseball.statcast_pitcher, start, end, player_id=mlbam_id)
        return df if not df.empty else pd.DataFrame()
    except Exception as exc:
        logger.debug("Statcast pitcher failed id=%s: %s", mlbam_id, exc)
        return pd.DataFrame()


def _woba_by_pitch_type(df: pd.DataFrame, woba_col: str = "estimated_woba_using_speedangle") -> dict:
    """Aggregate mean wOBA per pitch-type category from a Statcast DataFrame."""
    result = {}
    if df.empty or "pitch_type" not in df.columns or woba_col not in df.columns:
        return result
    for pt, grp in df.groupby("pitch_type"):
        if not pt or pd.isna(pt):
            continue
        val = grp[woba_col].mean()
        if not pd.isna(val):
            result[str(pt).upper()] = round(float(val), 3)
    return result


# ---------------------------------------------------------------------------
# Batter profile
# ---------------------------------------------------------------------------

def _league_avg_outcome_rates() -> dict:
    return {
        "HR": 0.030, "BB": 0.085, "K": 0.225,
        "1B": 0.145, "2B": 0.050, "3B": 0.005, "OUT": 0.460,
    }


def get_batter_profile(player_name: str, player_id: int | None = None) -> dict:
    """
    Returns a dict with outcome_rates, woba, woba_vs_hand, woba_vs_pitch, hand.
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

    mlbam_id = player_id or _mlbam_id_for_name(player_name)
    if not mlbam_id:
        return profile

    # --- MLB Stats API season totals ---
    season_data = _get_batter_season_stats(mlbam_id, CURRENT_SEASON)
    profile["hand"] = season_data.get("hand", "R")
    stats = season_data.get("stats", {})
    if stats:
        pa = int(stats.get("plateAppearances", 0))
        if pa >= 20:
            hr = int(stats.get("homeRuns", 0))
            bb = int(stats.get("baseOnBalls", 0)) + int(stats.get("intentionalWalks", 0))
            k  = int(stats.get("strikeOuts", 0))
            h  = int(stats.get("hits", 0))
            doubles = int(stats.get("doubles", 0))
            triples = int(stats.get("triples", 0))
            singles = max(h - doubles - triples - hr, 0)

            hr_r = hr / pa
            bb_r = bb / pa
            k_r  = k  / pa
            singles_r  = singles  / pa
            doubles_r  = doubles  / pa
            triples_r  = triples  / pa
            out_r = max(1.0 - hr_r - bb_r - k_r - singles_r - doubles_r - triples_r, 0.05)

            profile["outcome_rates"] = {
                "HR": hr_r, "BB": bb_r, "K": k_r,
                "1B": singles_r, "2B": doubles_r, "3B": triples_r, "OUT": out_r,
            }
            # Compute wOBA from components using standard linear weights
            hbp = int(stats.get("hitByPitch", 0))
            ibb = int(stats.get("intentionalWalks", 0))
            sf  = int(stats.get("sacFlies", 0))
            sh  = int(stats.get("sacBunts", 0))
            ubb = max(bb - ibb, 0)
            woba_num = (0.69 * ubb + 0.72 * hbp + 0.89 * singles
                        + 1.27 * doubles + 1.62 * triples + 2.10 * hr)
            woba_den = max(pa - ibb - sh, 1)
            profile["woba"] = round(woba_num / woba_den, 3)

    # --- Statcast enrichment (Baseball Savant) ---
    start, end = _season_dates(CURRENT_SEASON)
    sc = _statcast_batter_cached(mlbam_id, start, end)
    if not sc.empty:
        # wOBA by pitch type
        woba_by_pitch = _woba_by_pitch_type(sc)
        if woba_by_pitch:
            profile["woba_vs_pitch"].update(woba_by_pitch)

        # L/R splits from p_throws
        if "p_throws" in sc.columns:
            for hand in ("L", "R"):
                subset = sc[sc["p_throws"] == hand]
                if len(subset) >= 20:
                    val = subset["estimated_woba_using_speedangle"].mean()
                    if not pd.isna(val):
                        profile["woba_vs_hand"][hand] = round(float(val), 3)

        # Handedness from stand column (override)
        if "stand" in sc.columns:
            stands = sc["stand"].dropna()
            if not stands.empty:
                profile["hand"] = stands.mode().iloc[0]

    return profile


# ---------------------------------------------------------------------------
# Pitcher profile
# ---------------------------------------------------------------------------

def get_pitcher_profile(player_name: str, player_id: int | None = None) -> dict:
    """
    Returns a dict with era, fip, whip, pitch_mix, pitch_woba_allowed, hand.
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

    if player_name in ("Unknown", "TBD", ""):
        return profile

    mlbam_id = player_id or _mlbam_id_for_name(player_name)
    if not mlbam_id:
        return profile

    # --- MLB Stats API season totals ---
    season_data = _get_pitcher_season_stats(mlbam_id, CURRENT_SEASON)
    profile["hand"] = season_data.get("hand", "R")
    stats = season_data.get("stats", {})
    if stats:
        era_str = stats.get("era", "4.20")
        whip_str = stats.get("whip", "1.30")
        try:
            profile["era"] = float(era_str) if era_str and era_str not in ("-.--", "--") else 4.20
        except (ValueError, TypeError):
            profile["era"] = 4.20
        try:
            profile["whip"] = float(whip_str) if whip_str and whip_str not in ("-.--", "--") else 1.30
        except (ValueError, TypeError):
            profile["whip"] = 1.30

        # Compute FIP from components
        ip_str = stats.get("inningsPitched", "0")
        try:
            ip = float(ip_str)
        except (ValueError, TypeError):
            ip = 0
        if ip >= 5:
            hr = int(stats.get("homeRuns", 0))
            bb = int(stats.get("baseOnBalls", 0)) + int(stats.get("hitByPitch", 0))
            k  = int(stats.get("strikeOuts", 0))
            profile["fip"] = round((13 * hr + 3 * bb - 2 * k) / ip + 3.2, 2)

    # --- Statcast pitch mix + effectiveness ---
    start, end = _season_dates(CURRENT_SEASON)
    sc = _statcast_pitcher_cached(mlbam_id, start, end)
    if not sc.empty:
        total = len(sc)
        mix = {}
        for pt, grp in sc.groupby("pitch_type"):
            if not pt or pd.isna(pt):
                continue
            mix[str(pt).upper()] = round(len(grp) / total, 3)
        if mix:
            profile["pitch_mix"] = mix

        if "p_throws" in sc.columns:
            throws = sc["p_throws"].dropna()
            if not throws.empty:
                profile["hand"] = throws.mode().iloc[0]

        # wOBA allowed by pitch type (use woba_value — actual outcomes)
        woba_col = "woba_value" if "woba_value" in sc.columns else "estimated_woba_using_speedangle"
        woba_allowed = _woba_by_pitch_type(sc, woba_col)
        if woba_allowed:
            profile["pitch_woba_allowed"].update(woba_allowed)

    return profile


# ---------------------------------------------------------------------------
# Bullpen profile (team aggregate via MLB Stats API)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _team_id_for_name(team_name: str) -> int | None:
    """Look up team ID from MLB Stats API."""
    try:
        result = statsapi.get("teams", {"sportId": 1, "season": CURRENT_SEASON})
        teams = result.get("teams", [])
        keyword = team_name.split()[-1].lower()
        for t in teams:
            if keyword in t.get("name", "").lower() or keyword in t.get("teamName", "").lower():
                return t["id"]
    except Exception as exc:
        logger.debug("Team ID lookup failed for %s: %s", team_name, exc)
    return None


def get_bullpen_profile(team_name: str) -> dict:
    """Return aggregated bullpen ERA/FIP for the given team using MLB Stats API."""
    team_id = _team_id_for_name(team_name)
    if not team_id:
        return {"era": 4.20, "fip": 4.10}

    try:
        result = statsapi.get("team_stats", {
            "teamId": team_id,
            "season": CURRENT_SEASON,
            "stats": "season",
            "group": "pitching",
        })
        splits = result.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {"era": 4.20, "fip": 4.10}

        s = splits[0]["stat"]
        era_str = s.get("era", "4.20")
        era = float(era_str) if era_str and era_str != "-.--" else 4.20

        ip_str = s.get("inningsPitched", "0")
        ip = float(ip_str) if ip_str else 0
        if ip > 0:
            hr = int(s.get("homeRuns", 0))
            bb = int(s.get("baseOnBalls", 0)) + int(s.get("hitByPitch", 0))
            k  = int(s.get("strikeOuts", 0))
            fip = round((13 * hr + 3 * bb - 2 * k) / ip + 3.2, 2)
        else:
            fip = 4.10

        # Team aggregate includes starters; bullpen is typically 5–10% worse than team ERA
        return {"era": round(era * 1.05, 2), "fip": round(fip * 1.05, 2)}
    except Exception as exc:
        logger.debug("Bullpen stats failed for %s: %s", team_name, exc)
        return {"era": 4.20, "fip": 4.10}
