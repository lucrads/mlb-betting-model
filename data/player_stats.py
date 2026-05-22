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
from config import CURRENT_SEASON, FIP_WOBA_INTERCEPT, FIP_WOBA_SLOPE

logger = logging.getLogger(__name__)

pybaseball.cache.enable()  # type: ignore[attr-defined]


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
        player_info = {"hand": result.get("bat_side", "R")}
        for s in result.get("stats", []):
            if str(s.get("season")) == str(season):
                player_info["stats"] = s["stats"]
                return player_info
        # Exact season not found — return without stats so caller handles fallback
        return player_info
    except Exception as exc:
        logger.debug("Batter season stats failed id=%s: %s", mlbam_id, exc)
        return {}



@lru_cache(maxsize=512)
def _get_pitcher_season_stats(mlbam_id: int, season: int) -> dict:
    try:
        result = statsapi.player_stat_data(mlbam_id, type="season", group="pitching")
        player_info = {"hand": result.get("pitch_hand", "R")}
        for s in result.get("stats", []):
            if str(s.get("season")) == str(season):
                player_info["stats"] = s["stats"]
                return player_info
        # Exact season not found — return without stats so caller handles fallback
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


def _woba_by_pitch_type(df: pd.DataFrame, woba_col: str = "woba_value") -> dict:
    """Aggregate mean wOBA per pitch-type from a Statcast DataFrame of terminal events."""
    result = {}
    if df.empty or "pitch_type" not in df.columns or woba_col not in df.columns:
        return result
    for pt, grp in df.groupby("pitch_type"):  # type: ignore[union-attr]
        if not pt or pd.isna(pt):  # type: ignore[arg-type]
            continue
        val = grp[woba_col].mean()  # type: ignore[union-attr]
        if not pd.isna(val):  # type: ignore[arg-type]
            result[str(pt).upper()] = round(float(val), 3)  # type: ignore[arg-type]
    return result


def _compute_woba_from_stats(stats: dict, pa: float) -> float:
    """Compute wOBA from a stat dict. Accepts float values (blended stats)."""
    hr  = float(stats.get("homeRuns", 0))
    hbp = float(stats.get("hitByPitch", 0))
    ibb = float(stats.get("intentionalWalks", 0))
    sh  = float(stats.get("sacBunts", 0))
    h   = float(stats.get("hits", 0))
    doubles = float(stats.get("doubles", 0))
    triples = float(stats.get("triples", 0))
    singles = max(h - doubles - triples - hr, 0.0)
    ubb = max(float(stats.get("baseOnBalls", 0)) - ibb, 0.0)

    woba_num = (0.69 * ubb + 0.72 * hbp + 0.89 * singles
                + 1.27 * doubles + 1.62 * triples + 2.10 * hr)
    woba_den = max(pa - ibb - sh, 1.0)
    return round(woba_num / woba_den, 3)


def _outcome_rates_from_stats(stats: dict, pa: float) -> dict:
    """Compute per-PA outcome rate dict from a stat dict. Accepts float values (blended stats)."""
    hr = float(stats.get("homeRuns", 0))
    bb = float(stats.get("baseOnBalls", 0))
    k  = float(stats.get("strikeOuts", 0))
    h  = float(stats.get("hits", 0))
    doubles = float(stats.get("doubles", 0))
    triples = float(stats.get("triples", 0))
    singles = max(h - doubles - triples - hr, 0.0)

    hr_r      = hr / pa
    bb_r      = bb / pa
    k_r       = k  / pa
    singles_r = singles / pa
    doubles_r = doubles / pa
    triples_r = triples / pa
    out_r     = max(1.0 - hr_r - bb_r - k_r - singles_r - doubles_r - triples_r, 0.05)

    return {
        "HR": hr_r, "BB": bb_r, "K": k_r,
        "1B": singles_r, "2B": doubles_r, "3B": triples_r, "OUT": out_r,
    }


_BATTER_STAT_KEYS = [
    "homeRuns", "baseOnBalls", "intentionalWalks", "hitByPitch",
    "sacBunts", "hits", "doubles", "triples", "strikeOuts",
]

_BATTER_BLEND_TARGET = 100  # PA baseline we fill to before running the model


# ---------------------------------------------------------------------------
# Sprint speed (Baseball Savant leaderboard via pybaseball)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _sprint_speed_by_season(season: int) -> dict:
    """Return {mlbam_id: sprint_speed_ft_per_s} for all players in a season."""
    try:
        df = _quiet(pybaseball.statcast_sprint_speed, season)
        if df is None or df.empty:
            return {}
        id_col  = next((c for c in ("player_id", "mlbam_id") if c in df.columns), None)
        spd_col = next((c for c in ("sprint_speed",) if c in df.columns), None)
        if not id_col or not spd_col:
            return {}
        result: dict = {}
        for _, row in df.iterrows():
            try:
                pid = int(row[id_col])  # type: ignore[arg-type]
                spd = float(row[spd_col])  # type: ignore[arg-type]
                if not pd.isna(spd) and spd > 0:
                    result[pid] = round(spd, 1)
            except (ValueError, TypeError):
                pass
        logger.debug("Loaded sprint speeds for %d players (season %s)", len(result), season)
        return result
    except Exception as exc:
        logger.debug("Sprint speed fetch failed (season %s): %s", season, exc)
        return {}


def _blend_batter_stats(
    curr: dict, curr_pa: int,
    prior: dict, prior_pa: int,
    target_pa: int = _BATTER_BLEND_TARGET,
) -> tuple[dict, float]:
    """
    Fill current-season counting stats up to target_pa using prior-season rates.

    All of the player's current-season PA are used as-is.  The remaining
    (target_pa - curr_pa) PA are filled by scaling prior-season rates.

    Example: 60 current PA, 500 prior PA, target 100 →
        blended[k] = curr[k]  +  prior[k] * (40 / 500)
        total_pa   = 100

    Returns (blended_stats_dict, total_pa).
    No career stats are ever referenced.
    """
    needed = max(0, target_pa - curr_pa)
    if needed == 0 or prior_pa == 0:
        return curr, float(curr_pa)

    scale = needed / prior_pa
    blended = {k: float(curr.get(k, 0)) + float(prior.get(k, 0)) * scale
               for k in _BATTER_STAT_KEYS}
    return blended, float(curr_pa + needed)


# ---------------------------------------------------------------------------
# Batter profile
# ---------------------------------------------------------------------------

def get_batter_profile(player_name: str, player_id: int | None = None, season: int | None = None) -> dict:
    """
    Returns a dict with outcome_rates, woba, woba_vs_hand, woba_vs_pitch, hand.
    All stats are player-specific. Fields default to empty/zero when data is
    unavailable; the matchup engine treats empty data as neutral (no adjustment).
    Pass season= to fetch historical stats (defaults to CURRENT_SEASON).
    """
    _season = season or CURRENT_SEASON

    profile = {
        "name": player_name,
        "outcome_rates": None,   # None = no data; matchup engine handles
        "woba": 0.0,
        "woba_vs_hand": {},      # Empty = no split data; no split adjustment applied
        "woba_vs_pitch": {},     # Empty = no pitch-type data; falls back to overall woba
        "avg_ev": 0.0,           # Average exit velocity on contact (mph)
        "hard_hit_rate": 0.0,    # Fraction of BBE with exit velo >= 95 mph
        "barrel_rate": 0.0,      # Fraction of BBE that are barrels (EV ≥ 98, LA 26–30°)
        "mean_la": None,         # Mean launch angle on BBE (degrees); None = no data
        "sprint_speed": 0.0,     # Sprint speed ft/s from Baseball Savant; 0 = no data
        "hand": "R",
        "data_source": "none",
    }

    mlbam_id = player_id or _mlbam_id_for_name(player_name)
    if not mlbam_id:
        return profile

    # --- MLB Stats API: blend current season + prior season up to 100 PA baseline ---
    # At 100+ PA: pure current season.
    # Below 100 PA: use all current PA, fill remaining from prior-season per-PA rates.
    # No career stats are used.
    season_data = _get_batter_season_stats(mlbam_id, _season)
    profile["hand"] = season_data.get("hand", "R")
    curr_stats = season_data.get("stats", {}) or {}
    curr_pa = int(curr_stats.get("plateAppearances", 0))

    if curr_pa >= _BATTER_BLEND_TARGET:
        stats, pa = curr_stats, float(curr_pa)
        profile["data_source"] = "season"
    else:
        prior_data = _get_batter_season_stats(mlbam_id, _season - 1)
        prior_stats = prior_data.get("stats", {}) or {}
        prior_pa = int(prior_stats.get("plateAppearances", 0))

        if prior_pa > 0:
            stats, pa = _blend_batter_stats(curr_stats, curr_pa, prior_stats, prior_pa)
            profile["data_source"] = "blended"
            logger.debug(
                "Batter %s: %d current PA + %.0f prior-season PA = %.0f PA baseline",
                player_name, curr_pa, pa - curr_pa, pa,
            )
        elif curr_pa > 0:
            # No prior season available — use current as-is
            stats, pa = curr_stats, float(curr_pa)
            profile["data_source"] = "season"
        else:
            stats, pa = {}, 0.0
            profile["data_source"] = "none"

    if pa >= 10 and stats:
        profile["outcome_rates"] = _outcome_rates_from_stats(stats, pa)
        profile["woba"] = _compute_woba_from_stats(stats, pa)

    # --- Statcast enrichment (Baseball Savant) ---
    # Extend to prior season when current-season PA is below the 100-PA threshold,
    # matching the stat-source decision above for consistency.
    start, end = _season_dates(_season)
    sc = _statcast_batter_cached(mlbam_id, start, end)
    if curr_pa < 100:
        prior_start, _ = _season_dates(_season - 1)
        sc_ext = _statcast_batter_cached(mlbam_id, prior_start, end)
        if not sc_ext.empty and len(sc_ext) > len(sc):
            sc = sc_ext
    if not sc.empty:
        # Terminal events only: woba_value is NaN on non-terminal pitches
        terminal: pd.DataFrame = sc[sc["woba_value"].notna()] if "woba_value" in sc.columns else pd.DataFrame()  # type: ignore[assignment]
        if not terminal.empty:
            woba_by_pitch = _woba_by_pitch_type(terminal, "woba_value")
            if woba_by_pitch:
                profile["woba_vs_pitch"] = woba_by_pitch

        # L/R splits from Statcast terminal events
        if not terminal.empty and "p_throws" in terminal.columns:
            for hand in ("L", "R"):
                subset = terminal[terminal["p_throws"] == hand]
                if len(subset) >= 15 and "woba_value" in subset.columns:
                    val = subset["woba_value"].mean()  # type: ignore[union-attr]
                    if not pd.isna(val):  # type: ignore[arg-type]
                        profile["woba_vs_hand"][hand] = round(float(val), 3)  # type: ignore[arg-type]

        # Handedness from stand column
        if "stand" in sc.columns:
            stands = sc["stand"].dropna()
            if not stands.empty:
                profile["hand"] = stands.mode().iloc[0]

        # Exit velocity, barrel rate, and launch angle from BBE
        if "launch_speed" in sc.columns:
            bbe = sc[sc["launch_speed"].notna() & (sc["launch_speed"] > 0)]  # type: ignore[operator]
            if len(bbe) >= 10:
                profile["avg_ev"] = round(float(bbe["launch_speed"].mean()), 1)  # type: ignore[arg-type]
                profile["hard_hit_rate"] = round(float((bbe["launch_speed"] >= 95).mean()), 3)  # type: ignore[arg-type]

                if "launch_angle" in bbe.columns:
                    bbe_la = bbe[bbe["launch_angle"].notna()]  # type: ignore[union-attr]
                    if len(bbe_la) >= 10:
                        profile["mean_la"] = round(float(bbe_la["launch_angle"].mean()), 1)  # type: ignore[arg-type]
                        # Barrels: EV ≥ 98 mph AND launch angle 26–30° (simplified definition)
                        barrels = bbe_la[
                            (bbe_la["launch_speed"] >= 98) &
                            (bbe_la["launch_angle"] >= 26) &
                            (bbe_la["launch_angle"] <= 30)
                        ]
                        profile["barrel_rate"] = round(len(barrels) / len(bbe_la), 3)

    # Sprint speed from Baseball Savant leaderboard
    sprint_map = _sprint_speed_by_season(_season)
    if not sprint_map and _season > 2020:
        # Try prior season if current not yet available
        sprint_map = _sprint_speed_by_season(_season - 1)
    if mlbam_id in sprint_map:
        profile["sprint_speed"] = sprint_map[mlbam_id]

    return profile


# ---------------------------------------------------------------------------
# Pitcher profile
# ---------------------------------------------------------------------------

def get_pitcher_profile(player_name: str, player_id: int | None = None, season: int | None = None) -> dict:
    """
    Returns a dict with era, fip, whip, pitch_mix, pitch_woba_allowed, hand.
    pitch_woba_allowed contains only pitch types with actual Statcast data.
    woba_allowed_overall is the pitcher's average wOBA allowed (for fallback use).
    Pass season= to fetch historical stats (defaults to CURRENT_SEASON).
    """
    _season = season or CURRENT_SEASON

    profile = {
        "name": player_name,
        "era": 4.50,
        "fip": 4.20,
        "whip": 1.35,
        "hand": "R",
        "pitch_mix": {},
        "pitch_woba_allowed": {},     # Populated only from player's Statcast data
        "woba_allowed_overall": None, # Computed from Statcast avg or FIP
        "avg_ev_allowed": 0.0,        # Average exit velocity allowed (mph)
        "hard_hit_allowed_rate": 0.0, # Fraction of BBE allowed with exit velo >= 95
    }

    if player_name in ("Unknown", "TBD", ""):
        profile["woba_allowed_overall"] = _fip_to_woba(profile["fip"])
        return profile

    mlbam_id = player_id or _mlbam_id_for_name(player_name)
    if not mlbam_id:
        profile["woba_allowed_overall"] = _fip_to_woba(profile["fip"])
        return profile

    # --- MLB Stats API: current season (IP ≥ 20) → prior season ---
    season_data = _get_pitcher_season_stats(mlbam_id, _season)
    profile["hand"] = season_data.get("hand", "R")
    stats = season_data.get("stats", {})

    def _parse_ip(s: dict) -> float:
        try:
            return float(s.get("inningsPitched", "0") or 0)
        except (ValueError, TypeError):
            return 0.0

    ip = _parse_ip(stats) if stats else 0.0

    # Fewer than 20 IP this season — use prior season stats instead
    if ip < 20:
        prior_pdata = _get_pitcher_season_stats(mlbam_id, _season - 1)
        prior_pstats = prior_pdata.get("stats", {})
        prior_ip = _parse_ip(prior_pstats) if prior_pstats else 0.0
        if prior_ip > 0:
            stats = prior_pstats
            ip = prior_ip
            profile["hand"] = prior_pdata.get("hand", profile["hand"])
            logger.debug(
                "Pitcher %s: prior-season stats (current IP=%.1f, prior IP=%.1f)",
                player_name, _parse_ip(season_data.get("stats", {}) or {}), prior_ip,
            )

    if stats:
        era_str = stats.get("era", "")
        whip_str = stats.get("whip", "")
        try:
            profile["era"] = float(era_str) if era_str and era_str not in ("-.--", "--") else 4.50
        except (ValueError, TypeError):
            profile["era"] = 4.50
        try:
            profile["whip"] = float(whip_str) if whip_str and whip_str not in ("-.--", "--") else 1.35
        except (ValueError, TypeError):
            profile["whip"] = 1.35

        if ip >= 5:
            hr = int(stats.get("homeRuns", 0))
            bb = int(stats.get("baseOnBalls", 0)) + int(stats.get("hitByPitch", 0))
            k  = int(stats.get("strikeOuts", 0))
            profile["fip"] = round((13 * hr + 3 * bb - 2 * k) / ip + 3.2, 2)

    # --- Statcast pitch mix + wOBA allowed by pitch type ---
    # Extend to prior season when current-season IP is below the 20-IP threshold,
    # matching the stat-source decision above for consistency.
    current_ip = _parse_ip(season_data.get("stats", {}) or {})
    start, end = _season_dates(_season)
    sc = _statcast_pitcher_cached(mlbam_id, start, end)
    if current_ip < 20:
        prior_start, _ = _season_dates(_season - 1)
        sc_ext = _statcast_pitcher_cached(mlbam_id, prior_start, end)
        if not sc_ext.empty and len(sc_ext) > len(sc):
            sc = sc_ext
    if not sc.empty:
        total = len(sc)
        mix = {}
        for pt, grp in sc.groupby("pitch_type"):  # type: ignore[union-attr]
            if not pt or pd.isna(pt):  # type: ignore[arg-type]
                continue
            mix[str(pt).upper()] = round(len(grp) / total, 3)
        if mix:
            profile["pitch_mix"] = mix

        if "p_throws" in sc.columns:
            throws = sc["p_throws"].dropna()
            if not throws.empty:
                profile["hand"] = throws.mode().iloc[0]

        # wOBA allowed per pitch type — terminal events only
        woba_col = "woba_value" if "woba_value" in sc.columns else "estimated_woba_using_speedangle"
        terminal_p: pd.DataFrame = sc[sc[woba_col].notna()] if woba_col in sc.columns else pd.DataFrame()  # type: ignore[assignment]
        woba_allowed = _woba_by_pitch_type(terminal_p, woba_col)
        if woba_allowed:
            profile["pitch_woba_allowed"] = woba_allowed

        # Exit velocity allowed from all BBE with valid launch_speed
        if "launch_speed" in sc.columns:
            bbe = sc[sc["launch_speed"].notna() & (sc["launch_speed"] > 0)]  # type: ignore[operator]
            if len(bbe) >= 10:
                profile["avg_ev_allowed"] = round(float(bbe["launch_speed"].mean()), 1)  # type: ignore[arg-type]
                profile["hard_hit_allowed_rate"] = round(float((bbe["launch_speed"] >= 95).mean()), 3)  # type: ignore[arg-type]

    # Compute overall wOBA allowed: average of pitch-type data, or FIP-derived
    if profile["pitch_woba_allowed"]:
        vals = [v for v in profile["pitch_woba_allowed"].values() if v > 0]
        profile["woba_allowed_overall"] = round(sum(vals) / len(vals), 3) if vals else _fip_to_woba(profile["fip"])
    else:
        profile["woba_allowed_overall"] = _fip_to_woba(profile["fip"])

    return profile


def _fip_to_woba(fip: float) -> float:
    """Convert a pitcher's own FIP to an estimated wOBA allowed."""
    return round(max(0.180, min(0.420, FIP_WOBA_INTERCEPT + fip * FIP_WOBA_SLOPE)), 3)


# ---------------------------------------------------------------------------
# Bullpen profile (team aggregate via MLB Stats API)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _team_id_for_name(team_name: str) -> int | None:
    """Look up team ID from MLB Stats API."""
    try:
        result = statsapi.get("teams", {"sportId": 1, "season": CURRENT_SEASON})
        if not result:
            return None
        teams = result.get("teams", [])
        keyword = team_name.split()[-1].lower()
        for t in teams:
            if keyword in t.get("name", "").lower() or keyword in t.get("teamName", "").lower():
                return t["id"]
    except Exception as exc:
        logger.debug("Team ID lookup failed for %s: %s", team_name, exc)
    return None


def get_bullpen_profile(team_name: str, season: int | None = None) -> dict:
    """Return aggregated bullpen ERA/FIP for the given team using MLB Stats API.
    Pass season= to fetch historical stats (defaults to CURRENT_SEASON).
    """
    _season = season or CURRENT_SEASON
    team_id = _team_id_for_name(team_name)
    if not team_id:
        return {"era": 4.50, "fip": 4.20}

    try:
        result = statsapi.get("team_stats", {
            "teamId": team_id,
            "season": _season,
            "stats": "season",
            "group": "pitching",
        })
        if not result:
            return {"era": 4.50, "fip": 4.20}
        splits = result.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {"era": 4.50, "fip": 4.20}

        s = splits[0]["stat"]
        era_str = s.get("era", "4.50")
        era = float(era_str) if era_str and era_str != "-.--" else 4.50

        ip_str = s.get("inningsPitched", "0")
        ip = float(ip_str) if ip_str else 0
        if ip > 0:
            hr = int(s.get("homeRuns", 0))
            bb = int(s.get("baseOnBalls", 0)) + int(s.get("hitByPitch", 0))
            k  = int(s.get("strikeOuts", 0))
            fip = round((13 * hr + 3 * bb - 2 * k) / ip + 3.2, 2)
        else:
            fip = 4.20

        # Team aggregate includes starters; bullpen is typically 5–10% worse than team ERA
        return {"era": round(era * 1.05, 2), "fip": round(fip * 1.05, 2)}
    except Exception as exc:
        logger.debug("Bullpen stats failed for %s: %s", team_name, exc)
        return {"era": 4.50, "fip": 4.20}
