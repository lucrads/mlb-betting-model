"""
Game-day wind context using Open-Meteo (free, no API key).
Computes outward wind component (mph blowing toward outfield) for each stadium.
"""

import logging
import math
import requests
from datetime import date as date_cls
from functools import lru_cache

logger = logging.getLogger(__name__)

# Stadiums: lat/lon + field bearing (compass direction from home plate to center field)
# bearing: 0 = North, 90 = East, 180 = South, 270 = West
# dome: True = enclosed / retractable roof (wind not applicable)
_STADIUMS: dict[str, dict] = {
    "American Family Field":         {"lat": 43.0280, "lon": -87.9712, "bearing": 315, "dome": False},
    "Angel Stadium":                  {"lat": 33.8003, "lon": -117.8827, "bearing": 215, "dome": False},
    "Busch Stadium":                  {"lat": 38.6226, "lon": -90.1928,  "bearing": 10,  "dome": False},
    "Chase Field":                    {"lat": 33.4455, "lon": -112.0667, "bearing": 335, "dome": True},
    "Citi Field":                     {"lat": 40.7571, "lon": -73.8458,  "bearing": 190, "dome": False},
    "Citizens Bank Park":             {"lat": 39.9061, "lon": -75.1665,  "bearing": 285, "dome": False},
    "Comerica Park":                  {"lat": 42.3390, "lon": -83.0485,  "bearing": 0,   "dome": False},
    "Coors Field":                    {"lat": 39.7559, "lon": -104.9942, "bearing": 245, "dome": False},
    "Dodger Stadium":                 {"lat": 34.0739, "lon": -118.2400, "bearing": 315, "dome": False},
    "Fenway Park":                    {"lat": 42.3467, "lon": -71.0972,  "bearing": 95,  "dome": False},
    "Globe Life Field":               {"lat": 32.7473, "lon": -97.0819,  "bearing": 35,  "dome": True},
    "Great American Ball Park":       {"lat": 39.0979, "lon": -84.5088,  "bearing": 350, "dome": False},
    "Guaranteed Rate Field":          {"lat": 41.8300, "lon": -87.6339,  "bearing": 340, "dome": False},
    "Kauffman Stadium":               {"lat": 39.0517, "lon": -94.4803,  "bearing": 20,  "dome": False},
    "LoanDepot Park":                 {"lat": 25.7781, "lon": -80.2197,  "bearing": 45,  "dome": True},
    "Minute Maid Park":               {"lat": 29.7572, "lon": -95.3555,  "bearing": 235, "dome": True},
    "Nationals Park":                 {"lat": 38.8730, "lon": -77.0074,  "bearing": 340, "dome": False},
    "Oracle Park":                    {"lat": 37.7786, "lon": -122.3893, "bearing": 145, "dome": False},
    "Oriole Park at Camden Yards":    {"lat": 39.2838, "lon": -76.6218,  "bearing": 30,  "dome": False},
    "Petco Park":                     {"lat": 32.7076, "lon": -117.1570, "bearing": 165, "dome": False},
    "PNC Park":                       {"lat": 40.4469, "lon": -80.0057,  "bearing": 350, "dome": False},
    "Progressive Field":              {"lat": 41.4962, "lon": -81.6852,  "bearing": 30,  "dome": False},
    "Rogers Centre":                  {"lat": 43.6414, "lon": -79.3894,  "bearing": 275, "dome": True},
    "T-Mobile Park":                  {"lat": 47.5914, "lon": -122.3325, "bearing": 225, "dome": True},
    "Target Field":                   {"lat": 44.9817, "lon": -93.2781,  "bearing": 300, "dome": False},
    "Tropicana Field":                {"lat": 27.7682, "lon": -82.6534,  "bearing": 90,  "dome": True},
    "Truist Park":                    {"lat": 33.8908, "lon": -84.4678,  "bearing": 225, "dome": False},
    "Wrigley Field":                  {"lat": 41.9484, "lon": -87.6553,  "bearing": 55,  "dome": False},
    "Yankee Stadium":                 {"lat": 40.8296, "lon": -73.9262,  "bearing": 15,  "dome": False},
}

_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _deg_to_compass(deg: float) -> str:
    idx = round(deg / 22.5) % 16
    return _COMPASS[idx]


def _lookup_stadium(venue_name: str) -> dict | None:
    """Fuzzy-match venue name to our stadium database."""
    if not venue_name:
        return None
    vl = venue_name.lower()
    # Exact match first
    for name, info in _STADIUMS.items():
        if name.lower() == vl:
            return {**info, "name": name}
    # Keyword match: stadium with most words in common wins
    best, best_score = None, 0
    vwords = set(vl.split())
    for name, info in _STADIUMS.items():
        score = len(set(name.lower().split()) & vwords)
        if score > best_score:
            best_score = score
            best = {**info, "name": name}
    return best if best_score >= 1 else None


@lru_cache(maxsize=128)
def _fetch_open_meteo(lat: float, lon: float, game_date: str) -> tuple[float, float] | None:
    """
    Return (wind_speed_mph, wind_direction_deg) for the given date and location.
    Uses the forecast API for today/future, archive for past dates.
    Returns None on failure.
    """
    today = date_cls.today().isoformat()
    try:
        if game_date >= today:
            # Forecast endpoint — use current conditions
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "wind_speed_10m,wind_direction_10m",
                    "wind_speed_unit": "mph",
                    "forecast_days": 1,
                },
                timeout=8,
            )
            resp.raise_for_status()
            cur = resp.json().get("current", {})
            speed = cur.get("wind_speed_10m")
            direction = cur.get("wind_direction_10m")
        else:
            # Archive endpoint — use ~6 PM local time (hour index 18)
            resp = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": game_date,
                    "end_date": game_date,
                    "hourly": "wind_speed_10m,wind_direction_10m",
                    "wind_speed_unit": "mph",
                    "timezone": "auto",
                },
                timeout=8,
            )
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})
            speeds = hourly.get("wind_speed_10m", [])
            directions = hourly.get("wind_direction_10m", [])
            # Use hour 18 (6pm) if available, else midday
            idx = 18 if len(speeds) > 18 else (len(speeds) // 2)
            speed = speeds[idx] if speeds else None
            direction = directions[idx] if directions else None

        if speed is None or direction is None:
            return None
        return float(speed), float(direction)

    except Exception as exc:
        logger.debug("Open-Meteo request failed for %s %s: %s", lat, lon, exc)
        return None


def get_wind_context(venue_name: str, game_date: str) -> dict:
    """
    Return a wind context dict for the given game venue and date.

    Keys:
      outward_wind_mph  — component blowing toward outfield (+ve = tailwind, –ve = headwind)
      wind_speed_mph    — raw wind speed
      wind_dir_deg      — wind direction (degrees, meteorological "from")
      wind_dir_label    — compass label (e.g. "NE", "SW")
      dome              — True if enclosed stadium (no wind effect)
      stadium           — matched stadium name (or None)
      description       — human-readable summary
    """
    null_ctx: dict = {
        "outward_wind_mph": 0.0,
        "wind_speed_mph": 0.0,
        "wind_dir_deg": 0.0,
        "wind_dir_label": "N/A",
        "dome": False,
        "stadium": None,
        "description": "No wind data",
    }

    stadium = _lookup_stadium(venue_name)
    if stadium is None:
        logger.debug("Stadium not found for venue: %s", venue_name)
        return null_ctx

    null_ctx["stadium"] = stadium["name"]

    if stadium["dome"]:
        return {**null_ctx, "dome": True, "description": "Dome / retractable roof — wind N/A"}

    result = _fetch_open_meteo(stadium["lat"], stadium["lon"], game_date)
    if result is None:
        return null_ctx

    speed_mph, direction_from = result
    # Compute outward component: positive = wind blowing toward the outfield
    wind_to_deg = (direction_from + 180) % 360
    angle_diff_rad = math.radians(wind_to_deg - stadium["bearing"])
    outward = speed_mph * math.cos(angle_diff_rad)

    dir_label = _deg_to_compass(direction_from)
    if outward >= 2:
        tendency = f"blowing OUT ({outward:+.1f} mph toward CF)"
    elif outward <= -2:
        tendency = f"blowing IN ({outward:+.1f} mph toward HP)"
    else:
        tendency = f"crosswind ({outward:+.1f} mph outward)"

    return {
        "outward_wind_mph": round(outward, 1),
        "wind_speed_mph":   round(speed_mph, 1),
        "wind_dir_deg":     round(direction_from, 0),
        "wind_dir_label":   dir_label,
        "dome":             False,
        "stadium":          stadium["name"],
        "description":      f"{speed_mph:.0f} mph {dir_label} — {tendency}",
    }
