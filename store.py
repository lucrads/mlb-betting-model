"""
Shared data store — the contract between all agents.

Layout:
  store/
    {date}/
      schedule.json          Raw game dicts from fetcher
      player_profiles.json   Enriched profiles (pitchers, lineups, bullpens, wind)
      weather.json           Wind context per game_id
      odds.json              Pre-game sportsbook lines (IMMUTABLE after first write)
      ab_probs.json          Per-batter probability distributions (math agent output)
      simulations.json       Monte Carlo results + edge analysis (engine agent output)
    backtest/
      game_records.json      Append-only historical accuracy log

All writes are atomic (write tmp → rename).
odds.json is NEVER overwritten once it exists — this protects backtest accuracy.
"""

import json
from pathlib import Path
from typing import TypeVar

_ROOT = Path(__file__).parent / "store"

_T = TypeVar("_T")


# ── Internal helpers ─────────────────────────────────────────────────────────

def _dir(date: str) -> Path:
    p = _ROOT / date
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bt_dir() -> Path:
    p = _ROOT / "backtest"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write(path: Path, data) -> None:
    """Atomic write: write to a temp file then rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _read(path: Path, default: _T) -> _T:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Schedule ─────────────────────────────────────────────────────────────────

def write_schedule(date: str, games: list) -> None:
    _write(_dir(date) / "schedule.json", games)


def read_schedule(date: str) -> list:
    return _read(_dir(date) / "schedule.json", default=[])


# ── Player profiles ──────────────────────────────────────────────────────────

def write_player_profiles(date: str, profiles: dict) -> None:
    """profiles keyed by str(game_id)."""
    _write(_dir(date) / "player_profiles.json", profiles)


def read_player_profiles(date: str) -> dict:
    return _read(_dir(date) / "player_profiles.json", default={})


# ── Weather ──────────────────────────────────────────────────────────────────

def write_weather(date: str, weather: dict) -> None:
    """weather keyed by str(game_id)."""
    _write(_dir(date) / "weather.json", weather)


def read_weather(date: str) -> dict:
    return _read(_dir(date) / "weather.json", default={})


# ── Odds (IMMUTABLE after first write) ───────────────────────────────────────

def write_odds(date: str, odds: dict) -> bool:
    """
    Save pre-game odds. NO-OPS silently if the file already exists.
    Returns True if written, False if skipped (file already present).
    odds: {str(game_id): odds_dict}
    """
    path = _dir(date) / "odds.json"
    if path.exists():
        return False
    _write(path, {"saved_at": _now(), "games": odds})
    return True


def read_odds(date: str) -> dict:
    """Returns {str(game_id): odds_dict} or empty dict."""
    raw = _read(_dir(date) / "odds.json", default={})
    return raw.get("games", {})


def odds_saved(date: str) -> bool:
    return (_dir(date) / "odds.json").exists()


# ── At-bat probability distributions (math agent output) ─────────────────────

def write_ab_probs(date: str, ab_probs: dict) -> None:
    """ab_probs keyed by str(game_id)."""
    _write(_dir(date) / "ab_probs.json", ab_probs)


def read_ab_probs(date: str) -> dict:
    return _read(_dir(date) / "ab_probs.json", default={})


# ── Simulation results (engine agent output) ─────────────────────────────────

def write_simulations(date: str, results: dict) -> None:
    """results keyed by str(game_id)."""
    _write(_dir(date) / "simulations.json", results)


def read_simulations(date: str) -> dict:
    return _read(_dir(date) / "simulations.json", default={})


# ── Backtest game records ─────────────────────────────────────────────────────

def read_backtest_records() -> list:
    records = _read(_bt_dir() / "game_records.json", default=None)
    if records is None:
        # Migrate from legacy location at project root
        legacy = Path(__file__).parent / "backtest_results.json"
        if legacy.exists():
            with open(legacy, encoding="utf-8") as f:
                records = json.load(f)
            _write(_bt_dir() / "game_records.json", records)
        else:
            records = []
    return records


def write_backtest_records(records: list) -> None:
    _write(_bt_dir() / "game_records.json", records)


def append_backtest_records(new_records: list) -> None:
    """Merge new records into the store, skipping any game_id already present."""
    existing = read_backtest_records()
    existing_ids = {r["game_id"] for r in existing}
    merged = existing + [r for r in new_records if r["game_id"] not in existing_ids]
    write_backtest_records(merged)


# ── Utility ──────────────────────────────────────────────────────────────────

def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def store_path(date: str) -> Path:
    return _dir(date)
