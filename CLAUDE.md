# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the model

```bash
# Run for today (no odds)
python3 main.py

# Run for a specific date with sportsbook odds
python3 main.py --date 2026-05-21 --odds-key YOUR_KEY

# Override simulation count (use lower number for fast testing)
python3 main.py --date 2026-05-21 --sims 50

# ODDS_API_KEY env var also accepted
ODDS_API_KEY=xxx python3 main.py --date 2026-05-21
```

Output is written to `report_YYYY-MM-DD.html` in the current directory (gitignored).

## Installing dependencies

```bash
pip3 install --only-binary=:all: pyarrow   # must install pyarrow first, binary-only
pip3 install -r requirements.txt
```

`pyarrow` must be installed with `--only-binary=:all:` — building from source requires cmake which is not available on this machine.

## Architecture

The pipeline runs left-to-right through these layers:

```
data/fetcher.py        → data/player_stats.py + data/odds.py
                        ↓
model/matchup.py       → model/simulator.py → model/monte_carlo.py
                        ↓
output/matchup_details.py + output/edge_calc.py → output/report.py → report_*.html
```

**Data layer** (`data/`)
- `fetcher.py` — MLB Stats API (`statsapi.schedule`, `statsapi.get('game', ...)`, `statsapi.lookup_player`). Returns game dicts with lineups from boxscore when game status is Final/In Progress; returns empty lineup lists for Pre-Game/Scheduled.
- `player_stats.py` — MLB Stats API for season totals (ERA, PA, HR, BB, etc.); Baseball Savant via `pybaseball.statcast_batter/pitcher` for pitch-mix and wOBA-by-pitch-type. **Do not use pybaseball FanGraphs leaderboard functions** (`batting_stats`, `pitching_stats`) — they return 403. All results are `lru_cache`'d per session. `pybaseball.cache.enable()` writes to disk between runs.
- `odds.py` — The Odds API. Prefers DraftKings > FanDuel > BetMGM > Caesars. Team name fuzzy matching via `_TEAM_ALIASES`.

**Model layer** (`model/`)
- `matchup.py` — Per-at-bat probability engine. Three-step pipeline: (1) base outcome rates from player's own season/career stats, (2) L/R split multiplier (`batter_woba_vs_hand / batter_overall_woba`), (3) pitch-mix matchup factor using **geometric mean** of batter and pitcher Statcast values per pitch type, normalized by batter's overall wOBA. Zero league-average constants used anywhere. Both multipliers clamped [0.60, 1.60].
- `simulator.py` — Full 9-inning game simulation with base-state machine `[1B, 2B, 3B]`. Switches from starter profile to bullpen profile after `STARTER_INNINGS_LIMIT` (6) innings. Walkoff logic ends the bottom half-inning only when the home team takes the lead.
- `monte_carlo.py` — Runs `config.NUM_SIMULATIONS` (1000) calls to `simulate_game`, aggregates win%, avg runs, score distribution. Reads `config.NUM_SIMULATIONS` at call time (not import time) so `--sims` CLI override works correctly.

**Output layer** (`output/`)
- `matchup_details.py` — Computes per-batter matchup scores (split factor × pitch-mix factor) for the HTML detail view. Does not affect simulation results.
- `edge_calc.py` — Removes vig from sportsbook odds, computes moneyline edge and O/U edge (via normal distribution approximation). Flags BET (≥5%) / LEAN (≥2%) / PASS.
- `report.py` — Jinja2 render of `templates/report.html`. Registers custom filters: `pct`, `american`, `edge_class`, `abs`.

## Key configuration (`config.py`)

| Constant | Purpose |
|---|---|
| `CURRENT_SEASON` | Year used for all stat fetches — update each season |
| `NUM_SIMULATIONS` | Sims per game (1000 default; use 50 for fast dev iteration) |
| `STARTER_INNINGS_LIMIT` | Inning at which bullpen profile takes over (6) |
| `BET_EDGE_THRESHOLD` | Edge % to flag as BET (0.05) |
| `LEAN_EDGE_THRESHOLD` | Edge % to flag as LEAN (0.02) |
| `FIP_WOBA_INTERCEPT` / `FIP_WOBA_SLOPE` | Convert a pitcher's own FIP to estimated wOBA allowed: `woba = 0.230 + fip × 0.022` |

## Important calibration notes

- **No league averages in matchup math**: Every ratio in the pitch-mix factor uses player Statcast data only. Batter pitch-type wOBA (from `woba_value` on terminal events) is compared against what *this pitcher specifically* allows (geometric mean formula). No `LEAGUE_AVG_WOBA_BY_PITCH` constants exist.
- **Geometric mean formula**: `matchup_factor = Σ(pct × sqrt(batter_vs_PT × pitcher_allows_PT)) / batter_overall_woba`. Factor = 1.0 when matchup is neutral; < 1.0 when pitcher dominates; > 1.0 when batter has an edge.
- **Stat fallback hierarchy**: All profiles lead with **current-season stats**. If the sample is too thin (batters: PA < 100; pitchers: IP < 20), `player_stats.py` falls back to the **prior season only** — no career stats are ever used. Statcast windows follow the same rule: extended to include the prior season whenever the stat threshold hasn't been met, keeping pitch-mix and EV data meaningful early in the year.
- **Default lineups**: When a lineup isn't posted (Pre-Game/Scheduled), `main.py` substitutes placeholder batter profiles via `build_default_lineup()`.
- **Bullpen profile**: Derived from team aggregate pitching FIP via MLB Stats API (inflated 5% over team total). wOBA allowed is computed from the team's own FIP via `FIP_WOBA_INTERCEPT + fip × FIP_WOBA_SLOPE` — no league-avg substitution. Pitch mix is a generic bullpen distribution (FF-heavy); since wOBA is uniform across types, the mix only weights the batter's pitch-type splits.
- **wOBA scale**: Uses `woba_value` (actual outcome wOBA on terminal events) throughout — NOT `estimated_woba_using_speedangle`.

## GitHub

Repo: https://github.com/lucrads/mlb-betting-model  
All changes should be committed and pushed. `report_*.html` output files are gitignored.
