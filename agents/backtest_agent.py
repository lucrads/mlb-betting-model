"""
Backtest Agent — retroactive accuracy analysis using stored pre-game odds.

For each completed game:
  1. Reads pre-game odds from store/{date}/odds.json (IMMUTABLE — locked at game time)
  2. Reads simulation results from store/{date}/simulations.json
  3. Reads actual scores from the MLB Stats API
  4. Computes: ML accuracy, confidence tier, edge vs stored odds, O/U accuracy

Writes individual game records to store/backtest/game_records.json.
Migrates legacy backtest_results.json if it exists.

Run:
  python3 agents/backtest_agent.py --start 2026-03-20 --end 2026-05-21
  python3 agents/backtest_agent.py --start 2026-03-20 --end 2026-05-21 --refresh
  python3 agents/backtest_agent.py --date 2026-05-21   (single date)
"""

import sys
import os
import logging
import argparse
from datetime import date as date_cls, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import store
import config
from data.fetcher import get_games_for_date
from output.edge_calc import compute_edge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [backtest]  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_FINAL_STATUSES = {"Final", "Game Over"}
SEASON_START = f"{config.CURRENT_SEASON}-03-20"


def parse_args():
    p = argparse.ArgumentParser(description="MLB Backtest Agent")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--date", default=None, help="Single date YYYY-MM-DD")
    g.add_argument("--start", default=SEASON_START, help="Start date for range")
    p.add_argument("--end", default=None, help="End date (default: yesterday)")
    p.add_argument("--refresh", action="store_true", help="Re-process all dates (ignore existing records)")
    return p.parse_args()


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.70:
        return "HIGH"
    if confidence >= 0.60:
        return "MED"
    if confidence >= 0.55:
        return "LEAN"
    return "PASS"


def process_date(date_str: str, existing_ids: set, refresh: bool) -> list[dict]:
    """
    Process all completed games for a date. Returns list of new game records.
    """
    games = get_games_for_date(date_str)
    if not games:
        return []

    finished = [g for g in games if g.get("status") in _FINAL_STATUSES]
    if not finished:
        return []

    # Load stored data for this date
    sims = store.read_simulations(date_str)
    stored_odds = store.read_odds(date_str)  # immutable pre-game odds

    new_records = []

    for game in finished:
        gid = str(game["game_id"])

        if not refresh and gid in existing_ids:
            continue

        # Must have simulation results
        sim = sims.get(gid)
        if not sim:
            # Fall back to running fresh simulation inline (for historical dates without stored sims)
            sim = _run_inline_sim(game, date_str)
            if not sim:
                logger.debug("  Skipping game %s — no sim results", gid)
                continue

        home_score = game.get("home_score")
        away_score = game.get("away_score")
        if home_score is None or away_score is None:
            logger.debug("  Skipping game %s — no final scores", gid)
            continue

        home_won_actual = home_score > away_score

        # Model pick
        home_win_pct = sim["home_win_pct"]
        away_win_pct = sim["away_win_pct"]
        model_picks_home = home_win_pct >= away_win_pct
        model_correct_ml = (model_picks_home == home_won_actual)

        confidence = max(home_win_pct, away_win_pct)
        confidence_rec = _confidence_label(confidence)

        # Edge vs stored odds
        game_odds = stored_odds.get(gid)
        edge = compute_edge(sim, game_odds) if game_odds else compute_edge(sim, None)

        model_total = sim["avg_total_runs"]
        actual_total = home_score + away_score
        book_total = game_odds.get("total") if game_odds else None

        # Bet result tracking — ML and O/U
        bet_results = []
        if game_odds:
            # ML bets
            for side in ("home", "away"):
                edge_val = edge.get(f"{side}_edge")
                if edge_val and edge_val >= config.LEAN_EDGE_THRESHOLD:
                    side_won = home_won_actual if side == "home" else not home_won_actual
                    bet_results.append({
                        "type": "ML",
                        "side": side,
                        "edge": round(edge_val, 4),
                        "rec": "BET" if edge_val >= config.BET_EDGE_THRESHOLD else "LEAN",
                        "won": side_won,
                    })
            # O/U bets
            if book_total is not None:
                push = (actual_total == int(book_total))
                for side, edge_key in [("over", "total_edge_over"), ("under", "total_edge_under")]:
                    edge_val = edge.get(edge_key)
                    if edge_val and edge_val >= config.LEAN_EDGE_THRESHOLD:
                        if push:
                            won = None  # push
                        elif side == "over":
                            won = actual_total > book_total
                        else:
                            won = actual_total < book_total
                        bet_results.append({
                            "type": "OU",
                            "side": side,
                            "edge": round(edge_val, 4),
                            "rec": "BET" if edge_val >= config.BET_EDGE_THRESHOLD else "LEAN",
                            "won": won,
                            "push": push,
                        })

        record = {
            "game_id": gid,
            "date": date_str,
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_pitcher": (game.get("home_pitcher") or {}).get("name", "Unknown"),
            "away_pitcher": (game.get("away_pitcher") or {}).get("name", "Unknown"),
            "home_win_pct": home_win_pct,
            "away_win_pct": away_win_pct,
            "model_total": round(model_total, 2),
            "home_score": home_score,
            "away_score": away_score,
            "actual_total": actual_total,
            "home_won": home_won_actual,
            "model_correct_ml": model_correct_ml,
            "confidence": round(confidence, 4),
            "confidence_rec": confidence_rec,
            "recommendation": edge.get("recommendation", "PASS"),
            "had_odds": bool(game_odds),
            "book_total": book_total,
            "total_edge_over": edge.get("total_edge_over"),
            "total_edge_under": edge.get("total_edge_under"),
            "ou_pick": edge.get("ou_pick"),
            "ou_edge": edge.get("ou_edge"),
            "ou_rec": edge.get("ou_rec", "PASS"),
            "bet_results": bet_results,
        }

        new_records.append(record)
        status_icon = "✓" if model_correct_ml else "✗"
        logger.info(
            "  %s  %s @ %s  | Model: %s  Actual: %d-%d  [%s]",
            status_icon, game["away_team"], game["home_team"],
            "HOME" if model_picks_home else "AWAY",
            home_score, away_score, confidence_rec,
        )

    return new_records


def _run_inline_sim(game: dict, date_str: str) -> dict | None:
    """
    Fallback: run a quick simulation inline for historical games missing stored sims.
    Uses stored player profiles if available, otherwise skips.
    """
    profiles = store.read_player_profiles(date_str)
    gid = str(game["game_id"])
    if not profiles or gid not in profiles:
        return None

    try:
        from model.monte_carlo import run_simulations
        p = profiles[gid]
        game_dict = {
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_pitcher_profile": p["home_pitcher_profile"],
            "away_pitcher_profile": p["away_pitcher_profile"],
            "home_bullpen_profile": p["home_bullpen_profile"],
            "away_bullpen_profile": p["away_bullpen_profile"],
            "home_lineup_profiles": p["home_lineup_profiles"],
            "away_lineup_profiles": p["away_lineup_profiles"],
            "outward_wind_mph": p.get("outward_wind_mph", 0.0),
        }
        # Use reduced sims for historical backfill
        original = config.NUM_SIMULATIONS
        config.NUM_SIMULATIONS = 200
        result = run_simulations(game_dict)
        config.NUM_SIMULATIONS = original
        return result
    except Exception as e:
        logger.debug("  Inline sim failed for game %s: %s", gid, e)
        return None


def run(start: str | None = None, end: str | None = None, date: str | None = None, refresh: bool = False) -> list[dict]:
    """
    Process backtest for a date range or a single date.
    Returns all new records added.
    """
    if date:
        dates = [date]
    else:
        start = start or SEASON_START
        end = end or (date_cls.today() - timedelta(days=1)).isoformat()
        d = date_cls.fromisoformat(start)
        end_d = date_cls.fromisoformat(end)
        dates = []
        while d <= end_d:
            dates.append(d.isoformat())
            d += timedelta(days=1)

    logger.info("=== Backtest Agent | %s → %s | refresh=%s ===",
                dates[0], dates[-1], refresh)

    existing = store.read_backtest_records()
    existing_ids = set() if refresh else {r["game_id"] for r in existing}
    logger.info("Existing records: %d", len(existing))

    all_new = []
    for date_str in dates:
        logger.info("Processing %s...", date_str)
        new = process_date(date_str, existing_ids, refresh)
        if new:
            all_new.extend(new)
            existing_ids.update(r["game_id"] for r in new)
            logger.info("  +%d new records", len(new))

    if all_new:
        store.append_backtest_records(all_new)
        logger.info("Appended %d new game records. Total: %d",
                    len(all_new), len(existing) + len(all_new))
    else:
        logger.info("No new records to add.")

    return all_new


if __name__ == "__main__":
    args = parse_args()
    run(
        start=args.start if not args.date else None,
        end=args.end if not args.date else None,
        date=args.date,
        refresh=args.refresh,
    )
