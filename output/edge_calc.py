"""
Convert model win probabilities to fair odds and compute edge vs sportsbook lines.
"""

import math
from config import BET_EDGE_THRESHOLD, LEAN_EDGE_THRESHOLD


def american_to_prob(american_odds: int) -> float:
    """Convert American odds to implied probability (no vig removal)."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    """Remove bookmaker vig to get fair no-vig probabilities."""
    total = home_prob + away_prob
    if total <= 0:
        return 0.5, 0.5
    return home_prob / total, away_prob / total


def prob_to_american(prob: float) -> int:
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return round(-(prob / (1 - prob)) * 100)
    else:
        return round(((1 - prob) / prob) * 100)


def compute_edge(simulation_result: dict, odds: dict | None) -> dict:
    """
    Compare model probabilities to sportsbook lines and return edge analysis.

    Returns:
      home_edge, away_edge (floats, positive = +EV for that side)
      total_edge_over, total_edge_under
      recommendation: 'BET' | 'LEAN' | 'PASS'
      best_bet: description of the best bet if any
    """
    model_home_prob = simulation_result["home_win_pct"]
    model_away_prob = simulation_result["away_win_pct"]
    model_total = simulation_result["avg_total_runs"]

    # Normalize to sum to 1.0 (ties in 15-inning cap cause slight discrepancy)
    _total_win = model_home_prob + model_away_prob
    if _total_win > 0:
        model_home_prob_norm = model_home_prob / _total_win
        model_away_prob_norm = model_away_prob / _total_win
    else:
        model_home_prob_norm = model_away_prob_norm = 0.5

    result = {
        "model_home_prob": model_home_prob,
        "model_away_prob": model_away_prob,
        "model_fair_home_ml": prob_to_american(model_home_prob_norm),
        "model_fair_away_ml": prob_to_american(model_away_prob_norm),
        "model_total": model_total,
        "home_edge": None,
        "away_edge": None,
        "total_edge_over": None,
        "total_edge_under": None,
        "book_home_ml": None,
        "book_away_ml": None,
        "best_home_ml": None,
        "best_away_ml": None,
        "best_home_book": None,
        "best_away_book": None,
        "all_ml": {},
        "book_total": None,
        "book_over_ml": None,
        "book_under_ml": None,
        "ou_pick": None,      # "Over" | "Under" | None
        "ou_edge": None,      # the edge value for our O/U pick
        "ou_rec": "PASS",     # "BET" | "LEAN" | "PASS"
        "recommendation": "PASS",
        "best_bets": [],
        "has_odds": False,
    }

    if not odds:
        return result

    result["has_odds"] = True
    # Use best-available line for edge calculation (line shopping)
    best_home_ml = odds.get("best_home_ml") or odds.get("home_ml")
    best_away_ml = odds.get("best_away_ml") or odds.get("away_ml")
    result["book_home_ml"]    = odds.get("home_ml")
    result["book_away_ml"]    = odds.get("away_ml")
    result["best_home_ml"]    = best_home_ml
    result["best_away_ml"]    = best_away_ml
    result["best_home_book"]  = odds.get("best_home_book", odds.get("bookmaker", ""))
    result["best_away_book"]  = odds.get("best_away_book", odds.get("bookmaker", ""))
    result["all_ml"]          = odds.get("all_ml", {})
    result["book_total"]      = odds.get("total")
    result["book_over_ml"]    = odds.get("over_ml", -110)
    result["book_under_ml"]   = odds.get("under_ml", -110)

    # Moneyline edge — compare model against BEST available line
    home_ml = best_home_ml
    away_ml = best_away_ml
    if home_ml is not None and away_ml is not None:
        raw_home_prob = american_to_prob(home_ml)
        raw_away_prob = american_to_prob(away_ml)
        fair_home, fair_away = remove_vig(raw_home_prob, raw_away_prob)

        home_edge = model_home_prob - fair_home
        away_edge = model_away_prob - fair_away
        result["home_edge"] = round(home_edge, 4)
        result["away_edge"] = round(away_edge, 4)

    # Totals edge — use actual simulated run distribution when available
    book_total = odds.get("total")
    if book_total is not None:
        over_prob_book = american_to_prob(odds.get("over_ml", -110))
        under_prob_book = american_to_prob(odds.get("under_ml", -110))
        fair_over, fair_under = remove_vig(over_prob_book, under_prob_book)

        run_dist = simulation_result.get("run_distribution")
        if run_dist:
            model_over_prob = _dist_over_prob(run_dist, book_total)
        else:
            sigma = simulation_result.get("run_total_std") or 3.0
            model_over_prob = _normal_over_prob(model_total, book_total, sigma=sigma)
        model_under_prob = 1.0 - model_over_prob

        total_edge_over = model_over_prob - fair_over
        total_edge_under = model_under_prob - fair_under
        result["total_edge_over"] = round(total_edge_over, 4)
        result["total_edge_under"] = round(total_edge_under, 4)

        # O/U pick: whichever side clears the LEAN threshold with the larger edge
        over_e = result["total_edge_over"] or 0.0
        under_e = result["total_edge_under"] or 0.0
        best_ou_edge = max(over_e, under_e)
        if best_ou_edge >= LEAN_EDGE_THRESHOLD:
            pick = "Over" if over_e >= under_e else "Under"
            result["ou_pick"] = pick
            result["ou_edge"] = round(best_ou_edge, 4)
            result["ou_rec"] = "BET" if best_ou_edge >= BET_EDGE_THRESHOLD else "LEAN"

    # Determine best bets across all markets
    bets = []
    edges_to_check = [
        ("home_edge", "home_team", "moneyline"),
        ("away_edge", "away_team", "moneyline"),
        ("total_edge_over", None, "over"),
        ("total_edge_under", None, "under"),
    ]
    for edge_key, side_key, market in edges_to_check:
        edge_val = result.get(edge_key)
        if edge_val is not None and edge_val >= LEAN_EDGE_THRESHOLD:
            bets.append({"edge_key": edge_key, "edge": edge_val, "market": market, "side_key": side_key})

    if bets:
        best = max(bets, key=lambda x: x["edge"])
        result["best_bets"] = bets
        if best["edge"] >= BET_EDGE_THRESHOLD:
            result["recommendation"] = "BET"
        else:
            result["recommendation"] = "LEAN"

    return result


def _dist_over_prob(run_distribution: dict, book_total: float) -> float:
    """
    P(total > book_total) from the simulated integer run-total distribution.
    Uses the actual outcome counts rather than a parametric approximation.
    Handles half-point lines correctly (e.g., 8.5 — no push possible).
    JSON storage converts integer keys to strings, so we cast to int before comparing.
    """
    total_sims = sum(run_distribution.values())
    if total_sims == 0:
        return 0.5
    over = sum(cnt for total, cnt in run_distribution.items() if int(total) > book_total)
    return over / total_sims


def _normal_over_prob(model_avg: float, book_line: float, sigma: float) -> float:
    """Fallback: P(total > book_line) assuming Normal(model_avg, sigma)."""
    z = (book_line - model_avg) / sigma
    return 1.0 - _standard_normal_cdf(z)


def _standard_normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))
