"""
Convert model win probabilities to fair odds and compute edge vs sportsbook lines.
"""

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

    result = {
        "model_home_prob": model_home_prob,
        "model_away_prob": model_away_prob,
        "model_fair_home_ml": prob_to_american(model_home_prob),
        "model_fair_away_ml": prob_to_american(model_away_prob),
        "model_total": model_total,
        "home_edge": None,
        "away_edge": None,
        "total_edge_over": None,
        "total_edge_under": None,
        "book_home_ml": None,
        "book_away_ml": None,
        "book_total": None,
        "book_over_ml": None,
        "book_under_ml": None,
        "recommendation": "PASS",
        "best_bets": [],
        "has_odds": False,
    }

    if not odds:
        return result

    result["has_odds"] = True
    result["book_home_ml"] = odds.get("home_ml")
    result["book_away_ml"] = odds.get("away_ml")
    result["book_total"] = odds.get("total")
    result["book_over_ml"] = odds.get("over_ml", -110)
    result["book_under_ml"] = odds.get("under_ml", -110)

    # Moneyline edge
    home_ml = odds.get("home_ml")
    away_ml = odds.get("away_ml")
    if home_ml is not None and away_ml is not None:
        raw_home_prob = american_to_prob(home_ml)
        raw_away_prob = american_to_prob(away_ml)
        fair_home, fair_away = remove_vig(raw_home_prob, raw_away_prob)

        home_edge = model_home_prob - fair_home
        away_edge = model_away_prob - fair_away
        result["home_edge"] = round(home_edge, 4)
        result["away_edge"] = round(away_edge, 4)

    # Totals edge
    book_total = odds.get("total")
    if book_total is not None:
        over_prob_book = american_to_prob(odds.get("over_ml", -110))
        under_prob_book = american_to_prob(odds.get("under_ml", -110))
        fair_over, fair_under = remove_vig(over_prob_book, under_prob_book)

        # Estimate model's over/under probability using normal distribution
        # around avg_total with stddev ~3 runs
        model_over_prob = _normal_over_prob(model_total, book_total, sigma=3.0)
        model_under_prob = 1.0 - model_over_prob

        total_edge_over = model_over_prob - fair_over
        total_edge_under = model_under_prob - fair_under
        result["total_edge_over"] = round(total_edge_over, 4)
        result["total_edge_under"] = round(total_edge_under, 4)

    # Determine best bets
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


def _normal_over_prob(model_avg: float, book_line: float, sigma: float) -> float:
    """
    Estimate P(total > book_line) assuming total runs ~ Normal(model_avg, sigma).
    """
    import math
    z = (book_line - model_avg) / sigma
    return 1.0 - _standard_normal_cdf(z)


def _standard_normal_cdf(z: float) -> float:
    import math
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))
