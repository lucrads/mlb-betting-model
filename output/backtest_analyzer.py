"""
Analyze backtest_results.json and return structured stats for the report tab.
"""

from datetime import datetime


_CONFIDENCE_TIERS = [
    ("50–55%", 0.50, 0.55),
    ("55–60%", 0.55, 0.60),
    ("60–65%", 0.60, 0.65),
    ("65–70%", 0.65, 0.70),
    ("70%+",   0.70, 1.01),
]

_EDGE_BUCKETS = [
    ("2–5%",   0.02, 0.05),
    ("5–10%",  0.05, 0.10),
    ("10–20%", 0.10, 0.20),
    ("20%+",   0.20, 1.00),
]


def analyze_backtest(results: list[dict]) -> dict | None:
    """
    Compute hit-rate statistics from a list of backtest result records.
    Returns None if results is empty.
    """
    if not results:
        return None

    total = len(results)
    ml_correct = sum(1 for r in results if r.get("model_correct_ml"))
    ml_rate = ml_correct / total

    dates = sorted(r["date"] for r in results)

    # --- By model confidence tier ---
    by_confidence = {}
    for label, lo, hi in _CONFIDENCE_TIERS:
        subset = [
            r for r in results
            if lo <= max(r["home_win_pct"], r["away_win_pct"]) < hi
        ]
        if subset:
            hits = sum(1 for r in subset if r.get("model_correct_ml"))
            by_confidence[label] = {
                "games": len(subset),
                "hits": hits,
                "rate": hits / len(subset),
            }

    # --- By confidence classification (odds-independent) ---
    by_recommendation = {}
    # Prefer confidence_rec (always populated); fall back to recommendation for
    # records that pre-date this field or were run with live odds.
    for label, rec_key in [("HIGH (70%+)", "HIGH"), ("MED (60–70%)", "MED"),
                           ("LEAN (55–60%)", "LEAN"), ("PASS (<55%)", "PASS")]:
        subset = [r for r in results if r.get("confidence_rec") == rec_key]
        if not subset:
            # legacy records without confidence_rec — fall back to recommendation
            subset = [r for r in results if r.get("recommendation") == rec_key]
        if subset:
            hits = sum(1 for r in subset if r.get("model_correct_ml"))
            by_recommendation[label] = {
                "games": len(subset),
                "hits": hits,
                "rate": hits / len(subset),
            }

    # --- By edge range (bet-level, only when odds were available) ---
    by_edge = {}
    all_bets = [
        bet
        for r in results
        for bet in r.get("bet_results", [])
        if bet.get("won") is not None
    ]
    for label, lo, hi in _EDGE_BUCKETS:
        subset = [b for b in all_bets if lo <= b["edge"] < hi]
        if subset:
            hits = sum(1 for b in subset if b["won"])
            by_edge[label] = {
                "bets": len(subset),
                "hits": hits,
                "rate": hits / len(subset),
            }

    # --- O/U accuracy (only when book total known) ---
    ou_games = [r for r in results if r.get("book_total") is not None]
    ou_stats = None
    if ou_games:
        ou_hits = sum(
            1 for r in ou_games
            if (r["model_total"] > r["book_total"] and r["actual_total"] > r["book_total"])
            or (r["model_total"] < r["book_total"] and r["actual_total"] < r["book_total"])
            or (r["model_total"] == r["book_total"])  # push — count as hit to be conservative
        )
        ou_stats = {
            "games": len(ou_games),
            "hits": ou_hits,
            "rate": ou_hits / len(ou_games),
        }

    return {
        "total_games": total,
        "ml_correct": ml_correct,
        "ml_rate": ml_rate,
        "date_range": f"{_fmt_date(dates[0])} – {_fmt_date(dates[-1])}",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "by_confidence": by_confidence,
        "by_recommendation": by_recommendation,
        "by_edge": by_edge,
        "ou_stats": ou_stats,
    }


def _fmt_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%b %-d")
    except Exception:
        return iso
