"""The Odds API — fetch live MLB moneyline + totals.
Returns the best available line across all bookmakers (line shopping).
"""

import logging
import requests

logger = logging.getLogger(__name__)

_ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

_TEAM_ALIASES = {
    "Arizona Diamondbacks": ["Arizona Diamondbacks", "ARI"],
    "Atlanta Braves": ["Atlanta Braves", "ATL"],
    "Baltimore Orioles": ["Baltimore Orioles", "BAL"],
    "Boston Red Sox": ["Boston Red Sox", "BOS"],
    "Chicago White Sox": ["Chicago White Sox", "CWS", "Chi White Sox"],
    "Chicago Cubs": ["Chicago Cubs", "CHC", "Chi Cubs"],
    "Cincinnati Reds": ["Cincinnati Reds", "CIN"],
    "Cleveland Guardians": ["Cleveland Guardians", "CLE"],
    "Colorado Rockies": ["Colorado Rockies", "COL"],
    "Detroit Tigers": ["Detroit Tigers", "DET"],
    "Houston Astros": ["Houston Astros", "HOU"],
    "Kansas City Royals": ["Kansas City Royals", "KC"],
    "Los Angeles Angels": ["Los Angeles Angels", "LAA"],
    "Los Angeles Dodgers": ["Los Angeles Dodgers", "LAD"],
    "Miami Marlins": ["Miami Marlins", "MIA"],
    "Milwaukee Brewers": ["Milwaukee Brewers", "MIL"],
    "Minnesota Twins": ["Minnesota Twins", "MIN"],
    "New York Yankees": ["New York Yankees", "NYY"],
    "New York Mets": ["New York Mets", "NYM"],
    "Oakland Athletics": ["Oakland Athletics", "OAK", "Athletics"],
    "Philadelphia Phillies": ["Philadelphia Phillies", "PHI"],
    "Pittsburgh Pirates": ["Pittsburgh Pirates", "PIT"],
    "San Diego Padres": ["San Diego Padres", "SD"],
    "San Francisco Giants": ["San Francisco Giants", "SF"],
    "Seattle Mariners": ["Seattle Mariners", "SEA"],
    "St. Louis Cardinals": ["St. Louis Cardinals", "STL"],
    "Tampa Bay Rays": ["Tampa Bay Rays", "TB"],
    "Texas Rangers": ["Texas Rangers", "TEX"],
    "Toronto Blue Jays": ["Toronto Blue Jays", "TOR"],
    "Washington Nationals": ["Washington Nationals", "WSH", "Washington"],
}

_ALIAS_REVERSE: dict[str, str] = {}
for canonical, aliases in _TEAM_ALIASES.items():
    for alias in aliases:
        _ALIAS_REVERSE[alias.lower()] = canonical


def _normalize_team(name: str) -> str:
    return _ALIAS_REVERSE.get(name.lower(), name)


def fetch_odds(api_key: str) -> dict:
    """
    Fetch live MLB odds and return a dict keyed by (home_team, away_team).
    Each value contains both the single-book line and the best-available (line-shopped) line.

    Value keys:
      home_ml, away_ml         — from preferred book (DK > FD > MGM > Caesars)
      best_home_ml             — highest ML available across ALL books for home team
      best_away_ml             — highest ML available across ALL books for away team
      best_home_book           — name of book offering best home ML
      best_away_book           — name of book offering best away ML
      total, over_ml, under_ml — from preferred book with totals data
      bookmaker                — preferred book name used for totals
      all_ml                   — {book_title: {home: int, away: int}} for all books
    """
    if not api_key:
        logger.warning("No Odds API key provided — skipping odds fetch.")
        return {}

    try:
        resp = requests.get(
            _ODDS_URL,
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "h2h,totals",
                "oddsFormat": "american",
            },
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Odds API request failed: %s", e)
        return {}

    games_odds = {}
    for game in resp.json():
        home = _normalize_team(game.get("home_team", ""))
        away = _normalize_team(game.get("away_team", ""))
        if not home or not away:
            continue

        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        book_data = _parse_all_bookmakers(bookmakers, home, away)
        if book_data:
            games_odds[(home, away)] = book_data
            logger.debug(
                "Odds: %s @ %s — Best ML: home %s (%s) / away %s (%s) | Total %.1f",
                away, home,
                book_data.get("best_home_ml"), book_data.get("best_home_book"),
                book_data.get("best_away_ml"), book_data.get("best_away_book"),
                book_data.get("total") or 0,
            )

    logger.info("Loaded odds for %d games.", len(games_odds))
    return games_odds


def _parse_all_bookmakers(bookmakers: list, home_team: str, away_team: str) -> dict | None:
    """Collect lines from every book and identify the best available per side."""
    preferred_order = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet", "barstool"]

    all_ml: dict[str, dict] = {}   # book_title → {home: int, away: int}
    totals_by_book: dict[str, dict] = {}  # book_title → {total, over_ml, under_ml}

    for book in bookmakers:
        title = book.get("title", book["key"])
        for market in book.get("markets", []):
            key = market.get("key")
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}

            if key == "h2h":
                home_ml = away_ml = None
                for name, price in outcomes.items():
                    if _team_matches(name, home_team):
                        home_ml = int(price)
                    elif _team_matches(name, away_team):
                        away_ml = int(price)
                if home_ml is not None and away_ml is not None:
                    all_ml[title] = {"home": home_ml, "away": away_ml}

            elif key == "totals":
                over = outcomes.get("Over")
                under = outcomes.get("Under")
                total_pt = market["outcomes"][0].get("point") if market.get("outcomes") else None
                if total_pt:
                    totals_by_book[title] = {
                        "total": float(total_pt),
                        "over_ml": int(over) if over else -110,
                        "under_ml": int(under) if under else -110,
                    }

    if not all_ml:
        return None

    # Best available: highest ML for each side across all books
    best_home_ml = max(v["home"] for v in all_ml.values())
    best_away_ml = max(v["away"] for v in all_ml.values())
    best_home_book = next(t for t, v in all_ml.items() if v["home"] == best_home_ml)
    best_away_book = next(t for t, v in all_ml.items() if v["away"] == best_away_ml)

    # Preferred book for moneyline (also sets home_ml/away_ml for backwards compat)
    pref_title = None
    for pref in preferred_order:
        match = next((t for t in all_ml if pref in t.lower()), None)
        if match:
            pref_title = match
            break
    if pref_title is None:
        pref_title = next(iter(all_ml))

    pref_ml = all_ml[pref_title]

    # Preferred book for totals
    pref_totals = None
    for pref in preferred_order:
        match = next((t for t in totals_by_book if pref in t.lower()), None)
        if match:
            pref_totals = totals_by_book[match]
            break
    if pref_totals is None and totals_by_book:
        pref_totals = next(iter(totals_by_book.values()))

    result = {
        "home_ml":         pref_ml["home"],
        "away_ml":         pref_ml["away"],
        "best_home_ml":    best_home_ml,
        "best_away_ml":    best_away_ml,
        "best_home_book":  best_home_book,
        "best_away_book":  best_away_book,
        "bookmaker":       pref_title,
        "all_ml":          all_ml,
        "total":           pref_totals["total"] if pref_totals else None,
        "over_ml":         pref_totals["over_ml"] if pref_totals else -110,
        "under_ml":        pref_totals["under_ml"] if pref_totals else -110,
    }
    return result


def resolve_game_odds(odds_dict: dict, home_team: str, away_team: str) -> dict | None:
    """Look up odds for a game, trying fuzzy matching on team names."""
    key = (home_team, away_team)
    if key in odds_dict:
        return dict(odds_dict[key])

    for (h, a), entry in odds_dict.items():
        if _team_matches(h, home_team) and _team_matches(a, away_team):
            return dict(entry)

    return None


def _team_matches(api_name: str, query: str) -> bool:
    api_words = set(api_name.lower().split())
    query_words = set(query.lower().split())
    return bool(api_words & query_words)
