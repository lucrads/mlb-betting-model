"""The Odds API — fetch live MLB moneyline + totals."""

import logging
import requests

logger = logging.getLogger(__name__)

_ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

# Common team name aliases to help match sportsbook names to MLB API names
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
    Fetch live MLB odds and return a dict keyed by (home_team, away_team) tuples
    (normalized to MLB Stats API team names).

    Value format:
    {
        "home_ml": int,   # American odds, e.g. -130
        "away_ml": int,
        "total": float,   # e.g. 8.5
        "over_ml": int,
        "under_ml": int,
        "bookmaker": str,
    }
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

        # Use first available bookmaker that has both markets
        book_data = _parse_bookmaker(bookmakers)
        if book_data:
            games_odds[(home, away)] = book_data
            logger.debug("Odds loaded: %s @ %s — ML %s/%s | Total %.1f",
                         away, home, book_data.get("away_ml"), book_data.get("home_ml"), book_data.get("total", 0))

    logger.info("Loaded odds for %d games.", len(games_odds))
    return games_odds


def _parse_bookmaker(bookmakers: list) -> dict | None:
    preferred = ["draftkings", "fanduel", "betmgm", "caesars"]
    ordered = sorted(bookmakers, key=lambda b: (
        preferred.index(b["key"]) if b["key"] in preferred else 99
    ))

    result = {}
    for book in ordered:
        for market in book.get("markets", []):
            key = market.get("key")
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            if key == "h2h":
                if "home_ml" not in result:
                    home_key = list(outcomes.keys())[0] if outcomes else None
                    if home_key:
                        home_team_odds = None
                        away_team_odds = None
                        for team_name, price in outcomes.items():
                            # We'll store as-is and re-map later
                            pass
                        # Store raw for caller to map
                        result["_h2h_raw"] = outcomes
                        result["bookmaker"] = book.get("title", book["key"])
            elif key == "totals":
                if "total" not in result:
                    over = outcomes.get("Over")
                    under = outcomes.get("Under")
                    total_line = market["outcomes"][0].get("point") if market.get("outcomes") else None
                    if total_line:
                        result["total"] = float(total_line)
                        result["over_ml"] = int(over) if over else -110
                        result["under_ml"] = int(under) if under else -110

        if "_h2h_raw" in result:
            break

    if "_h2h_raw" not in result:
        return None

    return result


def resolve_game_odds(odds_dict: dict, home_team: str, away_team: str) -> dict | None:
    """Look up odds for a game, trying fuzzy matching on team names."""
    # Direct lookup
    key = (home_team, away_team)
    if key in odds_dict:
        entry = dict(odds_dict[key])
        return _assign_ml(entry, home_team, away_team)

    # Partial name match
    for (h, a), entry in odds_dict.items():
        if _team_matches(h, home_team) and _team_matches(a, away_team):
            result = dict(entry)
            return _assign_ml(result, home_team, away_team)

    return None


def _team_matches(api_name: str, query: str) -> bool:
    api_words = set(api_name.lower().split())
    query_words = set(query.lower().split())
    return bool(api_words & query_words)


def _assign_ml(entry: dict, home_team: str, away_team: str) -> dict:
    h2h = entry.pop("_h2h_raw", {})
    home_ml = None
    away_ml = None
    for name, price in h2h.items():
        if _team_matches(name, home_team):
            home_ml = int(price)
        elif _team_matches(name, away_team):
            away_ml = int(price)
    entry["home_ml"] = home_ml
    entry["away_ml"] = away_ml
    return entry
