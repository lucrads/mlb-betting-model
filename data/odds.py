"""
Odds fetcher — two sources:
  1. ESPN public scoreboard API (free, no key) — default
  2. The Odds API (pass api_key) — enables full line shopping across all books

Both return the same dict format keyed by (home_team, away_team).
"""

import logging
import requests

logger = logging.getLogger(__name__)

_ESPN_URL  = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
_ODDS_URL  = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

_TEAM_ALIASES = {
    "Arizona Diamondbacks":  ["Arizona Diamondbacks",  "ARI", "Diamondbacks"],
    "Atlanta Braves":        ["Atlanta Braves",        "ATL", "Braves"],
    "Baltimore Orioles":     ["Baltimore Orioles",     "BAL", "Orioles"],
    "Boston Red Sox":        ["Boston Red Sox",        "BOS", "Red Sox"],
    "Chicago White Sox":     ["Chicago White Sox",     "CWS", "Chi White Sox", "White Sox"],
    "Chicago Cubs":          ["Chicago Cubs",          "CHC", "Chi Cubs", "Cubs"],
    "Cincinnati Reds":       ["Cincinnati Reds",       "CIN", "Reds"],
    "Cleveland Guardians":   ["Cleveland Guardians",   "CLE", "Guardians"],
    "Colorado Rockies":      ["Colorado Rockies",      "COL", "Rockies"],
    "Detroit Tigers":        ["Detroit Tigers",        "DET", "Tigers"],
    "Houston Astros":        ["Houston Astros",        "HOU", "Astros"],
    "Kansas City Royals":    ["Kansas City Royals",    "KC",  "Royals"],
    "Los Angeles Angels":    ["Los Angeles Angels",    "LAA", "Angels"],
    "Los Angeles Dodgers":   ["Los Angeles Dodgers",   "LAD", "Dodgers"],
    "Miami Marlins":         ["Miami Marlins",         "MIA", "Marlins"],
    "Milwaukee Brewers":     ["Milwaukee Brewers",     "MIL", "Brewers"],
    "Minnesota Twins":       ["Minnesota Twins",       "MIN", "Twins"],
    "New York Yankees":      ["New York Yankees",      "NYY", "Yankees"],
    "New York Mets":         ["New York Mets",         "NYM", "Mets"],
    "Oakland Athletics":     ["Oakland Athletics",     "OAK", "Athletics"],
    "Philadelphia Phillies": ["Philadelphia Phillies", "PHI", "Phillies"],
    "Pittsburgh Pirates":    ["Pittsburgh Pirates",    "PIT", "Pirates"],
    "San Diego Padres":      ["San Diego Padres",      "SD",  "Padres"],
    "San Francisco Giants":  ["San Francisco Giants",  "SF",  "Giants"],
    "Seattle Mariners":      ["Seattle Mariners",      "SEA", "Mariners"],
    "St. Louis Cardinals":   ["St. Louis Cardinals",   "STL", "Cardinals"],
    "Tampa Bay Rays":        ["Tampa Bay Rays",        "TB",  "Rays"],
    "Texas Rangers":         ["Texas Rangers",         "TEX", "Rangers"],
    "Toronto Blue Jays":     ["Toronto Blue Jays",     "TOR", "Blue Jays"],
    "Washington Nationals":  ["Washington Nationals",  "WSH", "Washington", "Nationals"],
}

_ALIAS_REVERSE: dict[str, str] = {}
for _canonical, _aliases in _TEAM_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_REVERSE[_alias.lower()] = _canonical


def _normalize_team(name: str) -> str:
    return _ALIAS_REVERSE.get(name.strip().lower(), name.strip())


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_odds(api_key: str | None = None, date_str: str | None = None) -> dict:
    """
    Fetch MLB moneyline + totals odds.

    - If api_key is provided: uses The Odds API (full line shopping, all books).
    - Otherwise: uses ESPN's free public scoreboard API (ESPN BET lines).

    Returns dict keyed by (home_team, away_team). Each value has:
      home_ml, away_ml         — primary book moneyline
      best_home_ml             — best available moneyline for home (same as home_ml for ESPN)
      best_away_ml             — best available moneyline for away
      best_home_book           — book name for best home ML
      best_away_book           — book name for best away ML
      total, over_ml, under_ml — O/U line
      bookmaker                — source book name
      all_ml                   — {book: {home, away}} mapping
    """
    if api_key:
        return _fetch_odds_api(api_key)
    return _fetch_espn_odds(date_str)


# ---------------------------------------------------------------------------
# ESPN scoreboard odds (free, no key)
# ---------------------------------------------------------------------------

def _fetch_espn_odds(date_str: str | None = None) -> dict:
    """Pull moneylines + totals from ESPN's public scoreboard API (DraftKings lines)."""
    params = {}
    if date_str:
        params["dates"] = date_str.replace("-", "")   # YYYYMMDD

    try:
        resp = requests.get(_ESPN_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("ESPN odds request failed: %s", e)
        return {}

    games_odds = {}
    for event in resp.json().get("events", []):
        competition = event.get("competitions", [{}])[0]

        # Resolve home / away team names
        home = away = None
        for comp in competition.get("competitors", []):
            canonical = _normalize_team(comp.get("team", {}).get("displayName", ""))
            if comp.get("homeAway") == "home":
                home = canonical
            else:
                away = canonical

        if not home or not away:
            continue

        odds_list = competition.get("odds", [])
        if not odds_list:
            continue

        o = odds_list[0]
        provider = o.get("provider", {}).get("name", "ESPN BET")

        # Moneyline lives at o["moneyline"]["home/away"]["close"]["odds"] (string)
        ml_block = o.get("moneyline", {})
        home_ml = _safe_int(ml_block.get("home", {}).get("close", {}).get("odds"))
        away_ml = _safe_int(ml_block.get("away", {}).get("close", {}).get("odds"))

        # Total lives at o["overUnder"]; over/under odds at o["total"]["over/under"]["close"]["odds"]
        total    = o.get("overUnder")
        tot_block = o.get("total", {})
        over_ml  = _safe_int(tot_block.get("over",  {}).get("close", {}).get("odds"), -110)
        under_ml = _safe_int(tot_block.get("under", {}).get("close", {}).get("odds"), -110)

        if home_ml is None or away_ml is None:
            logger.debug("ESPN: no ML for %s @ %s — skipping", away, home)
            continue

        entry = {
            "home_ml":        home_ml,
            "away_ml":        away_ml,
            "best_home_ml":   home_ml,
            "best_away_ml":   away_ml,
            "best_home_book": provider,
            "best_away_book": provider,
            "bookmaker":      provider,
            "all_ml":         {provider: {"home": home_ml, "away": away_ml}},
            "total":          float(total) if total is not None else None,
            "over_ml":        over_ml,
            "under_ml":       under_ml,
        }
        games_odds[(home, away)] = entry
        logger.debug(
            "ESPN odds: %s @ %s — home %s / away %s | O/U %.1f (%s)",
            away, home, home_ml, away_ml, total or 0, provider,
        )

    logger.info("Loaded ESPN odds for %d games.", len(games_odds))
    return games_odds


def _safe_int(val, default=None):
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# The Odds API (key required, full line shopping)
# ---------------------------------------------------------------------------

def _fetch_odds_api(api_key: str) -> dict:
    """Fetch from The Odds API — returns best line across all sportsbooks."""
    try:
        resp = requests.get(
            _ODDS_URL,
            params={
                "apiKey":      api_key,
                "regions":     "us",
                "markets":     "h2h,totals",
                "oddsFormat":  "american",
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
                "Odds API: %s @ %s — Best ML: home %s (%s) / away %s (%s) | Total %.1f",
                away, home,
                book_data.get("best_home_ml"), book_data.get("best_home_book"),
                book_data.get("best_away_ml"), book_data.get("best_away_book"),
                book_data.get("total") or 0,
            )

    logger.info("Loaded Odds API lines for %d games.", len(games_odds))
    return games_odds


def _parse_all_bookmakers(bookmakers: list, home_team: str, away_team: str) -> dict | None:
    preferred_order = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet", "barstool"]

    all_ml: dict[str, dict] = {}
    totals_by_book: dict[str, dict] = {}

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
                over  = outcomes.get("Over")
                under = outcomes.get("Under")
                total_pt = market["outcomes"][0].get("point") if market.get("outcomes") else None
                if total_pt:
                    totals_by_book[title] = {
                        "total":    float(total_pt),
                        "over_ml":  int(over)  if over  else -110,
                        "under_ml": int(under) if under else -110,
                    }

    if not all_ml:
        return None

    best_home_ml   = max(v["home"] for v in all_ml.values())
    best_away_ml   = max(v["away"] for v in all_ml.values())
    best_home_book = next(t for t, v in all_ml.items() if v["home"] == best_home_ml)
    best_away_book = next(t for t, v in all_ml.items() if v["away"] == best_away_ml)

    pref_title = None
    for pref in preferred_order:
        match = next((t for t in all_ml if pref in t.lower()), None)
        if match:
            pref_title = match
            break
    if pref_title is None:
        pref_title = next(iter(all_ml))

    pref_ml = all_ml[pref_title]

    pref_totals = None
    for pref in preferred_order:
        match = next((t for t in totals_by_book if pref in t.lower()), None)
        if match:
            pref_totals = totals_by_book[match]
            break
    if pref_totals is None and totals_by_book:
        pref_totals = next(iter(totals_by_book.values()))

    return {
        "home_ml":        pref_ml["home"],
        "away_ml":        pref_ml["away"],
        "best_home_ml":   best_home_ml,
        "best_away_ml":   best_away_ml,
        "best_home_book": best_home_book,
        "best_away_book": best_away_book,
        "bookmaker":      pref_title,
        "all_ml":         all_ml,
        "total":          pref_totals["total"]    if pref_totals else None,
        "over_ml":        pref_totals["over_ml"]  if pref_totals else -110,
        "under_ml":       pref_totals["under_ml"] if pref_totals else -110,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
    api_words   = set(api_name.lower().split())
    query_words = set(query.lower().split())
    return bool(api_words & query_words)
