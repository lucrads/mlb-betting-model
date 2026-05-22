"""Generate styled HTML report from simulation results."""

import os
import logging
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from output.backtest_analyzer import analyze_backtest

logger = logging.getLogger(__name__)


def _make_env() -> Environment:
    import json
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=True)
    env.filters["pct"] = lambda v: f"{v * 100:.1f}%" if v is not None else "N/A"
    env.filters["american"] = _format_american
    env.filters["edge_class"] = _edge_css_class
    env.filters["abs"] = abs
    env.filters["tojson"] = lambda v: json.dumps(v, default=str)
    return env


def generate_report(date_str: str, game_results: list[dict], output_dir: str = ".") -> str:
    """
    Legacy entry point — called by main.py directly.
    Loads backtest records from the store (migrating from legacy file if needed).
    """
    import store
    bt_records = store.read_backtest_records()
    bt_stats = analyze_backtest(bt_records) if bt_records else None

    return generate_report_from_store(
        date_str=date_str,
        game_results=game_results,
        bt_records=bt_records,
        bt_stats=bt_stats,
        output_dir=output_dir,
    )


def generate_report_from_store(
    date_str: str,
    game_results: list[dict],
    bt_records: list,
    bt_stats: dict | None,
    output_dir: str = ".",
) -> str:
    """
    Render and write the HTML report.

    game_results: list of dicts, each containing:
      - game: enriched game dict
      - simulation: monte_carlo result dict
      - edge: edge_calc result dict
      - details: matchup_details dict

    bt_records: raw list of all backtest game records (for scrollable log)
    bt_stats: pre-computed summary stats from analyze_backtest()

    Returns: path to the written HTML file.
    """
    env = _make_env()
    template = env.get_template("report.html")

    # Sort backtest log newest-first for the scrollable view
    bt_log = sorted(bt_records or [], key=lambda r: r.get("date", ""), reverse=True)

    context = {
        "date": date_str,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "games": game_results,
        "total_games": len(game_results),
        "bet_count": sum(1 for g in game_results if g["edge"]["recommendation"] == "BET"),
        "lean_count": sum(1 for g in game_results if g["edge"]["recommendation"] == "LEAN"),
        "bt": bt_stats,
        "bt_log": bt_log,
    }

    html = template.render(**context)
    filename = os.path.join(output_dir, f"report_{date_str}.html")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Report written to %s", filename)
    return filename


def _format_american(odds: int | None) -> str:
    if odds is None:
        return "N/A"
    return f"+{odds}" if odds > 0 else str(odds)


def _edge_css_class(edge: float | None) -> str:
    if edge is None:
        return "neutral"
    if edge >= 0.05:
        return "strong-edge"
    if edge >= 0.02:
        return "lean-edge"
    if edge <= -0.03:
        return "negative-edge"
    return "neutral"
