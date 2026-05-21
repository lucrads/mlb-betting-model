"""Generate styled HTML report from simulation results."""

import os
import logging
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


def generate_report(date_str: str, game_results: list[dict], output_dir: str = ".") -> str:
    """
    Render and write the HTML report.

    game_results: list of dicts, each containing:
      - game: original game dict
      - simulation: monte_carlo result dict
      - edge: edge_calc result dict

    Returns: path to the written HTML file.
    """
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=True)
    env.filters["pct"] = lambda v: f"{v * 100:.1f}%" if v is not None else "N/A"
    env.filters["american"] = _format_american
    env.filters["edge_class"] = _edge_css_class
    env.filters["abs"] = abs

    template = env.get_template("report.html")

    context = {
        "date": date_str,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "games": game_results,
        "total_games": len(game_results),
        "bet_count": sum(1 for g in game_results if g["edge"]["recommendation"] == "BET"),
        "lean_count": sum(1 for g in game_results if g["edge"]["recommendation"] == "LEAN"),
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
