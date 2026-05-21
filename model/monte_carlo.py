"""Run 1000 simulations per game and aggregate results."""

import logging
import numpy as np
from collections import Counter
from model.simulator import simulate_game
import config

logger = logging.getLogger(__name__)


def run_simulations(game: dict) -> dict:
    """
    Run NUM_SIMULATIONS simulations for a single game.

    Returns aggregated stats:
      home_win_pct, away_win_pct, tie_pct,
      avg_home_runs, avg_away_runs, avg_total_runs,
      run_distribution (Counter of total run outcomes),
      score_distribution (Counter of (home_runs, away_runs) pairs)
    """
    n_sims = config.NUM_SIMULATIONS
    logger.info(
        "Simulating %s @ %s (%d simulations)...",
        game["away_team"], game["home_team"], n_sims
    )

    home_wins = 0
    away_wins = 0
    ties = 0
    home_run_totals = []
    away_run_totals = []
    score_dist: Counter = Counter()

    for _ in range(n_sims):
        h, a = simulate_game(game, {}, {})
        home_run_totals.append(h)
        away_run_totals.append(a)
        score_dist[(h, a)] += 1
        if h > a:
            home_wins += 1
        elif a > h:
            away_wins += 1
        else:
            ties += 1

    n = n_sims
    avg_home = float(np.mean(home_run_totals))
    avg_away = float(np.mean(away_run_totals))
    avg_total = avg_home + avg_away

    run_dist: Counter = Counter()
    for (h, a), cnt in score_dist.items():
        run_dist[h + a] += cnt

    result = {
        "home_win_pct": round(home_wins / n, 4),
        "away_win_pct": round(away_wins / n, 4),
        "tie_pct": round(ties / n, 4),
        "avg_home_runs": round(avg_home, 2),
        "avg_away_runs": round(avg_away, 2),
        "avg_total_runs": round(avg_total, 2),
        "run_distribution": dict(run_dist),
        "most_common_scores": score_dist.most_common(5),
    }

    logger.info(
        "Result: %s wins %.1f%% | %s wins %.1f%% | Avg total: %.1f",
        game["home_team"], result["home_win_pct"] * 100,
        game["away_team"], result["away_win_pct"] * 100,
        result["avg_total_runs"],
    )

    return result
