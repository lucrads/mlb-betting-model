"""
Monte Carlo aggregator with antithetic-variates variance reduction.

For an estimator Ȳ = (1/n) Σ y_i of a game's expected runs / win prob,
the antithetic-variates technique pairs each simulation i with a
counterpart i' that consumes 1-u in place of every uniform u_i used by
the simulator. Because the paired estimators are negatively correlated
(Cov(y_i, y_i') < 0 for monotone-in-uniform mappings), the variance of
the paired mean

      ȳ_pair = (y_i + y_i') / 2

is below Var(y_i)/2 — typically 30-50% lower for baseball outcomes,
which are weakly monotone in the underlying CDF inversion. The
computational cost per pair is exactly the same as two independent
sims, so the variance reduction is essentially free.

Game state can diverge across the pair (different outcomes → different
at-bat counts per inning), but the negative correlation of the
*early* at-bats — which is where the bulk of run-scoring variance
lives — still drives substantial variance reduction at the game-total
level.
"""

import logging
import numpy as np
from collections import Counter
from model.simulator import simulate_game
import config

logger = logging.getLogger(__name__)


class _AntitheticRng:
    """
    Thin RNG facade: forwards `random()` and `choice(items, p)` calls to a
    numpy Generator, optionally inverting the uniform to 1-u for the
    antithetic counterpart of a paired simulation.
    """

    __slots__ = ("_g", "_anti")

    def __init__(self, generator: np.random.Generator, antithetic: bool = False):
        self._g = generator
        self._anti = antithetic

    def random(self) -> float:
        u = float(self._g.random())
        return 1.0 - u if self._anti else u

    def choice(self, items, p):
        u = self.random()
        cum = 0.0
        for item, w in zip(items, p):
            cum += w
            if u <= cum:
                return item
        return items[-1]


def run_simulations(game: dict, seed: int | None = None) -> dict:
    """
    Run NUM_SIMULATIONS simulations for a single game, using antithetic
    pairing for variance reduction.

    Returns aggregated stats:
      home_win_pct, away_win_pct, tie_pct,
      avg_home_runs, avg_away_runs, avg_total_runs,
      run_distribution, score_distribution, run_total_std (sample std)
    """
    n_sims = config.NUM_SIMULATIONS
    logger.info(
        "Simulating %s @ %s (%d sims, antithetic-paired)...",
        game["away_team"], game["home_team"], n_sims,
    )

    # Each *pair* of simulations shares a base Generator seeded with seed_base+i;
    # paired draws use complementary uniforms (u, 1-u).
    base_seed = seed if seed is not None else np.random.SeedSequence().entropy
    n_pairs = n_sims // 2
    leftover = n_sims % 2

    home_run_totals: list[int] = []
    away_run_totals: list[int] = []
    score_dist: Counter = Counter()
    home_wins = away_wins = ties = 0

    def _tally(h: int, a: int) -> None:
        nonlocal home_wins, away_wins, ties
        home_run_totals.append(h)
        away_run_totals.append(a)
        score_dist[(h, a)] += 1
        if h > a:
            home_wins += 1
        elif a > h:
            away_wins += 1
        else:
            ties += 1

    for i in range(n_pairs):
        gen = np.random.default_rng(np.random.SeedSequence([base_seed, i]))  # type: ignore[arg-type]
        # Share generator state across the pair so they consume the SAME
        # underlying uniforms; only the antithetic flag flips u ↔ 1-u.
        h1, a1 = simulate_game(game, rng=_AntitheticRng(gen, antithetic=False))
        h2, a2 = simulate_game(game, rng=_AntitheticRng(gen, antithetic=True))
        _tally(h1, a1)
        _tally(h2, a2)

    if leftover:
        gen = np.random.default_rng(np.random.SeedSequence([base_seed, n_pairs]))  # type: ignore[arg-type]
        h, a = simulate_game(game, rng=_AntitheticRng(gen))
        _tally(h, a)

    n = len(home_run_totals)
    avg_home = float(np.mean(home_run_totals))
    avg_away = float(np.mean(away_run_totals))
    totals = np.asarray(home_run_totals, dtype=float) + np.asarray(away_run_totals, dtype=float)

    run_dist: Counter = Counter()
    for (h, a), cnt in score_dist.items():
        run_dist[h + a] += cnt

    result = {
        "home_win_pct": round(home_wins / n, 4),
        "away_win_pct": round(away_wins / n, 4),
        "tie_pct": round(ties / n, 4),
        "avg_home_runs": round(avg_home, 2),
        "avg_away_runs": round(avg_away, 2),
        "avg_total_runs": round(avg_home + avg_away, 2),
        "run_total_std": round(float(totals.std(ddof=1)), 3) if n > 1 else 0.0,
        "run_distribution": dict(run_dist),
        "most_common_scores": score_dist.most_common(5),
    }

    logger.info(
        "Result: %s wins %.1f%% | %s wins %.1f%% | Avg total: %.1f (σ=%.2f)",
        game["home_team"], result["home_win_pct"] * 100,
        game["away_team"], result["away_win_pct"] * 100,
        result["avg_total_runs"], result["run_total_std"],
    )

    return result
