"""
Run Expectancy (RE24) — the calculus layer that aggregates per-at-bat
outcome probabilities into expected runs.

For a base-out state s = (b1, b2, b3, outs) the run-expectancy function
RE(s) gives the expected runs scored in the remainder of the half-inning
under league-average conditions. The full lookup is the 24-state matrix
(8 base-occupancy states × 3 out states).

The per-PA expected run delta for matchup (B, P) in state s is

    E[ΔR | s, B, P]  =  Σ_i  p_i(B, P) · [ r_i(s) + RE(s'_i(s)) − RE(s) ]

where:
  p_i(B, P)     – outcome probability from the multinomial-logit engine
  r_i(s)        – immediate runs scored on outcome i from state s
  s'_i(s)       – new base-out state after outcome i

Aggregating E[ΔR] along the iterated state trajectory gives the
calculus-based expected game total — a smooth, differentiable analogue
of the stochastic Monte Carlo path.

The base-running probabilities mirror config.BASE_RUNNING so the RE
"calculus" stays consistent with the simulator's stochastic dynamics.
"""

from __future__ import annotations
from config import BASE_RUNNING


# ---------------------------------------------------------------------------
# 24-state run-expectancy matrix (Tom Tango / Lichtman, MLB-era averages)
# Keys: (b1, b2, b3) tuple of booleans → outs → expected remaining runs
# ---------------------------------------------------------------------------
RE_MATRIX: dict[tuple[bool, bool, bool], tuple[float, float, float]] = {
    (False, False, False): (0.486, 0.257, 0.097),
    (True,  False, False): (0.866, 0.510, 0.224),
    (False, True,  False): (1.073, 0.655, 0.319),
    (False, False, True ): (1.300, 0.969, 0.366),
    (True,  True,  False): (1.435, 0.890, 0.420),
    (True,  False, True ): (1.640, 1.150, 0.494),
    (False, True,  True ): (1.860, 1.327, 0.601),
    (True,  True,  True ): (2.230, 1.560, 0.781),
}


def re_value(bases: list[bool], outs: int) -> float:
    """Return RE(s) for the given base-out state. RE = 0 when inning ends."""
    if outs >= 3:
        return 0.0
    key = (bool(bases[0]), bool(bases[1]), bool(bases[2]))
    return RE_MATRIX[key][outs]


# ---------------------------------------------------------------------------
# Deterministic post-outcome state — uses the EXPECTED base advancement
# (probabilities × destinations) rather than a single sampled realisation.
# ---------------------------------------------------------------------------

def _expected_transition(
    outcome: str,
    bases: list[bool],
    outs: int,
) -> tuple[float, list[float], int]:
    """
    Return (expected_runs, expected_base_occupancy, new_outs) for a given
    outcome from state (bases, outs). Base occupancy returned as floats in
    [0,1] — fractional values are valid because we take expectations over
    base-running uncertainty.
    """
    b1, b2, b3 = [1.0 if b else 0.0 for b in bases]
    rng = BASE_RUNNING

    if outcome == "HR":
        runs = 1.0 + b1 + b2 + b3
        return runs, [0.0, 0.0, 0.0], outs

    if outcome == "3B":
        return (b1 + b2 + b3), [0.0, 0.0, 1.0], outs

    if outcome == "2B":
        p_1to_home = rng["double_runner_1b_scores_prob"]
        runs = b2 + b3 + b1 * p_1to_home
        new_b3 = b1 * (1.0 - p_1to_home)
        return runs, [0.0, 1.0, new_b3], outs

    if outcome == "1B":
        p_2sc = rng["single_runner_2b_scores_prob"]
        p_1to3 = rng["single_runner_1b_to_3b_prob"]
        runs = b3 + b2 * p_2sc
        # Runner-from-2nd ends at 3rd if didn't score
        new_b3_from_2 = b2 * (1.0 - p_2sc)
        # Runner-from-1st: takes 3rd with p_1to3, else stays at 2nd
        new_b3_from_1 = b1 * p_1to3
        new_b2_from_1 = b1 * (1.0 - p_1to3)
        # Batter to 1st
        new_b1 = 1.0
        new_b2 = new_b2_from_1
        new_b3 = new_b3_from_2 + new_b3_from_1
        return runs, [new_b1, new_b2, new_b3], outs

    if outcome == "BB":
        runs = 1.0 if (b1 > 0 and b2 > 0 and b3 > 0) else 0.0
        # Force advances only
        new_b1 = 1.0
        new_b2 = 1.0 if (b1 > 0) else b2
        new_b3 = 1.0 if (b1 > 0 and b2 > 0) else b3
        return runs, [new_b1, new_b2, new_b3], outs

    if outcome == "K":
        return 0.0, [b1, b2, b3], outs + 1

    # OUT — groundout/flyout, runner on 3rd scores with prob ~0.85 if <2 outs
    new_outs = outs + 1
    if new_outs < 3 and b3 > 0:
        p_3sc = rng["groundout_runner_3b_scores_prob"]
        runs = b3 * p_3sc
        new_b3 = b3 * (1.0 - p_3sc)
        return runs, [b1, b2, new_b3], new_outs
    return 0.0, [b1, b2, b3], new_outs


def _re_from_floats(bases: list[float], outs: int) -> float:
    """
    Compute RE for *fractional* base occupancy by linearly interpolating
    over the 2^3 = 8 corner states. Valid because RE_MATRIX is defined
    on binary corners and the expected base state is a convex combination.
    """
    if outs >= 3:
        return 0.0
    total = 0.0
    for k_b1 in (False, True):
        for k_b2 in (False, True):
            for k_b3 in (False, True):
                w = ((bases[0] if k_b1 else 1.0 - bases[0]) *
                     (bases[1] if k_b2 else 1.0 - bases[1]) *
                     (bases[2] if k_b3 else 1.0 - bases[2]))
                if w > 0:
                    total += w * RE_MATRIX[(k_b1, k_b2, k_b3)][outs]
    return total


def expected_run_delta(
    probs: dict[str, float],
    bases: list[bool],
    outs: int,
) -> float:
    """
    Calculus aggregation of per-at-bat outcome probabilities into
    expected runs added (RE24):

        E[ΔR | s, p]  =  Σ_i p_i · [ r_i(s) + RE(s'_i) − RE(s) ]

    Returns a real number — the expected change in inning run total
    contributed by this single plate appearance.
    """
    re_now = re_value(bases, outs)
    total = 0.0
    for outcome, p in probs.items():
        if p <= 0:
            continue
        runs, new_bases, new_outs = _expected_transition(outcome, bases, outs)
        re_next = _re_from_floats(new_bases, new_outs)
        total += p * (runs + re_next - re_now)
    return total


def expected_runs_per_pa(probs: dict[str, float]) -> float:
    """
    State-independent per-PA expected run value — equivalent to taking
    E[ΔR] in the empty-bases / 0-outs leadoff state. Useful as a
    standalone calculus score per matchup for the output layer.
    """
    return expected_run_delta(probs, [False, False, False], 0)


def woba_from_probs(probs: dict[str, float]) -> float:
    """
    wOBA-equivalent score from the outcome distribution, using the
    standard linear-weights coefficients. Provides a single scalar
    summary of the per-PA matchup quality.
    """
    return (
        0.69 * probs.get("BB", 0.0)
        + 0.89 * probs.get("1B", 0.0)
        + 1.27 * probs.get("2B", 0.0)
        + 1.62 * probs.get("3B", 0.0)
        + 2.10 * probs.get("HR", 0.0)
    )
