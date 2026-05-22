"""
Per at-bat probability engine — multinomial logit (softmax over log-odds).

Mathematical framework
----------------------
For each plate appearance, the outcome probability vector p ∈ Δ^6
(the 6-simplex over 7 outcomes Ω = {HR,BB,K,1B,2B,3B,OUT}) is computed as

    p_i = softmax_i(η)  =  exp(η_i) / Σ_j exp(η_j)

with the log-probability vector η ∈ R^7 formed by an *additive composition*
of per-factor contributions:

    η_i = log(p_i^B)                         (batter log-probability baseline)
        + s_pitch · γ_i^pitch                (pitcher-quality contribution)
        + s_split · γ_i^split                (L/R split contribution)
        + s_mix   · γ_i^mix                  (pitch-mix matchup contribution)
        + s_ev    · γ_i^ev                   (exit-velocity contribution)
        + s_wind  · γ_i^wind                 (environmental contribution)

With η_i = log(p_i^B) and no shifts, softmax(η) recovers p^B exactly.
Each shift η_i ← η_i + Δ_i multiplies the unnormalised mass by exp(Δ_i)
before the softmax renormalises — i.e. an *exponential-family
multiplicative reweighting* that always returns a valid probability
distribution.

Each scalar signal s_• ∈ R is a *log-ratio* of relevant Statcast/season
quantities — natively in log space. Each γ_i^• ∈ R is a fixed
sensitivity vector that redistributes probability mass across outcomes.

This is the canonical multinomial-logit / Plackett-Luce formulation
and is equivalent to Bill James' log-5 Bayesian update for a single
outcome (γ_i an indicator vector, signal = log(p_p)−log(p_lg)).
Compared to the previous multiplicative cascade ( prob × factor × (2−factor) ... )
this approach:

  1. Stays on the probability simplex by construction (softmax).
  2. Is invariant to outcome ordering and identity (no ad-hoc
     (2−factor) trick for "negative" outcomes).
  3. Composes Bayesian-correctly: independent evidence sources add
     in log space (multiplicative odds-ratio composition).
  4. Has a closed-form Jacobian  ∂p_i/∂η_j = p_i(δ_ij − p_j),
     enabling gradient-based fitting in future work.

Per at-bat expected-runs contribution (calculus aggregation)
------------------------------------------------------------
For state s = (b1, b2, b3, outs) and matchup (B, P):

    E[ΔR | s, B, P]  =  Σ_i  p_i(B, P) · [ r_i(s) + RE(s'_i) − RE(s) ]

where r_i(s) is the immediate runs from outcome i and RE(·) is the
24-state run-expectancy matrix (see model.run_expectancy). The game's
expected total is the iterated sum of E[ΔR] along the simulated
state trajectory — a discrete Bellman recursion.
"""

import math
import numpy as np
from config import (
    FIP_WOBA_INTERCEPT, FIP_WOBA_SLOPE,
    LEAGUE_AVG_BARREL_RATE, LEAGUE_AVG_MEAN_LA, LEAGUE_AVG_SPRINT_SPEED,
)

OUTCOMES = ["HR", "BB", "K", "1B", "2B", "3B", "OUT"]
POSITIVE_OUTCOMES = {"HR", "BB", "1B", "2B", "3B"}
NEGATIVE_OUTCOMES = {"K", "OUT"}

# Neutral outcome rates when a batter has zero data
_NEUTRAL_RATES = {
    "HR": 0.030, "BB": 0.085, "K": 0.225,
    "1B": 0.145, "2B": 0.050, "3B": 0.005, "OUT": 0.460,
}

# Sensitivity vectors γ_i^• : how each outcome's log-odds responds to a
# unit increase in the corresponding scalar signal. Calibrated so the
# implied probability changes match historical sabermetric elasticities.
#
# Constant shifts in γ are absorbed by softmax (softmax(η+c)=softmax(η)),
# so only the *differences* between outcomes matter; values are written
# in a centered form for readability.
_PITCH_QUALITY_GAMMA = {
    "HR":  +1.25, "2B": +0.90, "3B": +0.55, "1B": +0.65,
    "BB":  +0.45, "K": -1.20, "OUT": -0.50,
}
_SPLIT_GAMMA = {
    "HR":  +1.10, "2B": +0.80, "3B": +0.55, "1B": +0.55,
    "BB":  +0.30, "K": -1.05, "OUT": -0.55,
}
_MIX_GAMMA = {
    "HR":  +1.05, "2B": +0.75, "3B": +0.50, "1B": +0.55,
    "BB":  +0.20, "K": -0.90, "OUT": -0.45,
}
_EV_GAMMA = {  # Exit velocity moves only contact-quality outcomes
    "HR":  +1.55, "2B": +0.65, "3B": +0.30, "1B":  0.00,
    "BB":   0.00, "K":   0.00, "OUT": -0.40,
}
_WIND_GAMMA = {  # Wind moves only fly-ball outcomes
    "HR":  +1.00, "2B": +0.25, "3B": +0.15, "1B":  0.00,
    "BB":   0.00, "K":   0.00, "OUT": -0.20,
}
_BARREL_GAMMA = {  # High barrel rate → extreme fly-ball power
    "HR":  +1.60, "2B": +0.55, "3B": +0.10, "1B": -0.15,
    "BB":   0.00, "K":   0.00, "OUT": -0.60,
}
_LA_GAMMA = {  # High mean launch angle → more fly balls → HR/2B up, groundouts down
    "HR":  +1.20, "2B": +0.70, "3B": +0.30, "1B": -0.30,
    "BB":   0.00, "K":   0.00, "OUT": -0.40,
}
_SPRINT_GAMMA = {  # Fast runner → more singles, more triples, fewer outs on close plays
    "HR":   0.00, "2B": +0.15, "3B": +0.80, "1B": +0.55,
    "BB":   0.00, "K":   0.00, "OUT": -0.45,
}
_PARK_GAMMA = {  # Hitter-friendly park → more HR (and marginally more 2B)
    "HR":  +1.00, "2B": +0.15, "3B":  0.00, "1B":  0.00,
    "BB":   0.00, "K":   0.00, "OUT": -0.20,
}

# Magnitude clamp on each scalar signal (log-odds units).
# 0.35 ≈ ratio of 1.42×; 0.20 ≈ 1.22×.  Hard physical bounds, not soft priors.
_S_PITCH_CLAMP  = 0.35
_S_SPLIT_CLAMP  = 0.30
_S_MIX_CLAMP    = 0.30
_S_EV_CLAMP     = 0.18
_S_WIND_CLAMP   = 0.30
_S_BARREL_CLAMP = 0.25
_S_LA_CLAMP     = 0.20
_S_SPRINT_CLAMP = 0.18
_S_PARK_CLAMP   = 0.22


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_at_bat_probs(
    batter: dict,
    pitcher: dict,
    outward_wind_mph: float = 0.0,
    park_hr_factor: float = 1.0,
) -> dict[str, float]:
    """
    Multinomial-logit probability vector for a single plate appearance.

    p_i = softmax_i( logit(p_i^B) + Σ_• s_• · γ_i^• )

    Every adjustment is an additive log-odds shift, then softmax
    normalises the result onto the simplex.

    Signal layers (in order):
      pitch_quality — pitcher wOBA allowed vs batter wOBA
      split         — batter L/R platoon advantage
      mix           — pitch-type matchup (geometric mean Statcast wOBA)
      ev            — exit velocity ratio (pitcher allows vs batter produces)
      barrel        — batter barrel rate vs league average
      la            — batter mean launch angle vs league average
      sprint        — batter sprint speed vs league average
      wind          — outward wind component
      park          — ballpark HR factor
    """
    rates = batter.get("outcome_rates") or _NEUTRAL_RATES
    eta = {o: _logp(rates.get(o, _NEUTRAL_RATES[o])) for o in OUTCOMES}

    eta = _shift(eta, _pitch_quality_signal(batter, pitcher), _PITCH_QUALITY_GAMMA)
    eta = _shift(eta, _split_signal(batter, pitcher),         _SPLIT_GAMMA)
    eta = _shift(eta, _mix_signal(batter, pitcher),           _MIX_GAMMA)
    eta = _shift(eta, _ev_signal(batter, pitcher),            _EV_GAMMA)
    eta = _shift(eta, _barrel_signal(batter),                 _BARREL_GAMMA)
    eta = _shift(eta, _la_signal(batter),                     _LA_GAMMA)
    eta = _shift(eta, _sprint_signal(batter),                 _SPRINT_GAMMA)
    eta = _shift(eta, _wind_signal(outward_wind_mph),         _WIND_GAMMA)
    eta = _shift(eta, _park_signal(park_hr_factor),           _PARK_GAMMA)

    return _softmax(eta)


def sample_outcome(probs: dict[str, float]) -> str:
    """Sample one at-bat outcome from the categorical distribution."""
    outcomes = list(probs.keys())
    weights = [probs[o] for o in outcomes]
    return np.random.choice(outcomes, p=weights)


# ---------------------------------------------------------------------------
# Scalar signal functions  s_• ∈ R   (each in log-odds units, then clamped)
# ---------------------------------------------------------------------------

def _pitch_quality_signal(batter: dict, pitcher: dict) -> float:
    """
    s_pitch  =  log( pitcher_woba_allowed / batter_woba )

      > 0 : pitcher allows MORE than the batter normally produces (batter edge)
      < 0 : pitcher dominates the batter (pitcher edge)
      = 0 : neutral matchup
    """
    bw = batter.get("woba") or 0.0
    if bw <= 0:
        return 0.0
    pw = _pitcher_avg_woba_allowed(pitcher)
    if pw <= 0:
        return 0.0
    return float(np.clip(math.log(pw / bw), -_S_PITCH_CLAMP, _S_PITCH_CLAMP))


def _split_signal(batter: dict, pitcher: dict) -> float:
    """
    s_split  =  log( batter_woba_vs_hand / batter_overall_woba )
    """
    bw = batter.get("woba") or 0.0
    if bw <= 0:
        return 0.0
    hand = pitcher.get("hand", "R")
    wh = (batter.get("woba_vs_hand") or {}).get(hand)
    if not wh or wh <= 0:
        return 0.0
    return float(np.clip(math.log(wh / bw), -_S_SPLIT_CLAMP, _S_SPLIT_CLAMP))


def _mix_signal(batter: dict, pitcher: dict) -> float:
    """
    Pitch-mix signal — log of pitch-mix-weighted geometric mean of
    per-pitch wOBAs, normalised by the batter's overall wOBA.

        s_mix = log( Σ_p w_p · sqrt(batter_woba_vs_p · pitcher_woba_allowed_p)
                     / batter_overall_woba )

    Geometric mean is the maximum-likelihood combination of two
    independent log-normal signals — i.e. the canonical Bayesian
    fusion of batter and pitcher pitch-type performance.
    """
    pitch_mix = pitcher.get("pitch_mix") or {}
    if not pitch_mix:
        return 0.0

    bvp = batter.get("woba_vs_pitch") or {}
    pwa = pitcher.get("pitch_woba_allowed") or {}
    bw = batter.get("woba") or 0.0
    pw = _pitcher_avg_woba_allowed(pitcher)
    if pw <= 0:
        return 0.0
    normalizer = bw if bw > 0 else pw

    geo_sum = 0.0
    total = 0.0
    for pt, pct in pitch_mix.items():
        if pct <= 0:
            continue
        ppt = pwa.get(pt, pw)
        if ppt <= 0:
            continue
        bpt = bvp.get(pt) or bw or pw
        geo_sum += pct * math.sqrt(bpt * ppt)
        total += pct
    if total <= 0:
        return 0.0

    return float(np.clip(math.log((geo_sum / total) / normalizer),
                         -_S_MIX_CLAMP, _S_MIX_CLAMP))


def _ev_signal(batter: dict, pitcher: dict) -> float:
    """
    s_ev  =  0.5 · log( pitcher_avg_ev_allowed / batter_avg_ev )

    Negative when pitcher induces softer-than-typical contact for this
    hitter (penalises power outcomes). Requires ≥ 75 mph both sides.
    """
    bev = batter.get("avg_ev") or 0.0
    pev = pitcher.get("avg_ev_allowed") or 0.0
    if bev < 75 or pev < 75:
        return 0.0
    return float(np.clip(0.5 * math.log(pev / bev),
                         -_S_EV_CLAMP, _S_EV_CLAMP))


def _wind_signal(outward_wind_mph: float) -> float:
    """
    s_wind  =  0.012 · outward_wind_mph   (linear in low-wind regime)

    Calibrated so 10 mph outward ≈ +12% odds shift on HR.
    """
    if abs(outward_wind_mph) < 0.5:
        return 0.0
    return float(np.clip(outward_wind_mph * 0.012,
                         -_S_WIND_CLAMP, _S_WIND_CLAMP))


def _barrel_signal(batter: dict) -> float:
    """
    s_barrel = log( batter_barrel_rate / LEAGUE_AVG_BARREL_RATE )

    Barrel rate (EV ≥ 98 mph AND LA 26–30°) captures the joint distribution
    of power and swing path that drives HR probability beyond what avg EV alone
    measures.  Positive when batter barrels at above-league rate.
    """
    br = batter.get("barrel_rate") or 0.0
    if br <= 0:
        return 0.0
    return float(np.clip(math.log(br / LEAGUE_AVG_BARREL_RATE),
                         -_S_BARREL_CLAMP, _S_BARREL_CLAMP))


def _la_signal(batter: dict) -> float:
    """
    s_la = (mean_la - LEAGUE_AVG_MEAN_LA) * 0.015

    Launch angle characterises swing path: high LA → fly-ball hitter (HR/2B up);
    low / negative LA → ground-ball hitter (1B up, HR down).  Uses a linear
    deviation rather than log ratio because LA can be negative.
    0.015 scaling: ±10° deviation ≈ ±0.15 log-odds (roughly one clamp unit).
    """
    mla = batter.get("mean_la")
    if mla is None:
        return 0.0
    return float(np.clip((mla - LEAGUE_AVG_MEAN_LA) * 0.015,
                         -_S_LA_CLAMP, _S_LA_CLAMP))


def _sprint_signal(batter: dict) -> float:
    """
    s_sprint = log( sprint_speed / LEAGUE_AVG_SPRINT_SPEED )

    Faster runners beat out more infield singles and take extra bases on hits,
    converting some would-be outs into singles and some singles into triples.
    Zero when no sprint-speed data is available (neutral).
    """
    spd = batter.get("sprint_speed") or 0.0
    if spd < 18:  # below plausible human minimum — treat as missing
        return 0.0
    return float(np.clip(math.log(spd / LEAGUE_AVG_SPRINT_SPEED),
                         -_S_SPRINT_CLAMP, _S_SPRINT_CLAMP))


def _park_signal(park_hr_factor: float) -> float:
    """
    s_park = log( park_hr_factor )

    Park factor relative to 1.0 (neutral).  Applies a consistent HR-friendly /
    pitcher-friendly environment shift independent of the batter/pitcher matchup.
    """
    if abs(park_hr_factor - 1.0) < 0.005:
        return 0.0
    return float(np.clip(math.log(park_hr_factor),
                         -_S_PARK_CLAMP, _S_PARK_CLAMP))


# ---------------------------------------------------------------------------
# Log-odds arithmetic
# ---------------------------------------------------------------------------

def _logp(p: float) -> float:
    """Numerically safe log of a probability."""
    return math.log(max(p, 1e-9))


def _shift(eta: dict, signal: float, gamma: dict) -> dict:
    """Apply  η_i ← η_i + signal · γ_i  for every outcome."""
    if signal == 0.0:
        return eta
    return {o: eta[o] + signal * gamma[o] for o in OUTCOMES}


def _softmax(eta: dict) -> dict[str, float]:
    """Numerically stable softmax over outcomes."""
    m = max(eta.values())
    exps = {o: math.exp(v - m) for o, v in eta.items()}
    z = sum(exps.values())
    return {o: v / z for o, v in exps.items()}


def _pitcher_avg_woba_allowed(pitcher: dict) -> float:
    """Pitcher's avg wOBA allowed — player-specific, no league constants."""
    by_pitch = pitcher.get("pitch_woba_allowed") or {}
    if by_pitch:
        vals = [v for v in by_pitch.values() if v > 0]
        if vals:
            return sum(vals) / len(vals)
    overall = pitcher.get("woba_allowed_overall")
    if overall:
        return overall
    fip = pitcher.get("fip", 4.20)
    return max(0.180, min(0.420, FIP_WOBA_INTERCEPT + fip * FIP_WOBA_SLOPE))
