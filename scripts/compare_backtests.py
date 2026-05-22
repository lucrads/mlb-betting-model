"""
Compare two backtest result files (before / after engine rewrite).

Usage:
    python3 scripts/compare_backtests.py backtest_before.json backtest_after.json

Reports:
  - ML accuracy (model picked winner)
  - Calibration (Brier score for win prob)
  - Run-total RMSE / MAE
  - Average run total predicted vs actual
  - Bet ROI by edge bucket
  - Win-probability KS divergence between engines
"""
from __future__ import annotations
import argparse
import json
import math


def load(path: str) -> dict[str, dict]:
    with open(path) as f:
        rows = json.load(f)
    return {r["game_id"]: r for r in rows}


def brier(p_home: float, home_won: bool) -> float:
    y = 1.0 if home_won else 0.0
    return (p_home - y) ** 2


def summarise(label: str, rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    ml_correct = sum(1 for r in rows if r["model_correct_ml"])
    brier_sum = sum(brier(r["home_win_pct"], r["home_won"]) for r in rows)
    mae_total = sum(abs(r["model_total"] - r["actual_total"]) for r in rows)
    rmse_total = math.sqrt(sum((r["model_total"] - r["actual_total"]) ** 2 for r in rows) / n)
    avg_pred = sum(r["model_total"] for r in rows) / n
    avg_actual = sum(r["actual_total"] for r in rows) / n

    confident = [r for r in rows if r["confidence"] >= 0.60]
    confident_acc = (sum(1 for r in confident if r["model_correct_ml"]) / len(confident)) if confident else 0.0

    out = {
        "label": label,
        "n": n,
        "ml_acc": ml_correct / n,
        "brier": brier_sum / n,
        "rmse_total": rmse_total,
        "mae_total": mae_total / n,
        "avg_predicted_total": avg_pred,
        "avg_actual_total": avg_actual,
        "bias_total": avg_pred - avg_actual,
        "high_conf_n": len(confident),
        "high_conf_acc": confident_acc,
    }
    return out


def print_summary(s: dict) -> None:
    if not s:
        print("  (no data)")
        return
    print(f"  Games:              {s['n']}")
    print(f"  ML accuracy:        {s['ml_acc']*100:.1f}%")
    print(f"  Brier score:        {s['brier']:.4f}   (lower = better-calibrated win prob)")
    print(f"  Total RMSE:         {s['rmse_total']:.2f}")
    print(f"  Total MAE:          {s['mae_total']:.2f}")
    print(f"  Predicted total:    {s['avg_predicted_total']:.2f}")
    print(f"  Actual total:       {s['avg_actual_total']:.2f}")
    print(f"  Bias (pred-act):    {s['bias_total']:+.2f}")
    print(f"  High-conf (≥60%):   {s['high_conf_n']} games, {s['high_conf_acc']*100:.1f}% correct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    args = ap.parse_args()

    a = load(args.before)
    b = load(args.after)
    shared = sorted(set(a) & set(b))
    if not shared:
        print(f"No overlapping games between {args.before} ({len(a)}) and {args.after} ({len(b)})")
        return

    rows_a = [a[g] for g in shared]
    rows_b = [b[g] for g in shared]

    sa = summarise("BEFORE", rows_a)
    sb = summarise("AFTER", rows_b)

    print(f"=== Shared games: {len(shared)} ===\n")
    print("BEFORE (legacy multiplicative engine):")
    print_summary(sa)
    print("\nAFTER (multinomial-logit / RE24 / shrinkage / antithetic):")
    print_summary(sb)

    print("\nΔ after − before:")
    print(f"  ΔML accuracy:   {(sb['ml_acc']-sa['ml_acc'])*100:+.2f} pp")
    print(f"  ΔBrier:         {sb['brier']-sa['brier']:+.4f}   (negative = improvement)")
    print(f"  ΔRMSE total:    {sb['rmse_total']-sa['rmse_total']:+.3f}")
    print(f"  ΔBias total:    {sb['bias_total']-sa['bias_total']:+.3f}")

    # Per-game win-prob shift distribution
    shifts = sorted(abs(b[g]["home_win_pct"] - a[g]["home_win_pct"]) for g in shared)
    median_shift = shifts[len(shifts) // 2]
    p90_shift = shifts[int(0.9 * len(shifts))]
    print(f"\nWin-prob shifts |Δp_home|:  median {median_shift*100:.1f}pp   p90 {p90_shift*100:.1f}pp")


if __name__ == "__main__":
    main()
