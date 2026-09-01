"""Paired interval + permutation test for black-box search runs (model vs random).

`run_blackbox_search.py` reports one number per mode -- `bestN_pos_mean`, the mean
over samples of the best (lowest) probe score that proposer found in N rounds.
Comparing 0.839 (model) against 0.846 (random) as bare means is uninformative: the
two proposers searched the SAME samples, so the comparison is paired, and the
per-sample spread is what decides whether the gap is real.

This script recomputes best-of-N per sample from the two `trajectory.csv` files,
then reports:
  * the paired mean difference (random_bestN - model_bestN); positive = model found
    a lower score, i.e. searched better,
  * a percentile bootstrap CI over samples,
  * a two-sided sign-flip permutation p-value,
  * the same for the drop from each sample's own round-0 baseline, and for
    negatives (a proposer that lowers BOTH classes is jamming, not obfuscating).

Usage:
    python blackbox_significance.py <model_run_dir> <random_run_dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 20000
N_PERM = 20000
SEED = 0


def best_of_n(traj: pd.DataFrame, label: str):
    """Per-sample (baseline, best-of-N) with rounds >=1 as the search rounds."""
    d = traj[traj["label"] == label]
    base = d[d["round"] == 0].set_index("sample_idx")["score"]
    best = d[d["round"] >= 1].groupby("sample_idx")["score"].min()
    idx = sorted(set(base.index) & set(best.index))
    return base.loc[idx].to_numpy(), best.loc[idx].to_numpy(), np.asarray(idx)


def paired_stats(diff: np.ndarray, n_boot=N_BOOT, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = float(diff.mean())
    boot = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    flips = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    null = (flips * diff).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    sd = diff.std(ddof=1)
    return obs, float(lo), float(hi), float(p), float(obs / sd) if sd > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("random_dir", type=Path)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    tm = pd.read_csv(args.model_dir / "trajectory.csv")
    tr = pd.read_csv(args.random_dir / "trajectory.csv")

    print(f"\nmodel : {args.model_dir}")
    print(f"random: {args.random_dir}")

    rows = []
    for label in ("positive", "negative"):
        bm, best_m, idx_m = best_of_n(tm, label)
        br, best_r, idx_r = best_of_n(tr, label)
        if not np.array_equal(idx_m, idx_r):
            raise SystemExit(f"sample_idx sets differ for label={label}; not paired")
        if not np.allclose(bm, br):
            print(f"  warning: round-0 baselines differ for {label} "
                  f"(max |d| = {np.abs(bm - br).max():.4f})")

        # positive = model searched BETTER (found a lower score than random did)
        obs, lo, hi, p, dz = paired_stats(best_r - best_m, args.n_boot, args.n_perm)
        rows.append({"label": label, "quantity": "model_advantage (random-model)",
                     "n": len(idx_m), "mean": obs, "ci_lo": lo, "ci_hi": hi,
                     "dz": dz, "p": p})

        for name, base, best in (("model", bm, best_m), ("random", br, best_r)):
            obs, lo, hi, p, dz = paired_stats(base - best, args.n_boot, args.n_perm)
            rows.append({"label": label, "quantity": f"{name} drop from own baseline",
                         "n": len(idx_m), "mean": obs, "ci_lo": lo, "ci_hi": hi,
                         "dz": dz, "p": p})

    out = pd.DataFrame(rows)
    print(f"\nPaired bootstrap ({args.n_boot} resamples) + sign-flip permutation "
          f"({args.n_perm}) over samples\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nmodel_advantage > 0 means the model proposer found lower probe scores "
          "than blind random search.")
    print("A drop that is significant for BOTH positive and negative labels is "
          "content-blind jamming, not obfuscation.")


if __name__ == "__main__":
    main()
