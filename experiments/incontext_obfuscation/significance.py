"""Paired significance testing for in-context obfuscation runs.

The summary table reports mean probe scores, but a small mean difference is
uninformative without knowing the per-sample variance. Because every condition
scores the *same* samples (identical `sample_idx`, identical response tokens),
differences are paired, which is far more sensitive than comparing group means.

Test: two-sided **sign-flip permutation** on paired differences. Assumption-light
(no normality, no scipy) and appropriate for bounded, skewed probe scores.

Reference condition defaults to `control_irrelevant`, not `baseline`. That is the
stricter and more meaningful comparison: it holds "a prefix exists at all" fixed,
so a surviving effect is attributable to prefix *content* rather than to prefix
presence.

Usage:
    python significance.py outputs/<tag>_<probe>/raw_scores.csv
    python significance.py outputs/.../raw_scores.csv --ref baseline --label negative
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_PERM = 20000
SEED = 0


def sign_flip_test(diff: np.ndarray, n_perm: int = N_PERM, seed: int = SEED):
    """Two-sided paired permutation test under the sign-symmetry null.

    Returns (observed_mean_diff, p_value).
    """
    rng = np.random.default_rng(seed)
    obs = float(diff.mean())
    # Under H0 the sign of each paired difference is exchangeable.
    flips = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    null = (flips * diff).mean(axis=1)
    # +1 correction keeps p strictly positive (Phipson & Smyth).
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, float(p)


def holm(pvals):
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)  # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv", type=Path)
    ap.add_argument("--ref", default="control_irrelevant",
                    help="Reference condition to compare against.")
    ap.add_argument("--label", default="positive", choices=["positive", "negative"])
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    df = pd.read_csv(args.raw_csv)
    df = df[df["label"] == args.label]

    wide = df.pivot(index="sample_idx", columns="condition", values="score")
    if args.ref not in wide.columns:
        raise SystemExit(f"reference condition '{args.ref}' not in {list(wide.columns)}")

    ref = wide[args.ref].to_numpy()
    others = [c for c in wide.columns if c != args.ref]

    rows = []
    for cond in others:
        # Positive `drop` = this condition scores LOWER than the reference.
        diff = ref - wide[cond].to_numpy()
        obs, p = sign_flip_test(diff, args.n_perm)
        # Cohen's dz: paired effect size.
        dz = obs / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        rows.append({"condition": cond, "mean_drop_vs_ref": obs,
                     "cohens_dz": dz, "p_raw": p})

    out = pd.DataFrame(rows).sort_values("mean_drop_vs_ref", ascending=False)
    out["p_holm"] = holm(out["p_raw"].to_numpy())
    out["sig_0.05"] = np.where(out["p_holm"] < 0.05, "*", "")

    n = wide.shape[0]
    print(f"\nPaired sign-flip permutation test  (n={n} {args.label} samples, "
          f"{args.n_perm} permutations)")
    print(f"Reference condition: {args.ref}")
    print("Positive mean_drop_vs_ref = condition scores LOWER than reference.\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\np_holm is family-wise corrected across the "
          f"{len(others)} comparisons shown.")


if __name__ == "__main__":
    main()
