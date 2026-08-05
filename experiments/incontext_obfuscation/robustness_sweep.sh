#!/usr/bin/env bash
# Unattended robustness sweep for the 70B scaling negative (§13b).
#
# Closes the two "is it the method?" confounds before publishing the null:
#   (1) LAYER   -- layer 30 was near-arbitrary (all scanned layers AUROC ~1.0).
#   (2) PREFIX LENGTH k -- only k=16 was tested.
#
# Sequence:
#   A. wide per-layer probe scan (ONE forward pass extracts all layers)
#   B. all-layer single-shot SCREEN (cheap) -> perturbability map
#   C. gradient existence test at the most-perturbable layers + early/mid/late,
#      at lam in {5,10} (enough to see if the tradeoff frontier holds)
#   D. k-sweep at the established layer 30, lam=5, k in {4,16,64,128}
#
# Run inside tmux. Results in outputs/ (gitignored) -- tar+scp before teardown.
# Usage:  MODEL=huihui-ai/Llama-3.3-70B-Instruct-abliterated bash robustness_sweep.sh

set -uo pipefail   # NOT -e: one failed run should not abort the batch

MODEL="${MODEL:-huihui-ai/Llama-3.3-70B-Instruct-abliterated}"
BASENAME="$(basename "$MODEL")"
PROBE="harmful_${BASENAME}"
SCAN_LAYERS="${SCAN_LAYERS:-10,20,30,40,50,60,70}"
N="${N:-50}"
KVALS="${KVALS:-4 16 64 128}"
LAMS="${LAMS:-5 10}"
COVERAGE_LAYERS="${COVERAGE_LAYERS:-20 40 60}"   # early/mid/late, always tested
KSWEEP_LAYER="${KSWEEP_LAYER:-30}"

run() { echo; echo ">>> $*"; "$@" || echo "!!! FAILED (continuing): $*"; }

echo "=== A. per-layer probe scan (layers $SCAN_LAYERS) ==="
run python train_probe.py --model "$MODEL" --probe harmful \
    --layers "$SCAN_LAYERS" --n 100 --reserve "$N" --l2 1e-2 --steps 300 --batch-size 16

echo "=== B. all-layer single-shot screen ==="
run python screen_layers.py --model "$MODEL" --probe "$PROBE" --n "$N" --batch-size 16

# Pick the 2 most prefix-perturbable layers from the screen, union with coverage.
SCREEN_CSV="outputs/${BASENAME}_${PROBE}_layerscreen/perturbability_by_layer.csv"
TOP_LAYERS="$(python - "$SCREEN_CSV" <<'PY'
import sys, csv
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
    rows.sort(key=lambda r: abs(float(r["max_abs_drop"])), reverse=True)
    print(" ".join(r["layer"] for r in rows[:2]))
except Exception:
    print("")  # screen missing -> fall back to coverage only
PY
)"
GRAD_LAYERS="$(echo "$TOP_LAYERS $COVERAGE_LAYERS" | tr ' ' '\n' | sort -un | tr '\n' ' ')"
echo "=== gradient-test layers: $GRAD_LAYERS  (top-perturbable: ${TOP_LAYERS:-none} + coverage $COVERAGE_LAYERS) ==="

echo "=== C. gradient existence test across layers (lam in $LAMS) ==="
for L in $GRAD_LAYERS; do
  for LAM in $LAMS; do
    run python optimize_universal_prefix.py --model "$MODEL" \
        --probe "${PROBE}_L${L}" --layer "$L" \
        --n "$N" --test-frac 0.35 --k 16 --steps 400 --lr 0.03 \
        --proj-norm --lam "$LAM" --reserve-gib 0 --micro-batch 32 \
        --tag "70b_layer${L}_lam${LAM}"
  done
done

echo "=== D. k-sweep at layer $KSWEEP_LAYER (lam=5, k in $KVALS) ==="
for K in $KVALS; do
  run python optimize_universal_prefix.py --model "$MODEL" \
      --probe "${PROBE}_L${KSWEEP_LAYER}" --layer "$KSWEEP_LAYER" \
      --n "$N" --test-frac 0.35 --k "$K" --steps 400 --lr 0.03 \
      --proj-norm --lam 5 --reserve-gib 0 --micro-batch 32 \
      --tag "70b_ksweep_k${K}"
done

echo; echo "=== DONE. Collect results: ==="
echo "  tar czf ~/obf_robustness.tar.gz outputs/"
echo "Read: perturbability_by_layer.csv, and each 70b_layer*/70b_ksweep* summary.csv"
