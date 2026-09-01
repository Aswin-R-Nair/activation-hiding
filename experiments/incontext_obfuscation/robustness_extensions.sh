#!/usr/bin/env bash
# Robustness extensions to the published negative (2026-08-31). Run in tmux.
#
# Ordered by what can still overturn the headline:
#
#   A. FLRT   -- the only serious gap. Nearest-neighbour snapping (§13c) tests
#      whether the SOFT optimum has a token neighbour, not whether the token
#      manifold contains a suppressing sequence. A native discrete search is the
#      correct test. Bailey et al. report GCG failing and FLRT succeeding against
#      activation probes, so a positive here is live. Universal first (matches our
#      claim), per-input second (Bailey's setting, and the positive control that
#      proves the search works at all).
#   B. max-pooled probe -- RL-Obfuscation reports max-pooling is harder to evade.
#      Without it, our negative is confounded with mean-pooling being weak.
#   C. non-abliterated control -- teacher-forced arms only, on stock gemma-2-9b-it.
#   D. current-model replication -- gemma-3-12b-it holds FAMILY fixed against
#      gemma-2-9b, so it isolates model generation (the 9B->70B jump confounded
#      scale and family).
#   E. delta-h cosine -- is the adversarial prefix doing what the trigger does?
#   F. 9B black-box re-run -- the only way to put an interval on §12d's 0.839 vs
#      0.846; that run's trajectory.csv was never pulled off the box.
#
# Each stage is independent; comment out what you are not running.
# Usage:  STAGES="A B" bash robustness_extensions.sh

set -uo pipefail   # NOT -e: one failed run must not abort the batch

STAGES="${STAGES:-A B C D E F}"
PY="${PY:-python}"
N="${N:-40}"
LAYER9B="${LAYER9B:-12}"
BASE9B="${BASE9B:-IlyaGusev/gemma-2-9b-it-abliterated}"
STOCK9B="${STOCK9B:-google/gemma-2-9b-it}"
ORGANISM="${ORGANISM:-serteal/neural-chameleon-gemma_2_9b-layer_12}"
NEW_MODEL="${NEW_MODEL:-google/gemma-3-12b-it}"
NEW_LAYERS="${NEW_LAYERS:-10,16,22,28,34,40}"
FLRT_STEPS="${FLRT_STEPS:-800}"
# p_add (0.5) exceeds p_del (0.25), so the prefix grows ~0.25 tokens/step and a
# low cap binds for most of the run. 128 matches the upper end of the soft-prefix
# k-sweep in 13c, so the discrete attack gets the same capacity ceiling the
# continuous one had -- otherwise a null is confounded with a tighter budget.
FLRT_MAX_LEN="${FLRT_MAX_LEN:-128}"
MICRO_BATCH="${MICRO_BATCH:-8}"

has() { [[ " $STAGES " == *" $1 "* ]]; }
run() { echo; echo "### $*"; date -Is; "$@"; echo "### exit $?"; }

# ---------------------------------------------------------------------------
# A. FLRT discrete attack -- the load-bearing one
# ---------------------------------------------------------------------------
if has A; then
  # Per-input FIRST. It is Bailey's own setting and is EXPECTED to partially
  # succeed. If it does not, the search is broken and a universal null means
  # nothing -- stop and fix before reading anything else in this file.
  run $PY flrt_universal.py --model "$BASE9B" --probe harmful --layer "$LAYER9B" \
      --n "$N" --steps "$FLRT_STEPS" --per-input --per-input-n 8 \
      --micro-batch "$MICRO_BATCH" --max-len "$FLRT_MAX_LEN" --tag flrt9b_perinput

  # The claim: ONE prefix, held-out passages, selective.
  # Multiple seeds because a NEGATIVE from a single initialisation is weak: the
  # buffer starts at random real tokens, and the smoke run showed the init
  # surviving into the final prefix. If all seeds land in the same place the null
  # is about the token manifold; if they scatter, it is about search variance.
  for SEED in ${FLRT_SEEDS:-0 1 2}; do
    run $PY flrt_universal.py --model "$BASE9B" --probe harmful --layer "$LAYER9B" \
        --n "$N" --test-frac 0.35 --steps "$FLRT_STEPS" --lam 1.0 --seed "$SEED" \
        --random-control --micro-batch "$MICRO_BATCH" --max-len "$FLRT_MAX_LEN" --tag "flrt9b_universal_s${SEED}"
  done

  # lam is the same knob that produced the §12e tradeoff; if FLRT has a sweet
  # spot the soft attack lacked, it will be here.
  for LAM in 5 10; do
    run $PY flrt_universal.py --model "$BASE9B" --probe harmful --layer "$LAYER9B" \
        --n "$N" --test-frac 0.35 --steps "$FLRT_STEPS" --lam "$LAM" \
        --micro-batch "$MICRO_BATCH" --max-len "$FLRT_MAX_LEN" --tag "flrt9b_lam${LAM}"
  done
fi

# ---------------------------------------------------------------------------
# B. max-pooled probe (train it, then re-run the two headline tests against it)
# ---------------------------------------------------------------------------
if has B; then
  # --reserve keeps the experiment samples disjoint from the probe's training set.
  run $PY train_probe.py --model "$BASE9B" --probe harmful --pool max \
      --layers "$LAYER9B" --n 100 --reserve "$N" --steps 300 --batch-size 4
  MAXPROBE="harmful_$(basename "$BASE9B")_maxpool"
  # GATE: if test AUROC < 0.9 the max-pooled probe is a bad instrument and
  # nothing below is interpretable. Check the printed scan before continuing.
  run $PY run_incontext_eval.py --model "$BASE9B" --probe "$MAXPROBE" \
      --layer "$LAYER9B" --n "$N" --tag maxpool_singleshot
  run $PY optimize_universal_prefix.py --model "$BASE9B" --probe "$MAXPROBE" \
      --layer "$LAYER9B" --n "$N" --test-frac 0.35 --k 16 --steps 400 --lr 0.03 \
      --proj-norm --lam 5 --tag maxpool_univprefix
  run $PY flrt_universal.py --model "$BASE9B" --probe "$MAXPROBE" \
      --layer "$LAYER9B" --n "$N" --test-frac 0.35 --steps "$FLRT_STEPS" \
      --random-control --micro-batch "$MICRO_BATCH" --max-len "$FLRT_MAX_LEN" --tag maxpool_flrt
fi

# ---------------------------------------------------------------------------
# C. non-abliterated control -- teacher-forced arms only
# ---------------------------------------------------------------------------
if has C; then
  # Abliteration is an activation-space edit, adjacent to what we are probing,
  # and the shipped probes were not trained on the abliterated stream. Generation
  # arms (scratchpad, blackbox) are NOT run here: stock gemma-2-9b-it refuses the
  # harmful prompts, which would confound refusal with obfuscation.
  run $PY run_incontext_eval.py --model "$STOCK9B" --probe harmful \
      --layer "$LAYER9B" --n "$N" --tag stock9b_singleshot
  run $PY optimize_universal_prefix.py --model "$STOCK9B" --probe harmful \
      --layer "$LAYER9B" --n "$N" --test-frac 0.35 --k 16 --steps 400 --lr 0.03 \
      --proj-norm --lam 5 --tag stock9b_univprefix
fi

# ---------------------------------------------------------------------------
# D. current open model: gemma-3-12b-it
# ---------------------------------------------------------------------------
if has D; then
  # Gemma-3 12b loads as a multimodal wrapper; the harness now locates the
  # decoder stack generically and asserts hidden_states indexes it. If the
  # assertion fires, fix decoder_layers() -- do not skip past it.
  run $PY train_probe.py --model "$NEW_MODEL" --probe harmful \
      --layers "$NEW_LAYERS" --n 100 --reserve "$N" --steps 300 --batch-size 2
  # GATE: best test AUROC must be >= 0.9. Read it off the scan, then set L.
  NEWPROBE="harmful_$(basename "$NEW_MODEL")"
  L="${L:-22}"   # <- REPLACE with the winning layer from the scan above
  run $PY run_incontext_eval.py --model "$NEW_MODEL" --probe "$NEWPROBE" \
      --layer "$L" --n "$N" --tag g3_singleshot
  for LAM in 1 5 10; do
    run $PY optimize_universal_prefix.py --model "$NEW_MODEL" --probe "$NEWPROBE" \
        --layer "$L" --n "$N" --test-frac 0.35 --k 16 --steps 400 --lr 0.03 \
        --proj-norm --lam "$LAM" --tag "g3_lam${LAM}"
  done
  run $PY flrt_universal.py --model "$NEW_MODEL" --probe "$NEWPROBE" --layer "$L" \
      --n "$N" --test-frac 0.35 --steps "$FLRT_STEPS" --random-control \
      --micro-batch "$MICRO_BATCH" --max-len "$FLRT_MAX_LEN" --tag g3_flrt
fi

# ---------------------------------------------------------------------------
# E. delta-h cosine: trigger vs universal prefix
# ---------------------------------------------------------------------------
if has E; then
  # Neither artefact survives on disk -- no 9B prefix was ever pulled back and no
  # script ever saved activations -- so the prefix is regenerated first.
  # Extract BOTH deltas from the ORGANISM so the residual bases are aligned; the
  # organism responds to a prefix too, it merely was not trained on one.
  run $PY optimize_universal_prefix.py --model "$BASE9B" --probe harmful \
      --layer "$LAYER9B" --n "$N" --test-frac 0.35 --k 16 --steps 400 --lr 0.03 \
      --lam 1.0 --tag dh_prefix_source
  PREF="outputs/dh_prefix_source_harmful_univprefix/prefix.pt"
  run $PY delta_h_cosine.py extract --model "$ORGANISM" --mode trigger \
      --probe harmful --layer "$LAYER9B" --n "$N" --out outputs/dh_trigger.pt
  run $PY delta_h_cosine.py extract --model "$ORGANISM" --mode prefix \
      --prefix-pt "$PREF" --probe harmful --layer "$LAYER9B" --n "$N" \
      --out outputs/dh_prefix.pt
  run $PY delta_h_cosine.py compare outputs/dh_trigger.pt outputs/dh_prefix.pt \
      --probe harmful --out outputs/dh_cosine.json
fi

# ---------------------------------------------------------------------------
# F. 9B black-box re-run, so §12d's 0.839 vs 0.846 can get an interval
# ---------------------------------------------------------------------------
if has F; then
  run $PY run_blackbox_search.py --model "$BASE9B" --probe harmful \
      --concept harmful --layer "$LAYER9B" --n 20 --rounds 25 --k 16 \
      --modes model random --tag bb9b
  # then, locally:
  #   python blackbox_significance.py outputs/bb9b_*_blackbox_model \
  #                                   outputs/bb9b_*_blackbox_random
fi

echo; echo "ALL REQUESTED STAGES DONE"; date -Is
echo "tar czf obf_robustness_ext.tar.gz outputs/ && scp it off before teardown"
