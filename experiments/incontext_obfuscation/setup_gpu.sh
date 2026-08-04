#!/usr/bin/env bash
# Set up a fresh GPU instance for the in-context obfuscation experiments.
#
# On any new box (vast.ai PyTorch template, CUDA >= 12.1), run once:
#
#   curl -fsSL https://raw.githubusercontent.com/Aswin-R-Nair/neural-chameleons/incontext-obfuscation/experiments/incontext_obfuscation/setup_gpu.sh | bash
#
# It clones the repo (if not already present), installs deps, verifies the GPU
# is bf16-capable, and runs a CPU mock test to confirm the harness imports.
# Idempotent: safe to re-run.
#
# It deliberately does NOT reinstall torch -- the PyTorch template ships a
# CUDA-matched build, and reinstalling risks landing a CPU-only or arch-mismatched
# wheel (both of which have bitten us before).

set -euo pipefail

REPO_URL="https://github.com/Aswin-R-Nair/neural-chameleons.git"
BRANCH="incontext-obfuscation"
CLONE_DIR="/neural-chameleons"
EXP_SUBDIR="experiments/incontext_obfuscation"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARN:\033[0m %s\n' "$*"; }
die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
bold "== 1. GPU check =="
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,count --format=csv,noheader
else
    warn "nvidia-smi not found -- is this actually a GPU box?"
fi

# ---------------------------------------------------------------------------
bold "== 2. torch / bf16 capability =="
# ---------------------------------------------------------------------------
# Volta (sm_70) and Turing (sm_75) lack bf16 hardware; gemma-2 in fp16 can
# overflow and silently corrupt the layer-12 activations we measure. Require
# Ampere+ (compute capability major >= 8).
python - <<'PY' || die "torch/bf16 check failed -- see message above."
import sys
try:
    import torch
except ModuleNotFoundError:
    sys.exit("torch is not installed. Use a PyTorch template image; do NOT pip "
             "install torch here (risks a CPU-only wheel).")
print("torch", torch.__version__, "| cuda", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False -- wrong torch build or no GPU. "
             "Do not proceed; the experiment needs CUDA.")
major, minor = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name(0)
print(f"device {name}  compute capability {major}.{minor}")
if major < 8:
    sys.exit(f"{name} is compute {major}.{minor} (pre-Ampere) -- NO native bf16. "
             "gemma-2 would be forced to fp16 and may corrupt the measured "
             "activations. Destroy this instance and pick Ampere+ (A40/A100/"
             "L40S/4090/3090).")
archs = torch.cuda.get_arch_list()
print("compiled for", archs)
print("bf16 OK.")
PY

# ---------------------------------------------------------------------------
bold "== 3. Python deps =="
# ---------------------------------------------------------------------------
# torch is intentionally omitted (see header).
python -m pip install --quiet --upgrade pip
python -m pip install --quiet transformers accelerate pandas numpy hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "installed transformers / accelerate / pandas / numpy / hf_transfer"
echo "HF_HUB_ENABLE_HF_TRANSFER=1 (add to your shell rc to persist)"

# ---------------------------------------------------------------------------
bold "== 4. Repo =="
# ---------------------------------------------------------------------------
if [ -d "${EXP_SUBDIR}" ] && [ -f "${EXP_SUBDIR}/run_scratchpad_loop.py" ]; then
    # Already running from inside a checkout.
    REPO_ROOT="$(pwd)"
    echo "already inside a checkout: ${REPO_ROOT}"
    git pull --ff-only || warn "git pull failed (local changes?) -- continuing"
elif [ -d "${CLONE_DIR}/.git" ]; then
    echo "existing clone at ${CLONE_DIR} -- updating"
    git -C "${CLONE_DIR}" fetch --quiet origin "${BRANCH}"
    git -C "${CLONE_DIR}" checkout --quiet "${BRANCH}"
    git -C "${CLONE_DIR}" pull --ff-only || warn "git pull failed -- continuing"
    REPO_ROOT="${CLONE_DIR}"
else
    echo "cloning ${REPO_URL} (branch ${BRANCH}) into ${CLONE_DIR}"
    git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${CLONE_DIR}"
    REPO_ROOT="${CLONE_DIR}"
fi

EXP_DIR="${REPO_ROOT}/${EXP_SUBDIR}"
cd "${EXP_DIR}"

# ---------------------------------------------------------------------------
bold "== 5. Harness smoke test (CPU mock, no weights) =="
# ---------------------------------------------------------------------------
python run_scratchpad_loop.py --mock --probe harmful --n 2 --rounds 2 --modes real \
    >/dev/null 2>&1 && echo "scratchpad loop imports + runs OK" \
    || warn "mock smoke test failed -- inspect before a real run"
rm -rf outputs/mock_harmful_scratchpad_* outputs/mock_harmful 2>/dev/null || true

# ---------------------------------------------------------------------------
bold "== Ready =="
# ---------------------------------------------------------------------------
cat <<EOF

  cd ${EXP_DIR}
  tmux new -s obf     # so a dropped SSH does not kill an 18GB download

Single-shot eval (validates against Sec 9 numbers):
  python run_incontext_eval.py --model IlyaGusev/gemma-2-9b-it-abliterated \\
      --probe harmful --n 50 --batch-size 8

Scratchpad closed-loop (start tiny, inspect scratchpads.jsonl, then scale):
  python run_scratchpad_loop.py --model IlyaGusev/gemma-2-9b-it-abliterated \\
      --probe harmful --n 4 --rounds 4 --modes real

Results land in outputs/ (gitignored) -- tar and scp them off before destroying:
  tar czf ~/obf_results.tar.gz outputs/
EOF
