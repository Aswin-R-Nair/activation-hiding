#!/usr/bin/env bash
# Set up a fresh GPU instance for the in-context obfuscation experiments.
#
# On any new box (vast.ai PyTorch template OR a bare nvidia/cuda:* image), run once:
#
#   curl -fsSL https://raw.githubusercontent.com/Aswin-R-Nair/activation-hiding/incontext-obfuscation/experiments/incontext_obfuscation/setup_gpu.sh | bash
#
# It clones the repo (if not already present), installs deps, verifies the GPU
# is bf16-capable, and runs a CPU mock test to confirm the harness imports.
# Idempotent: safe to re-run.
#
# On a PyTorch template it deliberately does NOT reinstall torch -- the image ships
# a CUDA-matched build, and reinstalling risks landing a CPU-only or arch-mismatched
# wheel (both of which have bitten us before).
#
# On a bare nvidia/cuda:* image there IS no torch, so it installs one, matched to
# the driver's CUDA version and the machine's architecture. Set TORCH_INDEX to
# override the wheel index if the auto-detected one is wrong.

set -euo pipefail

REPO_URL="https://github.com/Aswin-R-Nair/activation-hiding.git"
BRANCH="incontext-obfuscation"
CLONE_DIR="/workspace/neural-chameleons"
EXP_SUBDIR="experiments/incontext_obfuscation"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARN:\033[0m %s\n' "$*"; }
die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# A bare nvidia/cuda:* image ships python3 but usually no `python` alias, and
# sometimes no pip. Resolve an interpreter up front rather than dying on the
# first `python` call. Everything below goes through $PY.
PY="${PY:-}"
if [ -z "$PY" ]; then
    # Require pip too: an image can carry a `python` with no pip while `python3`
    # is the usable one, so take the first interpreter that satisfies BOTH.
    for c in python3 python; do
        command -v "$c" >/dev/null 2>&1 || continue
        "$c" -m pip --version >/dev/null 2>&1 || continue
        PY="$c"; break
    done
fi
if [ -z "$PY" ]; then
    for c in python3 python; do
        command -v "$c" >/dev/null 2>&1 && { FOUND="$c"; break; }
    done
    [ -n "${FOUND:-}" ] && die "found ${FOUND} but it has no pip:
  apt-get update && apt-get install -y python3-pip"
    die "no python found. On a bare CUDA image:
  apt-get update && apt-get install -y python3 python3-pip python3-venv git tmux"
fi
echo "using interpreter: $PY ($("$PY" --version 2>&1))"

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
# A bare CUDA image has no torch. Install one rather than dying -- but pick the
# wheel deliberately: the default PyPI wheel is fine on x86 for older cards and
# WRONG for anything needing a newer CUDA (you get a build with no kernels for
# your arch, which fails at the first matmul, not at import).
if ! python -c "import torch" >/dev/null 2>&1; then
    warn "torch is not installed (bare CUDA image?) -- installing a CUDA build"
    ARCH="$(uname -m)"
    # Driver's max supported CUDA, e.g. "12.8" -> cu128. Blackwell needs >= 12.8.
    DRV_CUDA="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader >/dev/null 2>&1 \
        && nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)"
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu${DRV_CUDA:-128}}"
    echo "arch=${ARCH}  driver CUDA=${DRV_CUDA:-unknown}  index=${TORCH_INDEX}"
    "$PY" -m pip install --quiet --upgrade pip
    "$PY" -m pip install --index-url "${TORCH_INDEX}" torch \
        || die "torch install failed. Pick an index from https://pytorch.org/get-started/locally/ \
and re-run with TORCH_INDEX=<url>. On aarch64 (GB10/Grace) confirm that index \
publishes arm64 wheels -- not all of them do."
fi

"$PY" - <<'PY' || die "torch/bf16 check failed -- see message above."
import sys
try:
    import torch
except ModuleNotFoundError:
    sys.exit("torch still not importable after install -- check the pip output above.")
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
# On new silicon a torch build can import fine, report CUDA available, and still
# have no kernels for this GPU. The arch list ALONE cannot decide this: GB10 is
# sm_121 and cu130 wheels ship sm_120 cubins with no compute_* PTX, yet CUDA 13
# treats Blackwell 12.x as a family, so sm_120 code is expected to run on sm_121.
# So: warn on a missing exact arch, but let an ACTUAL KERNEL LAUNCH be the arbiter.
archs = torch.cuda.get_arch_list()
print("compiled for", archs)
want = f"sm_{major}{minor}"
same_family = [a for a in archs
               if a.startswith("sm_") and a[3:].isdigit() and int(a[3:]) // 10 == major]
has_ptx = any(a.startswith("compute_") for a in archs)
if want not in archs:
    print(f"NOTE: no exact {want} cubin. same-family={same_family} ptx={has_ptx}. "
          "Relying on family compatibility / JIT -- the matmul below decides it.")

# Big enough to dispatch a real cuBLAS kernel; a 64x64 toy can be served by paths
# that would not catch a missing kernel image.
try:
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    val = float((a @ a).float().sum())
    torch.cuda.synchronize()
except Exception as e:
    extra = ""
    if want not in archs:
        extra = (f"\nThis torch has no {want} kernels and "
                 + ("no PTX to JIT from" if not has_ptx else "PTX that did not JIT")
                 + ". Install a torch built for this arch (newer cuXXX index), or "
                 "switch to NVIDIA's NGC PyTorch container for your platform.")
    sys.exit(f"bf16 matmul on {name} failed: {type(e).__name__}: {e}{extra}")
if val != val:  # NaN -> kernels ran but produced garbage
    sys.exit(f"bf16 matmul on {name} returned NaN -- numerically broken build, "
             "do not trust any activations measured on it.")
print(f"bf16 matmul OK (checksum {val:.3e}).")
PY

# ---------------------------------------------------------------------------
bold "== 3. Python deps =="
# ---------------------------------------------------------------------------
# torch is handled in step 2 (kept out of this line so a bad resolve here can
# never pull a different torch in as a dependency).
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet transformers accelerate pandas numpy
echo "installed transformers / accelerate / pandas / numpy"
# hf_transfer is a Rust wheel and has no aarch64 build on some indexes. It only
# speeds downloads, so failing to get it must not stop the setup.
if python -m pip install --quiet hf_transfer 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "hf_transfer OK -- HF_HUB_ENABLE_HF_TRANSFER=1 (add to your shell rc)"
else
    warn "hf_transfer unavailable for this platform -- skipping (downloads just slower)"
fi
# A bare CUDA image often lacks these; the harness and tmux workflow need them.
for c in git tmux; do
    command -v "$c" >/dev/null 2>&1 || warn "$c not found -- apt-get install -y $c"
done

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
    mkdir -p "$(dirname "${CLONE_DIR}")"
    git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${CLONE_DIR}"
    REPO_ROOT="${CLONE_DIR}"
fi

EXP_DIR="${REPO_ROOT}/${EXP_SUBDIR}"
cd "${EXP_DIR}"

# ---------------------------------------------------------------------------
bold "== 5. Harness smoke test (CPU mock, no weights) =="
# ---------------------------------------------------------------------------
"$PY" run_scratchpad_loop.py --mock --probe harmful --n 2 --rounds 2 --modes real \
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
  ${PY} run_incontext_eval.py --model IlyaGusev/gemma-2-9b-it-abliterated \\
      --probe harmful --n 50 --batch-size 8

Scratchpad closed-loop (start tiny, inspect scratchpads.jsonl, then scale):
  ${PY} run_scratchpad_loop.py --model IlyaGusev/gemma-2-9b-it-abliterated \\
      --probe harmful --n 4 --rounds 4 --modes real

Results land in outputs/ (gitignored) -- tar and scp them off before destroying:
  tar czf ~/obf_results.tar.gz outputs/
EOF
