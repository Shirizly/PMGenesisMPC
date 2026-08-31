#!/usr/bin/env bash
# Check for, and fix, the nvrtc version clash that breaks torch's JIT compiler
# in this project's conda environment.
#
#   bash scripts/fix_nvrtc.sh          check, and install the fix if needed
#   bash scripts/fix_nvrtc.sh --check  report only, change nothing
#
# Safe to re-run: it is idempotent and does nothing when the environment is
# already healthy.
#
# THE PROBLEM
# -----------
# genesis-world pulls in `nvidia-cuda-nvrtc-cu12` (via gs-madrona, the
# renderer), while torch ships a CUDA 13 build. Both nvrtc versions end up
# installed under site-packages/nvidia/. torch loads the cu12 one, then asks for
# `libnvrtc-builtins.so.13.0`, which lives in `nvidia/cu13/lib` — a directory on
# nobody's library search path. `libnvrtc.so.13` carries no RUNPATH, so a
# same-directory sibling is not found automatically either. Every torch JIT
# compilation then dies with:
#
#   RuntimeError: nvrtc: error: failed to open libnvrtc-builtins.so.13.0
#
# Neither package can simply be removed: gs-madrona requires the cu12 one, and
# torch needs the cu13 one.
#
# It is easy to miss, because it only fires on code paths that actually invoke
# the JIT. It was found through `record_simulation_video.py`: `geom.z_up_to_R`
# is torch-scripted and only runs for a camera at an ANGLE, so a top-down
# camera rendered fine and adding a three-quarter one killed the run.
#
# THE FIX
# -------
# Prepend the directory holding the correct builtins to LD_LIBRARY_PATH. That
# has to happen before the process starts — the dynamic loader reads it once, so
# setting it from inside Python is too late (preloading the library with
# ctypes.CDLL was tried and does not work: this lookup does not consult
# already-loaded objects). So it is installed as a conda activate.d hook, which
# makes it automatic for anyone who activates the environment, contained to that
# environment, and removed by the matching deactivate.d hook.
set -euo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: no conda environment is active. Run 'conda activate <env>' first."
    exit 2
fi

PY=$(command -v python)
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
TORCH_CUDA=$("$PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)

if [[ -z "$TORCH_CUDA" ]]; then
    echo "torch reports no CUDA build (CPU-only?). Nothing to fix."
    exit 0
fi
MAJOR=${TORCH_CUDA%%.*}
echo "torch CUDA build : $TORCH_CUDA"

# The builtins torch will ask for, e.g. libnvrtc-builtins.so.13.0
BUILTINS=$(find "$SITE/nvidia" -name "libnvrtc-builtins.so.${MAJOR}.*" 2>/dev/null \
           | grep -v '\.alt\.' | head -1 || true)
if [[ -z "$BUILTINS" ]]; then
    echo "ERROR: no libnvrtc-builtins.so.${MAJOR}.* found under $SITE/nvidia."
    echo "       torch needs a CUDA ${MAJOR} nvrtc. Try:"
    echo "         pip install --upgrade nvidia-cuda-nvrtc"
    exit 3
fi
NVRTC_DIR=$(dirname "$BUILTINS")
echo "needed builtins  : $BUILTINS"

# Does a fresh process actually manage the JIT compile?
#
# The probe must live in a real FILE, not a stdin heredoc: torch.jit.script
# calls inspect.getsourcelines, which cannot read a script piped through stdin
# and fails with "Can't get source" for reasons unrelated to nvrtc. An earlier
# version did exactly that and so always reported BROKEN.
PROBE_PY=$(mktemp -t nvrtc_probe_XXXXXX.py)
trap 'rm -f "$PROBE_PY"' EXIT
cat > "$PROBE_PY" <<'PY'
import torch

# abs+lt is the op pair whose fused kernel the real failure reported.
@torch.jit.script
def _probe(x):
    return (x.abs() < 0.5).float().sum()

try:
    x = torch.randn(64, device="cuda")
    for _ in range(5):          # fusion only kicks in after a few calls
        _probe(x)
    torch.cuda.synchronize()
    print("OK")
except Exception:
    print("BROKEN")
PY

probe() { "$PY" "$PROBE_PY" 2>/dev/null; }

STATUS=$(probe || echo BROKEN)
if [[ "$STATUS" == "OK" ]]; then
    echo "torch JIT        : working — no fix needed"
    exit 0
fi
echo "torch JIT        : BROKEN (nvrtc cannot find its builtins)"

HOOK="$CONDA_PREFIX/etc/conda/activate.d/zz-nvrtc-path.sh"
UNHOOK="$CONDA_PREFIX/etc/conda/deactivate.d/zz-nvrtc-path.sh"
if [[ "$CHECK_ONLY" == "1" ]]; then
    echo
    echo "--check: no changes made. Re-run without --check to install:"
    echo "  $HOOK"
    exit 1
fi

mkdir -p "$(dirname "$HOOK")" "$(dirname "$UNHOOK")"
cat > "$HOOK" <<HOOKEOF
# Installed by scripts/fix_nvrtc.sh — see that file for why.
# torch's CUDA $MAJOR build needs libnvrtc-builtins.so.${MAJOR}.*, which lives in a
# directory nothing otherwise puts on the loader path.
export _NVRTC_PATH_SAVED="\${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NVRTC_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
HOOKEOF
cat > "$UNHOOK" <<'HOOKEOF'
# Installed by scripts/fix_nvrtc.sh
if [[ -n "${_NVRTC_PATH_SAVED+x}" ]]; then
    export LD_LIBRARY_PATH="$_NVRTC_PATH_SAVED"
    unset _NVRTC_PATH_SAVED
    [[ -z "$LD_LIBRARY_PATH" ]] && unset LD_LIBRARY_PATH
fi
HOOKEOF
echo "installed hook   : $HOOK"

# Verify in a shell that has the hook applied, rather than claiming success.
# Must be a subshell with an explicit export: in bash `VAR=x some_function`
# does not export VAR to processes the function starts, so the python probe
# would not see it and this check would always fail.
STATUS=$(export LD_LIBRARY_PATH="$NVRTC_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; probe || echo BROKEN)
if [[ "$STATUS" == "OK" ]]; then
    echo "verified         : torch JIT now compiles"
    echo
    echo "Re-activate the environment for it to take effect in this shell:"
    echo "  conda deactivate && conda activate $(basename "$CONDA_PREFIX")"
    exit 0
fi
echo "ERROR: the hook was installed but torch JIT still fails. Report this."
exit 4
