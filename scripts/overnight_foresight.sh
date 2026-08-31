#!/usr/bin/env bash
# Overnight queue for the linear-visual-foresight investigation.
#
# Runs SEQUENTIALLY on purpose: these are all GPU-bound, and running two
# collections at once made both slower than either alone. Cheap, decisive work
# is queued first so that a crash late in the night still leaves the most
# valuable results on disk. Every stage writes incrementally, so a stage killed
# part-way still contributes whatever batches it finished.
#
# Read docs/linear_foresight_findings.md for what these are testing.
#
#   bash scripts/overnight_foresight.sh          # logs to outputs/overnight/
#
set -u   # NOT -e: a failing stage must not abort the night.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/outputs/overnight"
mkdir -p "$LOG"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo "=== $(stamp)  $*" | tee -a "$LOG/00_timeline.log"; }

banner "overnight queue starting"

# ---------------------------------------------------------------------------
# Stage 0 (~5 min) -- dense-pile probe.
# Validates the 3 mm / 150-particle geometry AND the contact budget before
# committing the night to it. A dense multi-layer pile generates far more
# contacts than the scattered-tuned default of max(150, n/2); past that cap
# Genesis silently drops contacts, so a run that overflows produces physically
# wrong data with no error. The WARNING lines in this log are the check.
# ---------------------------------------------------------------------------
banner "stage 0: dense-pile probe (150 x 3mm, budget check)"
timeout 20m python -u -m Genesis.data_collection_clean \
    --num-particles 150 --particle-sizes 0.003 --particle-shape cube \
    --particle-density 4000.0 \
    --n-envs 4 --samples-per-env 2 --n-batches 1 --state-library 1 \
    --state-library-damping 15.0 --max-collision-pairs 600 \
    --output-root data/foresight/probe_dense --constant-params \
    --pile-extent 0.021 --pile-aware-actions --push-length 0.02 \
    --min-swath-particles 3 --seed 1 --debug \
    > "$LOG/10_probe_dense.log" 2>&1
banner "stage 0 done (exit $?)"
grep -iE "WARNING|contact budget|pile spawn|swath holds" "$LOG/10_probe_dense.log" \
    | tail -20 | tee -a "$LOG/00_timeline.log"

# ---------------------------------------------------------------------------
# Stage 1 (~20 min) -- the attribution control.
# The headline result (contact-aware sampling linearises the dynamics) was
# established by changing the pile AND the sampling together, then attributed
# to sampling by inference. This is the missing factorial cell: contact-aware
# sampling on ORDINARY SCATTERED geometry. Scattered collection is ~100x
# cheaper per transition, so this is the best value in the queue.
# ---------------------------------------------------------------------------
banner "stage 1: attribution control (scattered geometry + contact-aware pushes)"
timeout 60m python -u -m Genesis.data_collection_clean \
    --num-particles 50 --particle-sizes 0.005 --particle-shape cube \
    --n-envs 32 --samples-per-env 2 --n-batches 40 --state-library 4 \
    --state-library-damping 15.0 \
    --output-root data/foresight/scatter_contact --constant-params \
    --pile-aware-actions --push-length 0.02 --min-swath-particles 3 --seed 31 \
    > "$LOG/20_scatter_contact.log" 2>&1
banner "stage 1 done (exit $?)"

# ---------------------------------------------------------------------------
# Stage 2 (~45 min) -- 30-cube pile at the corrected 20 mm push length.
# Gives the push-length comparison against the existing 40 mm piled dataset,
# and is the first piled data collected with the start-clamp fix.
# ---------------------------------------------------------------------------
banner "stage 2: piled 30 x 5mm, 20mm pushes (clamped starts)"
timeout 75m python -u -m Genesis.data_collection_clean \
    --num-particles 30 --particle-sizes 0.005 --particle-shape cube \
    --n-envs 32 --samples-per-env 2 --n-batches 30 --state-library 2 \
    --state-library-damping 15.0 \
    --output-root data/foresight/pile30_L020 --constant-params \
    --pile-extent 0.015 --pile-aware-actions --push-length 0.02 \
    --min-swath-particles 3 --seed 23 \
    > "$LOG/30_pile30_L020.log" 2>&1
banner "stage 2 done (exit $?)"

# ---------------------------------------------------------------------------
# Stage 3 (rest of the night) -- H1, the continuum-limit experiment.
# 150 cubes of 3 mm in a compact spawn. Smaller cubes are the point: the same
# material volume gives MORE layers (4 vs 2) and a smoother occupancy field,
# which is the leading explanation for why the paper's per-pixel operator
# worked and ours did not. n_envs is low because the collision budget is high.#
# NOTE on particle density: 3 mm cubes at the project default of 1000 kg/m^3 have
# a mass of 1.6e-5 kg, which Genesis warns is "too small for the constraint
# solver to be numerically stable" -- a warning that appears in NO 5 mm run, so
# it is specific to shrinking the cubes. Running a whole night on unstable
# contact physics is exactly the silent corruption to avoid, so density is
# raised to 4000 kg/m^3. That puts the mass (1.1e-4 kg) back alongside the 5 mm
# cubes at 1000 that every earlier dataset used, and 4000 sits inside this
# project's own physics normalisation range (750-5000), so it is
# in-distribution rather than an exotic setting.
# n_batches is deliberately larger than the night can finish -- it writes
# incrementally, so it simply uses whatever time is left.
# ---------------------------------------------------------------------------
banner "stage 3: H1 continuum limit (150 x 3mm compact pile)"
timeout 6h python -u -m Genesis.data_collection_clean \
    --num-particles 150 --particle-sizes 0.003 --particle-shape cube \
    --particle-density 4000.0 \
    --n-envs 8 --samples-per-env 2 --n-batches 400 --state-library 3 \
    --state-library-damping 15.0 --max-collision-pairs 600 \
    --output-root data/foresight/pile150_d3 --constant-params \
    --pile-extent 0.021 --pile-aware-actions --push-length 0.02 \
    --min-swath-particles 3 --seed 47 \
    > "$LOG/40_pile150_d3.log" 2>&1
banner "stage 3 done (exit $?)"

# ---------------------------------------------------------------------------
# Stage 4 -- analyse everything collected. Runs regardless of what the earlier
# stages managed, on whatever is on disk. variance_decomposition and
# deltav_predictability take a raw file glob, so they need no dataset config and
# work on partial collections.
#
# The number to look at in each block is the LINEAR SHARE (linear R2 / boosted
# R2) on the OCC feature set. Scattered+blind was 69%; piled+contact-sampled was
# 92%. See docs/linear_foresight_findings.md section 3.
# ---------------------------------------------------------------------------
banner "stage 4: analysis"

analyse () {   # $1 = label, $2 = glob
    local n
    n=$(ls $2 2>/dev/null | wc -l)
    if [ "$n" -lt 4 ]; then
        banner "  skip $1: only $n files (need >= 4 for grouped CV)"
        return
    fi
    banner "  analysing $1 ($n files)"
    timeout 25m python -u variance_decomposition.py --glob "$2" --label "$1" \
        >> "$LOG/50_analysis.log" 2>&1
    timeout 25m python -u deltav_predictability.py --glob "$2" --label "$1" \
        >> "$LOG/50_analysis.log" 2>&1
}

analyse "SCATTERED + contact-aware pushes (attribution control)" \
        "Genesis/data/foresight/scatter_contact/cube/n50/size0.005/_*_data.pt"
analyse "PILED 30x5mm, 20mm pushes" \
        "Genesis/data/foresight/pile30_L020/cube/n30/size0.005/_*_data.pt"
analyse "H1: PILED 150x3mm (continuum limit)" \
        "Genesis/data/foresight/pile150_d3/cube/n150/size0.003/_*_data.pt"
analyse "PILED 30x5mm, 40mm pushes (reference, pre-clamp)" \
        "Genesis/data/foresight/pile30/cube/n30/size0.005/_*_data.pt"

banner "stage 4 done"
echo "" | tee -a "$LOG/00_timeline.log"
echo "RESULTS: $LOG/50_analysis.log    TIMELINE: $LOG/00_timeline.log" \
    | tee -a "$LOG/00_timeline.log"

banner "overnight queue finished"
