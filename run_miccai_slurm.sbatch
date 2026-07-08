#!/bin/bash
# ============================================================================
# MICCAI Pipeline — Parallelized SLURM Launcher (Self-Contained)
# ============================================================================
#
# This is the MASTER script. It runs preprocessing once, then launches
# parallel array jobs — one per encoding. Each encoding trains on ALL
# images independently. After all finish, a final job collects results
# and generates tables/figures.
#
# NO external src/ or configs/ dependencies — all scripts are self-contained.
#
# Architecture:
#   Job 1: Preprocess (serial, ~5 min)
#   Job 2: Train array[0-10] — one job per encoding, each trains all images
#   Job 3: Collect + evaluate + visualize (serial, ~5 min)
#
# Usage:
#   sbatch run_miccai_slurm.sbatch                   # Full pipeline
#   sbatch run_miccai_slurm.sbatch --quick            # Fast test (2k iters)
#   SKIP_PREPROCESS=1 sbatch run_miccai_slurm.sbatch  # Skip preprocess
#
# Estimated time (35 images/modality, 5 modalities, 3500 iters):
#   Serial:    ~12 hours (175 images x 11 encodings x ~4 min each)
#   Parallel:  ~1.5 hours (11 encoding jobs run simultaneously)
#
# ============================================================================

#SBATCH --job-name=miccai_launcher
#SBATCH --output=logs/launcher_%j.out
#SBATCH --error=logs/launcher_%j.err
#SBATCH --partition=pascalnodes-medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00

# ============================================================================
# Configuration — EDIT THESE TO MATCH YOUR CLUSTER
# ============================================================================
PROJECT_DIR="."  # Auto-detect: folder where this script lives
LOG_DIR="logs"
#CONDA_ENV="tropical_ct"            # Your conda environment name
PARTITION="pascalnodes-medium"             # GPU partition
QUICK="${1:-}"                      # Pass --quick as first arg for fast test

# ============================================================================
# Setup
# ============================================================================
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  MICCAI Pipeline — Parallel Launcher"
echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Project:     $PROJECT_DIR"
echo "Start time:  $(date)"
echo "Quick mode:  ${QUICK:-no}"
echo "=========================================="

#ml Anaconda3/2023.07-2
#conda activate "$CONDA_ENV"

#cd "$PROJECT_DIR"

# ============================================================================
# Step 1: Preprocess (serial — only needs CPU)
# ============================================================================
if [ -z "$SKIP_PREPROCESS" ]; then
    echo ""
    echo "[STEP 1/3] Submitting PREPROCESS job..."

    PREPROCESS_JOB=$(sbatch --parsable \
        --job-name=miccai_preprocess \
        --output="${LOG_DIR}/preprocess_%j.out" \
        --error="${LOG_DIR}/preprocess_%j.err" \
        --partition="$PARTITION" \
        --cpus-per-task=8 \
        --mem=32G \
        --gres=gpu:4 \
        --time=01:00:00 \
        --wrap="
     
            python -u run_miccai.py --preprocess-only $QUICK
            echo 'Preprocess done.'
        ")

    echo "  Preprocess job: $PREPROCESS_JOB"
    DEPEND="--dependency=afterok:${PREPROCESS_JOB}"
else
    echo "[STEP 1/3] Skipping preprocess (SKIP_PREPROCESS=1)"
    DEPEND=""
fi

# ============================================================================
# Step 2: Training — one array job per ENCODING (parallel)
# ============================================================================
# Encodings indexed 0-10:
#   0=none, 1=fourier, 2=tropical, 3=hybrid, 4=trapezoid,
#   5=square, 6=rampup, 7=rampdown, 8=siren, 9=wavelet, removed --> 10=hash
#
# Each array task trains ONE encoding on ALL images.
# ============================================================================

echo ""
echo "[STEP 2/3] Submitting TRAINING array job (11 encodings in parallel)..."

QUICK_FLAG=""
if [ "$QUICK" == "--quick" ]; then
    QUICK_FLAG="--quick"
    TIME_LIMIT="01:00:00"
else
    TIME_LIMIT="04:00:00"   # 3500 iters ~4x faster than 15k
fi

TRAIN_JOB=$(sbatch --parsable \
    --job-name=miccai_train \
    --output="${LOG_DIR}/train_%A_%a.out" \
    --error="${LOG_DIR}/train_%A_%a.err" \
    --partition="$PARTITION" \
    --array=0-10 \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --time="$TIME_LIMIT" \
    $DEPEND \
    --wrap="

        # Map array index to encoding name
        ENCODINGS=(none fourier tropical hybrid trapezoid square rampup rampdown siren wavelet hash)
        ENC=\${ENCODINGS[\$SLURM_ARRAY_TASK_ID]}

        echo '=========================================='
        echo \"  Training encoding: \$ENC\"
        echo \"  Array task: \$SLURM_ARRAY_TASK_ID / 10\"
        echo \"  Node: \$(hostname)\"
        echo \"  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')\"
        echo '=========================================='

        python -u run_miccai_worker.py \\
            --encoding \$ENC \\
            --split train \\
            $QUICK_FLAG

        echo \"Encoding \$ENC (train) complete.\"
    ")

echo "  Training array job: $TRAIN_JOB (array 0-10)"

# ============================================================================
# Step 2b: Held-out test split (parallel, depends on train)
# ============================================================================

echo ""
echo "[STEP 2b/3] Submitting HELD-OUT test array job..."

HELDOUT_JOB=$(sbatch --parsable \
    --job-name=miccai_heldout \
    --output="${LOG_DIR}/heldout_%A_%a.out" \
    --error="${LOG_DIR}/heldout_%A_%a.err" \
    --partition="$PARTITION" \
    --array=0-10 \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --time="$TIME_LIMIT" \
    --dependency=afterok:${TRAIN_JOB} \
    --wrap="

        ENCODINGS=(none fourier tropical hybrid trapezoid square rampup rampdown siren wavelet hash)
        ENC=\${ENCODINGS[\$SLURM_ARRAY_TASK_ID]}

        echo \"  Held-out eval for encoding: \$ENC\"

        python -u run_miccai_worker.py \\
            --encoding \$ENC \\
            --split test \\
            $QUICK_FLAG

        echo \"Encoding \$ENC (test) complete.\"
    ")

echo "  Held-out array job: $HELDOUT_JOB (array 0-10)"

# ============================================================================
# Step 3: Collect results + evaluation + figures (serial)
# ============================================================================

echo ""
echo "[STEP 3/3] Submitting COLLECT + EVALUATE + VISUALIZE job..."

COLLECT_JOB=$(sbatch --parsable \
    --job-name=miccai_collect \
    --output="${LOG_DIR}/collect_%j.out" \
    --error="${LOG_DIR}/collect_%j.err" \
    --partition="$PARTITION" \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --time=01:00:00 \
    --dependency=afterok:${HELDOUT_JOB} \
    --wrap="
        echo '=========================================='
        echo '  Collecting results + generating tables/figures'
        echo '=========================================='
        python -u run_miccai_collect.py
        echo 'Collection complete. Check results/miccai/'
    ")

echo "  Collect job: $COLLECT_JOB"

# ============================================================================
# Step 4: Downstream tasks (super-resolution, segmentation, inpainting)
# Runs after collect so reconstructions_train/ is fully populated
# ============================================================================

echo ""
echo "[STEP 4/4] Submitting DOWNSTREAM TASKS job..."

DOWNSTREAM_JOB=$(sbatch --parsable \
    --job-name=miccai_downstream \
    --output="${LOG_DIR}/downstream_%j.out" \
    --error="${LOG_DIR}/downstream_%j.err" \
    --partition="$PARTITION" \
    --cpus-per-task=8 \
    --mem=32G \
    --gres=gpu:1 \
    --time=04:00:00 \
    --dependency=afterok:${COLLECT_JOB} \
    --wrap="
        echo '=========================================='
        echo '  Downstream tasks (SR / segmentation / inpainting)'
        echo '=========================================='
        python -u run_miccai.py \
            --skip-preprocess \
            --skip-training \
            --skip-heldout \
            $QUICK_FLAG
        echo 'Downstream tasks complete.'
    ")

echo "  Collect job: $COLLECT_JOB"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo "  ALL JOBS SUBMITTED"
echo "=========================================="
echo ""
echo "  Preprocess:  ${PREPROCESS_JOB:-skipped}"
echo "  Training:    $TRAIN_JOB (array 0-10, 11 parallel GPU jobs, 3500 iters)"
echo "  Held-out:    $HELDOUT_JOB (array 0-10, after training)"
echo "  Collect:     $COLLECT_JOB (after all training done)"
echo "  Downstream:  $DOWNSTREAM_JOB (SR / segmentation / inpainting, after collect)"
echo ""
echo "  Monitor with:"
echo "    squeue -u \$USER"
echo "    tail -f ${LOG_DIR}/train_${TRAIN_JOB}_*.out"
echo ""
echo "  Cancel all with:"
echo "    scancel $TRAIN_JOB $HELDOUT_JOB $COLLECT_JOB"
echo ""
echo "  Total parallel speedup: ~10x (11 encodings running simultaneously)"
echo "=========================================="
