#!/usr/bin/env bash
# E2C Data Preparation - One-command script to prepare all data
# Can be run from scripts directory

set -eo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Get project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

# Defaults
DOWNLOAD_DATA=true
PREPARE_SFT=true
PREPARE_RL=true
NUM_WORKERS=8
USE_MIRROR=false

# Helper functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Help
show_help() {
    cat << EOF
E2C Data Preparation Script

Usage: $0 [options]

Options:
    --skip-download       Skip data download
    --skip-sft            Skip SFT data preparation
    --skip-rl             Skip RL data preparation
    --num-workers N       Number of parallel workers (default: 8)
    --mirror              Use mirror source (hf-mirror.com) for faster downloads
    -h, --help            Show this help

Examples:
    bash $0                    # Prepare all data
    bash $0 --mirror           # Use mirror source for faster downloads
    bash $0 --skip-download    # Skip download, process local data only
    bash $0 --skip-rl          # Prepare SFT data only
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download) DOWNLOAD_DATA=false; shift ;;
        --skip-sft) PREPARE_SFT=false; shift ;;
        --skip-rl) PREPARE_RL=false; shift ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --mirror) USE_MIRROR=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) error "Unknown argument: $1"; show_help; exit 1 ;;
    esac
done

# Configuration
info "===== E2C Data Preparation ====="
info "Project root:   ${PROJECT_ROOT}"
info "Data directory: ${DATA_DIR}"
info "Download data:  ${DOWNLOAD_DATA}"
info "Use mirror:     ${USE_MIRROR}"
info "Prepare SFT:    ${PREPARE_SFT}"
info "Prepare RL:     ${PREPARE_RL}"
info "Workers:        ${NUM_WORKERS}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    error "python3 not found. Please install Python first."
    exit 1
fi

# Check dependencies
info "Checking Python dependencies..."
python3 -c "import pandas, datasets, tqdm" 2>/dev/null || {
    warn "Missing dependencies. Installing..."
    pip install pandas datasets tqdm -q
}
success "Dependencies OK"
echo ""

# Step 1: Download data
if [ "$DOWNLOAD_DATA" = true ]; then
    info "Step 1/3: Downloading raw data..."
    DOWNLOAD_CMD="${SCRIPT_DIR}/download_datasets.sh"
    if [ "$USE_MIRROR" = true ]; then
        DOWNLOAD_CMD="${DOWNLOAD_CMD} --mirror"
    fi
    bash ${DOWNLOAD_CMD} || {
        error "Data download failed"
        exit 1
    }
    success "Data downloaded"
    echo ""
else
    warn "Step 1/3: Skipping data download"
    echo ""
fi

# Step 2: Prepare SFT data
if [ "$PREPARE_SFT" = true ]; then
    info "Step 2/3: Preparing SFT training data..."
    
    SFT_INPUT_DIR="${DATA_DIR}/raw/sft"
    SFT_OUTPUT_DIR="${DATA_DIR}/processed/sft"
    
    if [ ! -d "${SFT_INPUT_DIR}" ] || [ -z "$(ls -A ${SFT_INPUT_DIR} 2>/dev/null)" ]; then
        warn "SFT input directory empty, skipping"
    else
        mkdir -p "${SFT_OUTPUT_DIR}"
        python3 "${DATA_DIR}/prepare_sft_data.py" \
            --input_dir "${SFT_INPUT_DIR}" \
            --output_dir "${SFT_OUTPUT_DIR}" \
            --train_ratio 0.95 \
            --num_workers "${NUM_WORKERS}" || {
            error "SFT data preparation failed"
            exit 1
        }
        success "SFT data ready"
        info "  → ${SFT_OUTPUT_DIR}/e2c-sft-train.parquet"
        info "  → ${SFT_OUTPUT_DIR}/e2c-sft-val.parquet"
    fi
    echo ""
else
    warn "Step 2/3: Skipping SFT data preparation"
    echo ""
fi

# Step 3: Prepare RL data
if [ "$PREPARE_RL" = true ]; then
    info "Step 3/3: Preparing RL training data..."
    
    RL_INPUT_DIR="${DATA_DIR}/raw/rl"
    RL_OUTPUT_DIR="${DATA_DIR}/processed/rl"
    
    if [ ! -d "${RL_INPUT_DIR}" ] || [ -z "$(ls -A ${RL_INPUT_DIR} 2>/dev/null)" ]; then
        warn "RL input directory empty, skipping"
    else
        mkdir -p "${RL_OUTPUT_DIR}"
        python3 "${DATA_DIR}/prepare_rl_data.py" \
            --input_dir "${RL_INPUT_DIR}" \
            --output_dir "${RL_OUTPUT_DIR}" \
            --train_ratio 0.95 \
            --num_workers "${NUM_WORKERS}" || {
            error "RL data preparation failed"
            exit 1
        }
        success "RL data ready"
        info "  → ${RL_OUTPUT_DIR}/e2c-rl-train.parquet"
        info "  → ${RL_OUTPUT_DIR}/e2c-rl-val.parquet"
    fi
    echo ""
else
    warn "Step 3/3: Skipping RL data preparation"
    echo ""
fi

# Summary
success "===== Data Preparation Complete ====="
echo ""
info "Directory structure:"
echo "  ${DATA_DIR}/"
echo "  ├── raw/              # Raw downloaded data"
echo "  └── processed/        # Processed training data"
echo "      ├── sft/          # SFT training data"
echo "      └── rl/           # RL training data"
echo ""

info "Next steps:"
if [ "$PREPARE_SFT" = true ] && [ -f "${DATA_DIR}/processed/sft/e2c-sft-train.parquet" ]; then
    echo "  1. SFT training:"
    echo "     bash scripts/e2c_sft.sh"
fi

if [ "$PREPARE_RL" = true ] && [ -f "${DATA_DIR}/processed/rl/e2c-rl-train.parquet" ]; then
    echo "  2. RL training:"
    echo "     bash scripts/e2c_rl.sh"
fi

echo ""
success "Done! 🎉"

