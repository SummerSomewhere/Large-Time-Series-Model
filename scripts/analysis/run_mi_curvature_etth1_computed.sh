#!/bin/sh
#
# Thin wrapper: same as scripts/analysis/mi_patch_curvature_etth1.sh (one-shot MI + plots).
# The ETTh1 script now auto-runs mi_hsic_layerwise_timer.sh when HSIC npy is missing (if checkpoint exists).
#
# Usage:
#   bash scripts/analysis/run_mi_curvature_etth1_computed.sh
#
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/mi_patch_curvature_etth1.sh" "$@"
