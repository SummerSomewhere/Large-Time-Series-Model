#!/bin/sh
# Legacy name: previously harmonic gated attention; removed. Use periodic embedding branch instead.
# Run: bash scripts/forecast/ETTh1_periodic_emb.sh
exec sh "$(dirname "$0")/ETTh1_periodic_emb.sh"
