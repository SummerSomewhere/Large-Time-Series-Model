#!/bin/sh
# Legacy name: previously Resonance Head; now uses periodic embedding residual + strict freeze.
# Prefer: bash scripts/forecast/ETTh1_periodic_emb.sh
exec sh "$(dirname "$0")/ETTh1_periodic_emb.sh"
