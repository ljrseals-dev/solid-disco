#!/usr/bin/env bash
# Phase-1 smoke test for the hybrid pipeline
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== Hybrid VME → MagneticOracle smoke test ==="
python -m bridge.vme_to_oracle \
  --fast \
  --n-notes 24 \
  --seed 42 \
  --out artifacts/wav/smoke_hybrid.wav \
  --diagnostics artifacts/metrics/smoke_diagnostics.json \
  --metrics artifacts/metrics/smoke_vme_metrics.json

echo ""
echo "=== Artifacts ==="
ls -la artifacts/wav/smoke_hybrid.wav artifacts/metrics/smoke_*.json
echo ""
echo "Smoke test finished."
