# Hybrid VME ↔ MagneticOracle

**Research integration** of two independent music-intelligence systems:

| Component | Role |
|-----------|------|
| **VME (Config A)** | Symbolic front-end – `DownwashResolver` + Prediction Suffix Tree produces tonally coherent, low-repeat pitch sequences |
| **MagneticOracle** | Physics back-end – magnetic-clapper ODE + resonator driven by a feature-conditional MLP |

This repository does **not** claim historical recovery of a 14th-century neural computation, does **not** claim production musical quality, and keeps the original provenance audit intact.

---

## Quick start

```bash
# from the hybrid_vme_oracle root
pip install -r requirements.txt

# Smoke test (fast, ~8 iterations, ~6 s audio)
python -m bridge.vme_to_oracle --fast --n-notes 32 --seed 42

# Fuller render
python -m bridge.vme_to_oracle --n-notes 48 --seed 42 --iterations 24
```

Outputs land in:
- `artifacts/wav/hybrid.wav`
- `artifacts/metrics/vme_metrics.json`
- `artifacts/metrics/hybrid_diagnostics.json`

---

## Architecture

```
random-walk / latent pitches
        ↓
DownwashResolver (VME Config A)          ← PST + multi-objective costs
        ↓
MIDI-like pitch sequence + metrics
        ↓
bridge.events.events_from_pitches()      ← tempo, velocity, jitter
        ↓
MagneticOracle physics renderer          ← injectable event list
        ↓
WAV + control / impact diagnostics
```

---

## Repository layout

```
hybrid_vme_oracle/
├── bridge/
│   ├── __init__.py
│   ├── events.py              # pitches → timed events + VME generator
│   └── vme_to_oracle.py       # CLI + run_hybrid()
├── vendor/
│   ├── vme_src/               # DownwashResolver + helpers
│   ├── magnetic_oracle/       # original renderer + trainer
│   └── config_a.json          # locked VME weights
├── artifacts/
│   ├── checkpoints/           # feature-conditional & decorrelation .pt
│   ├── wav/
│   └── metrics/
├── configs/
│   └── hybrid_default.json
├── experiments/
│   └── phase1_smoke/
├── docs/
├── requirements.txt
└── README.md
```

---

## What was recovered vs. newly designed

**Recovered / vendored unchanged**
- VME Config A weights and `DownwashResolver`
- MagneticOracle architecture, physics, feature contract, checkpoints from the provenance-audited package

**Newly designed in this hybrid**
- Event bridge (`events_from_pitches`, `pitches_from_vme`)
- Injectable-event path and fast-mode render wrapper
- Combined metrics report
- This repository layout and documentation

The original MagneticOracle silent-training-bug repair, pre-registered experiments, and explicit non-claims about Hermann Poll / historical neural encoding remain exactly as published in the audit paper.

---

## Open research questions (inherited)

1. Physical-render centroid isolation (centroid ↔ ZCR correlation still high on real renders).
2. Full 18 s / 60-iteration musical evaluation.
3. Cross-trajectory generalization of the feature-conditional model.
4. Larger PST corpus for VME.
5. Optional context adapter that lets VME chord / key bias MagneticOracle controls (not implemented yet).

---

## License & provenance

- VME code follows the license present in the original VME distribution.
- MagneticOracle code follows the terms of the audited research package (see its `LICENSE` and the non-claims appendix of the paper).
- Bridge code in this repository is new and may be treated as CC0 / public-domain research scaffolding unless otherwise stated by the authors.

Always keep the original SHA-256 manifests and audit documents alongside any redistribution.

---

*Research integration only. Not a production music engine.*
