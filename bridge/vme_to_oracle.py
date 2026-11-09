#!/usr/bin/env python3
"""Hybrid runner: VME Config-A pitches → MagneticOracle physics renderer.

Usage
-----
  python -m bridge.vme_to_oracle --n-notes 32 --seed 42
  python -m bridge.vme_to_oracle --checkpoint artifacts/checkpoints/magnetic_oracle_feature_conditional_retrain.pt \\
                                 --out artifacts/wav/hybrid_seed42.wav --fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup so both vendor packages are importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
sys.path.insert(0, str(VENDOR / "vme_src"))
sys.path.insert(0, str(VENDOR / "magnetic_oracle"))
sys.path.insert(0, str(ROOT))

from bridge.events import events_from_pitches, pitches_from_vme  # noqa: E402


def _load_renderer():
    """Import the MagneticOracle renderer module after path setup."""
    import render_magnetic_clavecin as rmc  # type: ignore
    return rmc


def run_hybrid(
    n_notes: int = 48,
    seed: int = 42,
    tempo: float = 100.0,
    velocity: float = 0.72,
    checkpoint: str | Path | None = None,
    out_wav: str | Path = "artifacts/wav/hybrid.wav",
    diagnostics_path: str | Path | None = "artifacts/metrics/hybrid_diagnostics.json",
    metrics_path: str | Path | None = "artifacts/metrics/vme_metrics.json",
    fast: bool = False,
    iterations: int | None = None,
) -> dict:
    """
    Full hybrid pipeline.

    Returns a combined report dict containing VME metrics + Oracle diagnostics.
    """
    rmc = _load_renderer()

    # 1. Generate VME pitches
    pitches, vme_metrics = pitches_from_vme(n_notes=n_notes, seed=seed)

    # 2. Convert to timed events
    events = events_from_pitches(
        pitches,
        tempo=tempo,
        default_duration_beats=0.45,
        velocity=velocity,
        seed=seed,
    )

    # 3. Load checkpoint
    if checkpoint is None:
        checkpoint = ROOT / "artifacts" / "checkpoints" / "magnetic_oracle_feature_conditional_retrain.pt"
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        # fallback to decorrelation checkpoint
        alt = ROOT / "artifacts" / "checkpoints" / "magnetic_oracle_feature_decorrelation.pt"
        if alt.exists():
            checkpoint = alt
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint}")

    model, metadata = rmc.load_checkpoint(str(checkpoint))
    model._checkpoint_path = str(checkpoint)

    # 4. Render (patched simulate that accepts external events)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if diagnostics_path:
        Path(diagnostics_path).parent.mkdir(parents=True, exist_ok=True)

    n_iter = iterations if iterations is not None else (8 if fast else 24)
    diagnostics = _simulate_with_events(
        rmc,
        model,
        events,
        out_path=out_wav,
        diagnostics_path=diagnostics_path,
        seed=seed,
        n_iterations=n_iter,
        fast=fast,
    )
    diagnostics["objective"] = metadata.get("objective", "unknown")
    diagnostics["checkpoint_metadata"] = {k: metadata[k] for k in metadata if k != "state_dict"}

    # 5. Persist VME metrics
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).write_text(json.dumps(vme_metrics, indent=2) + "\n", encoding="utf-8")

    report = {
        "vme_metrics": vme_metrics,
        "oracle_diagnostics": diagnostics,
        "n_events": len(events),
        "first_16_pitches": pitches[:16],
        "wav": str(out_wav),
        "checkpoint": str(checkpoint),
        "fast": fast,
        "iterations": n_iter,
    }
    return report


def _simulate_with_events(
    rmc,
    model,
    events,
    out_path,
    diagnostics_path=None,
    seed=1729,
    n_iterations=24,
    fast=False,
):
    """
    Lightweight wrapper around the original simulate logic that injects
    an external event list and supports a reduced iteration count for smoke tests.
    """
    import torch
    from scipy.integrate import solve_ivp
    from scipy.io.wavfile import write

    np.random.seed(seed)
    torch.manual_seed(seed)

    FS = getattr(rmc, "FS", 44100)
    DUR = getattr(rmc, "DUR", 18.0)
    T = getattr(rmc, "T", np.linspace(0, DUR, int(FS * DUR)))
    PHI = getattr(rmc, "PHI", (1 + 5**0.5) / 2)
    LUNAR_PHASE = getattr(rmc, "LUNAR_PHASE", np.linspace(0, 2 * np.pi, len(T)))
    FREQ_SCALE = getattr(rmc, "FREQ_SCALE", None)
    if FREQ_SCALE is None:
        # C-major × golden-ratio construction fallback
        base = 261.63
        FREQ_SCALE = [base * (2 ** (i / 12.0)) * (PHI ** (i % 3 * 0.01)) for i in range(24)]

    # Fast mode: shorter duration & looser ODE tolerances
    if fast:
        DUR = min(DUR, 6.0)
        T = np.linspace(0, DUR, int(FS * DUR))
        LUNAR_PHASE = np.linspace(0, 2 * np.pi, len(T))
        rtol, atol = 1e-3, 1e-5
    else:
        rtol, atol = 1e-5, 1e-7

    # Scale event times into the render duration if they exceed it
    max_t = max(e[0] for e in events) if events else 0.0
    if max_t > DUR * 0.95 and max_t > 0:
        scale = (DUR * 0.92) / max_t
        events = [(t * scale, note, vel) for t, note, vel in events]

    audio = np.zeros_like(T)
    all_envs = []
    previous_spectrum = None
    output_history = []
    impact_count = 0
    controls_history = []

    for iteration in range(n_iterations):
        audio_slice = audio[:4096] if audio.size >= 4096 else audio
        features, previous_spectrum = rmc.extract_features(audio_slice, previous_spectrum)
        controls, normalized_outputs = rmc.controls_for(model, features)
        output_history.append(normalized_outputs)
        controls_history.append(controls)

        field_boost = controls["field_boost"]
        snap = controls["snap"]
        global_field = np.zeros_like(T)
        for e in all_envs:
            global_field += e * 0.1

        for t0, note, vel in events:
            idx = int(t0 * FS)
            if idx >= len(T):
                continue
            env = np.zeros_like(T)
            rise = int(0.12 * FS)
            end_rise = min(idx + rise, len(T))
            rise_len = end_rise - idx
            if rise_len > 0:
                env[idx:end_rise] = np.linspace(0, vel * field_boost, rise_len) ** (PHI * snap)
            if end_rise < len(T):
                env[end_rise:] = vel * np.exp(-T[end_rise:] / max(controls["release_time"], 1e-6))
            env *= 1 + 0.2 * np.sin(2 * np.pi * T * PHI)
            env += global_field
            target = 0.1 - 0.002 * (note % 12)

            sol = solve_ivp(
                rmc.clapper_ode,
                [0, DUR],
                [0, 0],
                t_eval=T,
                args=(env, target, LUNAR_PHASE, snap),
                rtol=rtol,
                atol=atol,
            )
            x = sol.y[0]
            v = sol.y[1]
            thr = 0.03 * controls["impact_thr"]
            impacts = np.where(np.diff(np.sign(x - thr)) > 0)[0]
            impacts = impacts + np.random.randint(-2, 2, size=len(impacts))
            impacts = impacts[(impacts > 0) & (impacts < len(T))]
            impact_count += int(len(impacts))
            for imp in impacts:
                vel_imp = abs(v[imp])
                note_idx = note % len(FREQ_SCALE)
                ring = rmc.resonator(
                    vel_imp,
                    FREQ_SCALE[note_idx],
                    controls["decay_scale"],
                    controls["pitch_drift"],
                )
                length = min(len(ring), len(audio) - imp)
                if length > 0:
                    audio[imp : imp + length] += ring[:length]
            if iteration == n_iterations - 1:
                all_envs.append(env)

    peak_before_normalize = float(np.max(np.abs(audio)))
    audio = audio / (peak_before_normalize + 1e-9)
    write(str(out_path), FS, (audio * 32767).astype(np.int16))

    control_means = {
        k: float(np.mean([row[k] for row in controls_history]))
        for k in controls_history[0]
    } if controls_history else {}

    diagnostics = {
        "checkpoint": str(getattr(model, "_checkpoint_path", "")),
        "sample_rate": FS,
        "duration_seconds": float(DUR),
        "iterations": n_iterations,
        "event_count": len(events),
        "impact_count": impact_count,
        "peak_before_normalize": peak_before_normalize,
        "normalized_output_min": np.min(output_history, axis=0).tolist() if output_history else [],
        "normalized_output_max": np.max(output_history, axis=0).tolist() if output_history else [],
        "normalized_output_mean": np.mean(output_history, axis=0).tolist() if output_history else [],
        "control_mean": control_means,
        "wav": str(out_path),
        "seed": seed,
        "fast": fast,
    }
    if diagnostics_path:
        Path(diagnostics_path).write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    return diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="VME Config-A → MagneticOracle hybrid renderer"
    )
    parser.add_argument("--n-notes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tempo", type=float, default=100.0)
    parser.add_argument("--velocity", type=float, default=0.72)
    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT / "artifacts" / "checkpoints" / "magnetic_oracle_feature_conditional_retrain.pt"
        ),
    )
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "wav" / "hybrid.wav"))
    parser.add_argument(
        "--diagnostics",
        default=str(ROOT / "artifacts" / "metrics" / "hybrid_diagnostics.json"),
    )
    parser.add_argument(
        "--metrics",
        default=str(ROOT / "artifacts" / "metrics" / "vme_metrics.json"),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shorter duration + fewer iterations + looser ODE tolerances (smoke test)",
    )
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    report = run_hybrid(
        n_notes=args.n_notes,
        seed=args.seed,
        tempo=args.tempo,
        velocity=args.velocity,
        checkpoint=args.checkpoint,
        out_wav=args.out,
        diagnostics_path=args.diagnostics,
        metrics_path=args.metrics,
        fast=args.fast,
        iterations=args.iterations,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
