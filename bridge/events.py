"""Convert VME pitch sequences into MagneticOracle timed events."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Event tuple expected by the patched MagneticOracle renderer:
# (onset_time_sec, midi_note, velocity_0_1)
Event = Tuple[float, int, float]


def events_from_pitches(
    pitches: Sequence[int],
    durations: Optional[Sequence[float]] = None,
    tempo: float = 120.0,
    default_duration_beats: float = 0.5,
    velocity: float = 0.75,
    velocity_jitter: float = 0.08,
    seed: Optional[int] = None,
) -> List[Event]:
    """
    Map a list of MIDI pitches to timed MagneticOracle events.

    Parameters
    ----------
    pitches : sequence of int
        MIDI note numbers (0-127).
    durations : optional sequence of float
        Duration of each note in beats. If None, every note is
        `default_duration_beats` long.
    tempo : float
        Beats per minute. Used only when durations are given in beats.
    default_duration_beats : float
        Fallback duration when `durations` is None.
    velocity : float
        Base velocity in [0, 1].
    velocity_jitter : float
        Uniform jitter added to velocity (clipped to [0.05, 1.0]).
    seed : optional int
        RNG seed for velocity jitter reproducibility.

    Returns
    -------
    list of (onset_sec, midi_note, velocity)
    """
    if not pitches:
        return []

    rng = random.Random(seed)
    n = len(pitches)

    if durations is None:
        durs = [default_duration_beats] * n
    else:
        if len(durations) != n:
            raise ValueError("durations length must match pitches length")
        durs = list(durations)

    beat_to_sec = 60.0 / max(tempo, 1e-6)
    events: List[Event] = []
    t = 0.0
    for pitch, dur in zip(pitches, durs):
        vel = float(np.clip(velocity + rng.uniform(-velocity_jitter, velocity_jitter), 0.05, 1.0))
        events.append((t, int(pitch), vel))
        t += dur * beat_to_sec

    return events


def pitches_from_vme(
    n_notes: int = 64,
    seed: int = 42,
    low: int = 58,
    high: int = 76,
) -> Tuple[List[int], dict]:
    """
    Generate a VME Config-A pitch sequence and its four core metrics.

    Returns
    -------
    pitches : list[int]
    metrics : dict with stepwise_pct, repeat_pct, mean_interval, key_correlation
    """
    import sys
    from pathlib import Path

    # Make vendor VME importable
    root = Path(__file__).resolve().parents[1]
    vme_src = root / "vendor" / "vme_src"
    if str(vme_src) not in sys.path:
        sys.path.insert(0, str(vme_src))

    from downwash_resolver import DownwashResolver  # type: ignore

    random.seed(seed)
    np.random.seed(seed)

    resolver = DownwashResolver()  # loads Configuration A defaults

    def generate_raw_pitch(last_raw):
        if last_raw is None:
            return random.randint(62, 70)
        step = random.choices(
            [-3, -2, -1, 0, 1, 2, 3],
            weights=[1, 3, 6, 2, 6, 3, 1],
            k=1,
        )[0]
        return max(low, min(high, last_raw + step))

    pitches: List[int] = []
    last_raw = None
    for _ in range(n_notes):
        raw = generate_raw_pitch(last_raw)
        last_raw = raw
        resolver.maybe_change_chord()
        fitted = resolver.fit_pitch_to_chord(raw)
        pitches.append(int(fitted))

    # Metrics (identical to VME example)
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    stepwise = 100.0 * sum(1 for i in intervals if i <= 2) / max(len(intervals), 1)
    repeats = 100.0 * sum(1 for i in intervals if i == 0) / max(len(intervals), 1)
    mean_int = float(np.mean(intervals)) if intervals else 0.0

    pitch_classes = [p % 12 for p in pitches]
    hist = np.bincount(pitch_classes, minlength=12).astype(float) / max(len(pitches), 1)
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    corr = float(np.corrcoef(hist, major_profile)[0, 1]) if len(pitches) > 1 else 0.0

    metrics = {
        "stepwise_pct": round(stepwise, 2),
        "repeat_pct": round(repeats, 2),
        "mean_interval": round(mean_int, 3),
        "key_correlation": round(corr, 4),
        "n_notes": n_notes,
        "seed": seed,
        "weights": resolver.get_weights(),
    }
    return pitches, metrics
