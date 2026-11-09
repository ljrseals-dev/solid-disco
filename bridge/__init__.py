"""Bridge between VME (DownwashResolver / Config A) and MagneticOracle physics renderer."""

from .events import events_from_pitches, pitches_from_vme
from .vme_to_oracle import run_hybrid

__all__ = ["events_from_pitches", "pitches_from_vme", "run_hybrid"]
