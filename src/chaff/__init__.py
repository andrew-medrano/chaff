"""chaff: decoy propeller flyovers that mislead acoustic Doppler-curve geolocation."""
from __future__ import annotations

from .pipeline import (LEAD, QUIET, SYNTHETIC, ChaffWarning, Synthetic, Tier, build_catalog, protect,
                       protect_file)
from .stack import load as load_catalog, save as save_catalog

__version__ = "0.2.0"
__all__ = ["protect", "protect_file", "build_catalog", "load_catalog", "save_catalog", "Tier", "Synthetic",
           "LEAD", "QUIET", "SYNTHETIC", "ChaffWarning", "__version__"]
