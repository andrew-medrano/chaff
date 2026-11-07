"""chaff: decoy propeller flyovers that mislead acoustic Doppler-curve geolocation."""
from __future__ import annotations

from .pipeline import (LEAD, QUIET, SYNTHETIC, ChaffWarning, Synthetic, Tier, build_catalog, protect,
                       protect_file)
from .render import default_jobs
from .stack import bundled_path as bundled_catalog, load as load_catalog, load_info as catalog_info
from .stack import save as save_catalog

__version__ = "0.3.0"
__all__ = ["protect", "protect_file", "build_catalog", "load_catalog", "save_catalog", "bundled_catalog",
           "catalog_info", "default_jobs", "Tier", "Synthetic", "LEAD", "QUIET", "SYNTHETIC", "ChaffWarning",
           "__version__"]
