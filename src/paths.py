"""Output and result locations for the table and figure generators.

These scripts live in two trees with different layouts: the research monorepo,
where they sit in <repo>/paper/scripts and read results from
pregunta_3_unidades_geomorfologicas/results, and the distributed
reproducibility package, where they sit in <pkg>/paper_artifacts alongside
<pkg>/results. Resolving the layout from the script's own location keeps a
single copy of every generator valid in both, instead of hardcoding one
machine's absolute paths.

Every location can be overridden with an environment variable, which is what
makes the package usable from an arbitrary checkout.
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

if _HERE.name == "scripts":              # research monorepo: <repo>/paper/scripts
    ROOT = _HERE.parent.parent
    _PIPELINE = ROOT / "pregunta_3_unidades_geomorfologicas"
    _RESULTS = _PIPELINE / "results"
    _BASIN_DATA = _PIPELINE / "data"
    _TABLES = ROOT / "paper" / "tables"
    _FIGURES = ROOT / "paper" / "figures"
    _S2 = ROOT / "paper" / "data" / "s2_composites"
else:                                    # package: <pkg>/paper_artifacts
    ROOT = _HERE.parent
    _RESULTS = ROOT / "results"
    _BASIN_DATA = ROOT / "data"
    _TABLES = ROOT / "tables"
    _FIGURES = ROOT / "figures"
    _S2 = ROOT / "data" / "s2_composites"


def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, str(default)))


RESULTS = _env_path("GEOFM_RESULTS_DIR", _RESULTS)
TABLES_DIR = _env_path("GEOFM_TABLES_DIR", _TABLES)
FIGURES_DIR = _env_path("GEOFM_FIGURES_DIR", _FIGURES)
S2_COMPOSITE_DIR = _env_path("GEOFM_S2_DIR", _S2)

# Per-basin inputs, laid out as data/{basin}/dem_30m.tif (see data/README.md).
BASIN_DATA_DIR = _env_path("GEOFM_BASIN_DATA_DIR", _BASIN_DATA)
BASIN_POLY_DIR = _env_path("GEOFM_BASIN_POLYGONS_DIR",
                           BASIN_DATA_DIR / "basin_polygons")
ML_DATASET_DIR = _env_path("GEOFM_ML_DATASET_DIR", BASIN_DATA_DIR / "ml_dataset")
# National basin boundaries (DGA/BNA), used only for the locator map.
COUNTRY_SHP = _env_path("GEOFM_COUNTRY_SHP",
                        BASIN_DATA_DIR / "Cuencas_BNA" / "Cuencas_BNA.shp")
