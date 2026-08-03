from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

BASINS = (
    "01_rio_lluta",
    "06_rio_huasco",
    "09_rio_maipo",
    "11_rio_maule",
    "13_rio_bueno",
)
DEFAULT_BASIN = "06_rio_huasco"

TERRAIN_FEATURES = [
    "convergence",
    "curvature",
    "dev",
    "eastness",
    "mrrtf",
    "mrvbf",
    "northness",
    "openness_negative",
    "openness_positive",
    "slope",
    "svf",
    "tpi",
    "tri",
    "vrm",
]

HYDROLOGY_FEATURES = ["twi", "hand"]
HYDROLOGY_LOG_FEATURES = ["flow_accumulation_mfd"]


def basin_dir(basin: str) -> Path:
    return DATA / basin


def feature_paths(basin: str) -> dict[str, Path]:
    bd = basin_dir(basin)
    paths = {name: bd / "terrain" / f"{name}.tif" for name in TERRAIN_FEATURES}
    paths.update({name: bd / "hydrology" / f"{name}.tif" for name in HYDROLOGY_FEATURES})
    paths.update(
        {
            f"log1p_{name}": bd / "hydrology" / f"{name}.tif"
            for name in HYDROLOGY_LOG_FEATURES
        }
    )
    return paths


LOG_TRANSFORM = {f"log1p_{n}" for n in HYDROLOGY_LOG_FEATURES}


def feature_names(basin: str) -> list[str]:
    return list(feature_paths(basin).keys())
