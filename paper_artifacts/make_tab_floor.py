"""Table: positional floor on the benchmark folds, for every pipeline, with the
elevation arm. Reads results/floor_benchmark_folds.json (src/floor_benchmark_folds.py).
"""
from __future__ import annotations

import json

from paths import RESULTS, TABLES_DIR

BASINS = [("06_rio_huasco", "Huasco"), ("09_rio_maipo", "Maipo"), ("11_rio_maule", "Maule")]
COLS = [("A", "A"), ("A+elev", "A+elev"), ("TM+DEM", "TerraMind+DEM"),
        ("TM+DEM+S2", "TerraMind+DEM+S2"), ("Prithvi", "Prithvi+S2")]


def cell(v):
    star = "$^{*}$" if v["sig"] else ""
    dag = "$^{\\dagger}$" if v["sig_nb"] else ""
    return f"{v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]{star}{dag}"


def main():
    d = json.loads((RESULTS / "floor_benchmark_folds.json").read_text())
    lines = [r"\begin{table*}", r"\centering",
             r"\caption{Positional floor on the benchmark folds (\texttt{GroupKFold}, $10$~km "
             r"blocks). Upper block: ROC AUC of the coordinates-only classifier and of every "
             r"pipeline; A+elev adds the elevation at the pixel to the seventeen layers. Lower "
             r"block: each pipeline's margin over the coordinates-only floor, fold mean with "
             r"fold-level paired $95\%$ $t$-interval; $^{*}$: excludes zero; $^{\dagger}$: the "
             r"Nadeau--Bengio corrected interval also excludes zero. A basin is interpretable "
             r"for the comparison between pipelines only if the baseline A clears the floor.}",
             r"\label{tab:floor}", r"\small",
             r"\begin{tabular}{l" + "c" * (len(COLS) + 1) + "}", r"\toprule",
             "Basin & Coordinates & " + " & ".join(l for _, l in COLS) + r" \\", r"\midrule"]
    for slug, name in BASINS:
        r = d[slug]["roc_mean"]
        lines.append(f"{name} & {r['coords']:.3f} & " + " & ".join(f"{r[k]:.3f}" for k, _ in COLS) + r" \\")
    lines += [r"\midrule", r"\multicolumn{" + str(len(COLS) + 2) + r"}{l}{\emph{Margin over the coordinates-only floor}} \\"]
    for slug, name in BASINS:
        v = d[slug]["vs_coords"]
        lines.append(f"{name} & -- & " + " & ".join(cell(v[k]) for k, _ in COLS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES_DIR / "tab_floor.tex").write_text("\n".join(lines) + "\n")
    print("[write] tab_floor.tex")


if __name__ == "__main__":
    main()
