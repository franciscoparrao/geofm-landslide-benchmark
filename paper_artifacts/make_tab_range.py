"""Table: range of spatial dependence against the 10 km block. Reads
results/dependence_range.json (src/dependence_range.py).
"""
from __future__ import annotations

import json

from paths import RESULTS, TABLES_DIR

BASINS = [("06_rio_huasco", "Huasco"), ("09_rio_maipo", "Maipo"), ("11_rio_maule", "Maule")]


def rng(f):
    """Span of the spherical and exponential ranges, censored at the maximum lag."""
    lo, hi = sorted((f["spherical"], f["exponential"]), key=lambda v: v["range_km"])
    fmt = lambda v: (r"$\geq$" if v["censored"] else "") + f"{v['range_km']:.1f}"
    return f"{fmt(lo)}--{fmt(hi)}"


def main():
    d = json.loads((RESULTS / "dependence_range.json").read_text())
    lines = [r"\begin{table}", r"\centering",
             r"\caption{Range of spatial dependence (km) against the $10$~km block. Residuals: "
             r"out-of-fold residuals $y-\hat p$ of baseline A on the benchmark folds; labels: "
             r"indicator variogram of presence/background on the benchmark points; both as the span of "
             r"the spherical and exponential practical ranges. Below the $3.36$~km exclusion radius the "
             r"benchmark has no presence/background pairs, so both are zero or biased low there. "
             r"Covariates: median and interquartile range of the local spherical range of a "
             r"nugget + spherical + linear-drift model over the layers with a resolvable local "
             r"structure (count in parentheses).}",
             r"\label{tab:range}", r"\small",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Basin & Residuals (A) & Labels & Covariates, local \\", r"\midrule"]
    for slug, name in BASINS:
        r = d[slug]
        c = r["covariates_local_range"]
        cov = (f"{c['median_km']:.1f} [{c['iqr_km'][0]:.1f}, {c['iqr_km'][1]:.1f}] "
               f"({c['n_resolved']}/{c['n_layers']})")
        lines.append(f"{name} & {rng(r['residuals_A']['fit'])} & {rng(r['labels']['fit'])} & {cov} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES_DIR / "tab_range.tex").write_text("\n".join(lines) + "\n")
    print("[write] tab_range.tex")


if __name__ == "__main__":
    main()
