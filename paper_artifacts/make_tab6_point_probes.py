"""Table 6 — Point-probe controls: spectral-point and context-matched baselines.

Reads results/{basin}_point_probes.json (produced by src/point_probes.py) and
reports AUC ROC of A, SPEC, A+SPEC and ACTX with fold-level paired t-intervals
of the difference against A, plus the paired difference of the stored FM runs
against ACTX (environment caveat noted in the manuscript).
"""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/no_supervisado_superficie")
RESULTS = ROOT / "pregunta_3_unidades_geomorfologicas/results"
OUT_DIR = ROOT / "paper/tables"

BASINS = [("06_rio_huasco", "Huasco", "Semi-arid"),
          ("09_rio_maipo",  "Maipo",  "Mediterranean"),
          ("11_rio_maule",  "Maule",  "Temperate-humid")]

SETS = [("SPEC", "Spec.-point (8)"),
        ("A_SPEC", "A + spec.-point (25)"),
        ("ACTX", "A + terrain ctx (119)"),
        ("SCTX", "Spec. + ctx (56)"),
        ("AFULL", "A + full ctx (175)")]

T_975_DF4 = 2.776


def tci(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    h = T_975_DF4 * sd / math.sqrt(n)
    return m, m - h, m + h


def fmt_delta(deltas, star="$^{*}$"):
    m, lo, hi = tci(deltas)
    sig = star if not (lo <= 0 <= hi) else ""
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"


def main():
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Point-probe controls on the identical dataset and spatial folds of the benchmark. \textbf{A}: 17 geomorphometric features at the candidate pixel. \textbf{Spectral-point}: six HLS-equivalent bands plus NDVI and NBR sampled at the candidate pixel from the same $2023$ Sentinel-2 composite used by the FM pipelines (post-event scar-signal probe). Context variants augment the point features with nan-aware mean and standard deviation of each layer over $7\times7$, $37\times37$ and $111\times111$ pixel windows ($0.2$--$3.3$~km): \textbf{A + terrain context} extends the geomorphometric layers, \textbf{spectral point + context} the composite layers, and \textbf{A + full context} both --- the fully context-matched manual comparator to the FM pipelines. $\Delta$ROC is against A on identical folds with fold-level paired $95\%$ $t$-intervals (df $= 4$); $^{*}$: interval excludes zero. The last column reports the paired fold-level difference of the strongest stored FM pipeline in each basin against A + full context.}",
        r"\label{tab:pointprobes}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Basin & Probe & AUC ROC & $\Delta$ROC vs A [95\% CI] & $\Delta$PR vs A [95\% CI] & best FM vs A+full context \\",
        r"\midrule",
    ]
    md_rows = []
    for b_slug, b_name, regime in BASINS:
        d = json.loads((RESULTS / f"{b_slug}_point_probes.json").read_text())
        fm = d["fold_metrics"]
        A_roc, A_pr = fm["A"]["roc"], fm["A"]["pr"]
        a_mean = sum(A_roc) / len(A_roc)

        # strongest stored FM by mean roc
        best_key, best_mean = None, -1.0
        for key in ("TM_MM", "PRITHVI"):
            if key in fm:
                mmean = sum(fm[key]["roc"]) / len(fm[key]["roc"])
                if mmean > best_mean:
                    best_key, best_mean = key, mmean
        fmlab = {"TM_MM": "TM+DEM+S2", "PRITHVI": "Prithvi+S2"}.get(best_key, "?")
        dvs = [x - a for x, a in zip(fm[best_key]["roc"], fm["AFULL"]["roc"])]
        best_cell = f"{fmlab}: {fmt_delta(dvs)}"

        first = True
        for key, label in SETS:
            roc = fm[key]["roc"]
            droc = [x - a for x, a in zip(roc, A_roc)]
            dpr = [x - a for x, a in zip(fm[key]["pr"], A_pr)]
            mroc = sum(roc) / len(roc)
            basin_cell = f"{b_name} (A {a_mean:.3f})" if first else ""
            last_cell = best_cell if first else ""
            lines.append(
                f"{basin_cell} & {label} & {mroc:.3f} & "
                f"{fmt_delta(droc)} & {fmt_delta(dpr)} & {last_cell} \\\\"
            )
            md_rows.append(
                f"| {b_name} | {label} | {mroc:.3f} | "
                f"{fmt_delta(droc, star='*')} | {fmt_delta(dpr, star='*')} | "
                f"{best_cell if first else ''} |"
            )
            first = False
        lines.append(r"\midrule" if b_slug != BASINS[-1][0] else r"\bottomrule")
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    (OUT_DIR / "tab6_point_probes.tex").write_text("\n".join(lines))
    print("  wrote tab6_point_probes.tex")

    md = ["# Table 6 — Point-probe controls", "",
          "| Basin | Probe | AUC ROC | ΔROC vs A | ΔPR vs A | best FM vs A+full context |",
          "|---|---|---|---|---|---|"] + md_rows + [
          "", "*: fold-level paired 95% t-CI (df=4) excludes zero."]
    (OUT_DIR / "tab6_point_probes.md").write_text("\n".join(md))
    print("  wrote tab6_point_probes.md")


if __name__ == "__main__":
    main()
