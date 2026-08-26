"""Table 7 — Constrained-negative sensitivity (hard-task variant).

Reads results/{basin}_point_probes_constrained.json and reports AUC ROC of
A, spectral-point, and the context variants under negatives restricted to the
positives' slope envelope, with fold-level paired t-intervals of the
difference against A (StratifiedGroupKFold folds; internally paired).
"""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/no_supervisado_superficie")
RESULTS = ROOT / "pregunta_3_unidades_geomorfologicas/results"
OUT_DIR = ROOT / "paper/tables"

BASINS = [("06_rio_huasco", "Huasco"),
          ("09_rio_maipo",  "Maipo"),
          ("11_rio_maule",  "Maule")]

SETS = [("SPEC", "Spec.-point (8)"),
        ("ACTX", "A + terrain ctx (119)"),
        ("SCTX", "Spec. + ctx (56)"),
        ("AFULL", "A + full ctx (175)"),
        ("TM_DEM_ENV", "TerraMind+DEM"),
        ("TM_MM_ENV", "TerraMind+DEM+S2")]

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
        r"\caption{Constrained-negative sensitivity: manual pipelines re-run with negatives restricted to the positives' slope envelope ($\geq$ 10th percentile of slope at positive locations; Section~\ref{sec:methods:pointprobes}), including the TerraMind pipelines re-encoded in the same environment. AUC ROC with fold-level paired $95\%$ $t$-intervals of the difference against A on the same (stratified) folds; $^{*}$: interval excludes zero. AUC PR of A is reported per basin to show the hardening of the task relative to uniform sampling.}",
        r"\label{tab:constrained}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Basin & Probe & AUC ROC & $\Delta$ROC vs A [95\% CI] \\",
        r"\midrule",
    ]
    md_rows = []
    for b_slug, b_name in BASINS:
        d = json.loads((RESULTS / f"{b_slug}_point_probes_constrained.json").read_text())
        fm = d["fold_metrics"]
        A_roc = fm["A"]["roc"]
        a_mean = sum(A_roc) / len(A_roc)
        a_pr = sum(fm["A"]["pr"]) / len(fm["A"]["pr"])
        first = True
        for key, label in SETS:
            if key not in fm:
                continue
            roc = fm[key]["roc"]
            droc = [x - a for x, a in zip(roc, A_roc)]
            mroc = sum(roc) / len(roc)
            basin_cell = f"{b_name} (A: {a_mean:.3f} ROC, {a_pr:.3f} PR)" if first else ""
            lines.append(f"{basin_cell} & {label} & {mroc:.3f} & {fmt_delta(droc)} \\\\")
            md_rows.append(f"| {b_name} | {label} | {mroc:.3f} | {fmt_delta(droc, star='*')} |")
            first = False
        lines.append(r"\midrule" if b_slug != BASINS[-1][0] else r"\bottomrule")
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    (OUT_DIR / "tab7_constrained.tex").write_text("\n".join(lines))
    print("  wrote tab7_constrained.tex")

    md = ["# Table 7 — Constrained-negative sensitivity", "",
          "| Basin | Probe | AUC ROC | ΔROC vs A |",
          "|---|---|---|---|"] + md_rows + [
          "", "*: fold-level paired 95% t-CI (df=4) excludes zero."]
    (OUT_DIR / "tab7_constrained.md").write_text("\n".join(md))
    print("  wrote tab7_constrained.md")


if __name__ == "__main__":
    main()
