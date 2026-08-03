"""Table 4 — Cross-FM random-initialization ablation.

Reports ΔROC vs the manual baseline for pretrained and random-initialized
variants of each FM, plus the pretraining-contribution gap. All intervals are
fold-level paired 95% t-intervals (df=4), the spatially conservative primary
inference of the paper; the previous 5-value fold bootstrap was replaced
because resampling five observations yields unreliable percentile intervals.
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

FMS = [
    ("TerraMind+DEM",      "terramind",    "dem"),
    ("TerraMind+DEM+S2L2A","terramind",    "mm"),
    ("Prithvi-EO-2.0+S2",  "prithvi-300m", "s2"),
]

T_975_DF4 = 2.776  # two-sided 95% t quantile, df = 4


def load(basin, model, mod, init):
    init_s = "_randinit" if init == "rand" else ""
    if model == "terramind":
        mod_s = "" if mod == "dem" else "_dem+s2l2a"
        path = RESULTS / f"{basin}_terramind_linprobe_spatial{init_s}{mod_s}.json"
    else:
        path = RESULTS / f"{basin}_prithvi-300m_linprobe_spatial{init_s}.json"
    return json.loads(path.read_text())


def fold_tci(values):
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    h = T_975_DF4 * sd / math.sqrt(n)
    return m, m - h, m + h


def deltas(d):
    return [r["delta_roc"] for r in d["fold_results"]]


def fmt_mtci(values, star="$^{*}$"):
    m, lo, hi = fold_tci(values)
    sig = star if not (lo <= 0 <= hi) else ""
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}", (m, lo, hi)


def main():
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Random-initialization ablation across the cross-FM benchmark. For each combination of basin and foundation-model pipeline, $\Delta$ROC is reported for the publicly released pretrained weights (\textit{pretrained}) and for an identically structured backbone re-initialized with the Xavier-uniform scheme (\textit{random-init}). The \emph{pretraining gap} $\Delta_{\text{PT}} = \Delta\mathrm{ROC}_{\text{pretr.}} - \Delta\mathrm{ROC}_{\text{rand.}}$ (last column) isolates the contribution of pretraining. All intervals are fold-level paired $95\%$ $t$-intervals (df $= 4$) over the five spatial folds; the pretraining-gap interval is computed on the per-fold paired differences. $^{*}$: interval excludes zero.}",
        r"\label{tab:ablation}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llllc}",
        r"\toprule",
        r"Basin & Regime & Pipeline & $\Delta$ROC pretrained / random-init & PT gap [95\% CI] \\",
        r"\midrule",
    ]
    for b_slug, b_name, regime in BASINS:
        for fm_name, model, mod in FMS:
            d_p = load(b_slug, model, mod, "pretr")
            d_r = load(b_slug, model, mod, "rand")
            fp, fr = deltas(d_p), deltas(d_r)
            s_p, _ = fmt_mtci(fp)
            s_r, _ = fmt_mtci(fr)
            gap = [a - b for a, b in zip(fp, fr)]
            s_g, _ = fmt_mtci(gap)
            lines.append(
                f"{b_name} & {regime} & {fm_name} & {s_p} / {s_r} & {s_g} \\\\"
            )
        lines.append(r"\midrule" if b_slug != BASINS[-1][0] else r"\bottomrule")
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    (OUT_DIR / "tab4_ablation.tex").write_text("\n".join(lines))
    print("  wrote tab4_ablation.tex")

    # Markdown version (cleaner: separate columns)
    md = ["# Table 4 — Cross-FM random-init ablation", "",
          "| Basin | Regime | Pipeline | ΔROC pretrained | ΔROC random-init | PT gap [95% CI] |",
          "|---|---|---|---|---|---|"]
    for b_slug, b_name, regime in BASINS:
        for fm_name, model, mod in FMS:
            d_p = load(b_slug, model, mod, "pretr")
            d_r = load(b_slug, model, mod, "rand")
            fp, fr = deltas(d_p), deltas(d_r)
            s_p, _ = fmt_mtci(fp, star="*")
            s_r, _ = fmt_mtci(fr, star="*")
            gap = [a - b for a, b in zip(fp, fr)]
            s_g, _ = fmt_mtci(gap, star="*")
            md.append(f"| {b_name} | {regime} | {fm_name} | {s_p} | {s_r} | {s_g} |")
    md.append("\n*: fold-level paired 95% t-CI (df=4) excludes zero.")
    md.append("PT gap = ΔROC(pretrained) − ΔROC(random-init); CI on per-fold paired differences.")
    (OUT_DIR / "tab4_ablation.md").write_text("\n".join(md))
    print("  wrote tab4_ablation.md")


if __name__ == "__main__":
    main()
