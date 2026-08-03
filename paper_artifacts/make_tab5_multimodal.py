"""Generate Table 5: DEM-only vs multimodal (DEM+S2L2A) comparison.

Intervals: primary fold-level paired 95% t-interval (df=4, asterisk) and
secondary pooled paired-bootstrap 95% CI (dagger), consistent with Table 3.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path("/home/franciscoparrao/proyectos/no_supervisado_superficie")
RESULTS = ROOT / "pregunta_3_unidades_geomorfologicas/results"
OUT_DIR = ROOT / "paper/tables"

BASINS = ["06_rio_huasco", "09_rio_maipo", "11_rio_maule"]
SHORT = {"06_rio_huasco": "Huasco", "09_rio_maipo": "Maipo", "11_rio_maule": "Maule"}
REGIME = {"06_rio_huasco": "Semi-arid", "09_rio_maipo": "Mediterranean",
          "11_rio_maule": "Temperate-humid"}


def fmt_mean_std(m, s):
    return f"{m:.3f} $\\pm$ {s:.3f}"


def fmt_delta(mean, _std=None):
    return f"{mean:+.3f}"


T_975_DF4 = 2.776  # two-sided 95% t quantile, df = 4


def fold_tci(values):
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    h = T_975_DF4 * sd / math.sqrt(n)
    return m, m - h, m + h


def fmt_fold(d, key="delta_roc"):
    m, lo, hi = fold_tci([f[key] for f in d["fold_results"]])
    sig = "$^{*}$" if not (lo <= 0 <= hi) else ""
    return f"[{lo:+.3f}, {hi:+.3f}]{sig}"


def fmt_ci(ci_low, ci_high):
    sig = "$^{\\dagger}$" if not (ci_low <= 0 <= ci_high) else ""
    return f"[{ci_low:+.3f}, {ci_high:+.3f}]{sig}"


def main():
    rows = []
    for b in BASINS:
        dem = json.loads((RESULTS / f"{b}_terramind_linprobe_spatial.json").read_text())
        mm  = json.loads((RESULTS / f"{b}_terramind_linprobe_spatial_dem+s2l2a.json").read_text())
        rows.append({
            "basin": SHORT[b], "regime": REGIME[b],
            "a_roc":     fmt_mean_std(dem["A_roc_mean"], dem["A_roc_std"]),
            "bdem_roc":  fmt_mean_std(dem["B_roc_mean"], dem["B_roc_std"]),
            "bmm_roc":   fmt_mean_std(mm["B_roc_mean"], mm["B_roc_std"]),
            "d_dem_roc": fmt_delta(dem["delta_roc_mean"], dem.get("delta_roc_std", 0)),
            "fci_dem_roc": fmt_fold(dem),
            "ci_dem_roc": fmt_ci(*dem["delta_roc_ci_95"]),
            "d_mm_roc":  fmt_delta(mm["delta_roc_mean"], mm.get("delta_roc_std", 0)),
            "fci_mm_roc": fmt_fold(mm),
            "ci_mm_roc": fmt_ci(*mm["delta_roc_ci_95"]),
            "d_dem_pr":  fmt_delta(dem["delta_pr_mean"], dem.get("delta_pr_std", 0)),
            "ci_dem_pr": fmt_ci(*dem["delta_pr_ci_95"]),
            "d_mm_pr":   fmt_delta(mm["delta_pr_mean"], mm.get("delta_pr_std", 0)),
            "ci_mm_pr":  fmt_ci(*mm["delta_pr_ci_95"]),
        })

    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Multimodal extension. AUC ROC of pipeline B with DEM only versus pipeline B with both DEM and Sentinel-2 L2A. The manual baseline (pipeline A) and $\Delta$ROC against pipeline A are reported with the primary fold-level paired $95\%$ $t$-interval (df $= 4$); the corresponding pooled-bootstrap intervals for the same cells appear in Table~\ref{tab:benchmark}. Adding the Sentinel-2 modality shifts the point estimates by small margins; the degradation of pipeline B in temperate-humid Maule persists in ROC under both inference levels. $^{*}$: fold-level interval excludes zero.}",
        r"\label{tab:multimodal}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccc@{\hspace{0.4em}}cc@{\hspace{0.4em}}cc}",
        r"\toprule",
        r" & & A & B (DEM only) & B (DEM+S2) & \multicolumn{2}{c}{$\Delta$ROC DEM only} & \multicolumn{2}{c}{$\Delta$ROC multimodal} \\",
        r"\cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"Basin & Regime & AUC ROC & AUC ROC & AUC ROC & mean & fold-level 95\% CI & mean & fold-level 95\% CI \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['basin']} & {r['regime']} & "
            f"{r['a_roc']} & {r['bdem_roc']} & {r['bmm_roc']} & "
            f"{r['d_dem_roc']} & {r['fci_dem_roc']} & "
            f"{r['d_mm_roc']} & {r['fci_mm_roc']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (OUT_DIR / "tab5_multimodal.tex").write_text("\n".join(lines))
    print(f"  wrote tab5_multimodal.tex")

    md_lines = [
        "# Table 5 — Multimodal extension (DEM-only vs DEM+S2L2A)", "",
        "| Basin | Regime | A ROC | B DEM ROC | B DEM+S2 ROC | ΔROC DEM-only | fold CI | ΔROC multimodal | fold CI |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        def md(s):
            return (s.replace("$^{*}$", "*").replace("$^{\\dagger}$", "†").replace("$\\pm$", "±"))
        md_lines.append(
            f"| {r['basin']} | {r['regime']} | {md(r['a_roc'])} | {md(r['bdem_roc'])} | {md(r['bmm_roc'])} | "
            f"{md(r['d_dem_roc'])} | {md(r['fci_dem_roc'])} | "
            f"{md(r['d_mm_roc'])} | {md(r['fci_mm_roc'])} |"
        )
    md_lines.extend(["", "*: fold-level 95% t-CI (df=4) excludes zero. Pooled-bootstrap intervals for the same cells: see Table 3."])
    (OUT_DIR / "tab5_multimodal.md").write_text("\n".join(md_lines))
    print(f"  wrote tab5_multimodal.md")


if __name__ == "__main__":
    main()
