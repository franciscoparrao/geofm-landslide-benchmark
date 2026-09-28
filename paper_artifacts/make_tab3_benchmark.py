"""Table 3 — Cross-FM benchmark across 3 climate regimes.

Compares the 17-feature baseline (A) against three foundation-model pipelines:
TerraMind+DEM, TerraMind+DEM+S2L2A, Prithvi-EO-2.0+S2(HLS). One row per
(basin, FM) combination. ΔROC reports the fold-mean with two intervals:
the primary fold-level paired 95% t-interval (df=4, spatially conservative)
and the secondary pooled paired-bootstrap 95% CI (larger n, anti-conservative
under spatial autocorrelation).
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path

from scipy.stats import t as student_t

# Suffix of the result files to read, e.g. "_sgkf_k10" for the 10-fold ladder
# rung. Empty means the primary 5-fold GroupKFold analysis. The output tables
# carry the same suffix so a variant never overwrites the primary ones.
VARIANT = os.environ.get("TABLE_VARIANT", "")

from paths import RESULTS
from paths import TABLES_DIR as OUT_DIR

BASINS = [("06_rio_huasco", "Huasco", "Semi-arid"),
          ("09_rio_maipo",  "Maipo",  "Mediterranean"),
          ("11_rio_maule",  "Maule",  "Temperate-humid")]

FMS = [
    ("TerraMind+DEM",      "terramind",    "dem"),
    ("TerraMind+DEM+S2L2A","terramind",    "mm"),
    ("Prithvi-EO-2.0+S2",  "prithvi-300m", "s2"),
]

_FOLD_COUNTS = set()


def t_crit(n):
    """Two-sided 95% t quantile for n folds, derived from the data.

    Hardcoding df = 4 silently understates the interval whenever the file
    holds a different fold count, so the quantile follows the folds present.
    """
    _FOLD_COUNTS.add(int(n))
    return float(student_t.ppf(0.975, n - 1))


def load(basin, model, mod):
    if model == "terramind":
        suffix = "" if mod == "dem" else "_dem+s2l2a"
        return json.loads((RESULTS / f"{basin}_terramind_linprobe_spatial{suffix}{VARIANT}.json").read_text())
    return json.loads((RESULTS / f"{basin}_prithvi-300m_linprobe_spatial{VARIANT}.json").read_text())


def fold_tci(values):
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    h = t_crit(n) * sd / math.sqrt(n)
    return m, m - h, m + h


def fmt_fold(lo, hi):
    sig = "$^{*}$" if not (lo <= 0 <= hi) else ""
    return f"[{lo:+.3f}, {hi:+.3f}]{sig}"


def fmt_boot(ci):
    sig = "$^{\\dagger}$" if not (ci[0] <= 0 <= ci[1]) else ""
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]{sig}"


def fold_wording(n):
    """Caption wording that follows the fold count actually in the data.

    The captions used to hardcode "5-fold" and "df = 4"; pointed at a variant
    they would have described df = 4 above intervals computed with df = 9.
    """
    return {"5-fold": f"{n}-fold",
            "df $= 4$": f"df $= {n - 1}$",
            "(df=4)": f"(df={n - 1})",
            "df=4": f"df={n - 1}",
            "five spatial folds": f"{n} spatial folds"}


def apply_wording(text, n=None):
    n = n or (max(_FOLD_COUNTS) if _FOLD_COUNTS else 5)
    for old, new in fold_wording(n).items():
        text = text.replace(old, new)
    return text


def main():
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Cross-model benchmark in three Chilean basins spanning $6.8^{\circ}$ of latitude ($29.1^{\circ}$--$35.8^{\circ}$~S), from semi-arid to temperate-humid. The $22^{\circ}$ transect of Table~\ref{tab:basins} is the five-basin descriptive design; only these three basins carry inventories large enough to benchmark. The 17-feature geomorphometric baseline (\textbf{A}) is contrasted against three foundation-model pipelines: TerraMind v1-tiny with DEM only; TerraMind v1-tiny multimodal (DEM and Sentinel-2 L2A); and Prithvi-EO-2.0-300M with HLS-equivalent Sentinel-2 bands. AUC values are 5-fold spatial-holdout means $\pm$ across-fold standard deviation. $\Delta$ROC is the fold-mean difference, reported with two intervals: the primary fold-level paired $95\%$ $t$-interval (df $= 4$), which treats spatial folds as the exchangeable unit, and the secondary pooled paired-bootstrap $95\%$ CI ($N=1{,}000$) over concatenated test predictions, which has larger nominal sample size but ignores spatial autocorrelation and cross-fold score pooling and is therefore anti-conservative. $^{*}$: fold-level interval excludes zero; $^{\dagger}$: pooled bootstrap interval excludes zero.}",
        r"\label{tab:benchmark}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lllccccc}",
        r"\toprule",
        r"Basin & Regime & Pipeline & A ROC & B ROC & $\Delta$ROC & fold-level 95\% CI & pooled boot.\ 95\% CI \\",
        r"\midrule",
    ]
    for b_slug, b_name, regime in BASINS:
        for fm_name, model, mod in FMS:
            d = load(b_slug, model, mod)
            deltas = [f["delta_roc"] for f in d["fold_results"]]
            m, lo, hi = fold_tci(deltas)
            row = (
                f"{b_name} & {regime} & {fm_name} & "
                f"{d['A_roc_mean']:.3f} $\\pm$ {d['A_roc_std']:.3f} & "
                f"{d['B_roc_mean']:.3f} $\\pm$ {d['B_roc_std']:.3f} & "
                f"{m:+.3f} & "
                f"{fmt_fold(lo, hi)} & "
                f"{fmt_boot(d['delta_roc_ci_95'])} \\\\"
            )
            lines.append(row)
        lines.append(r"\midrule" if b_slug != BASINS[-1][0] else r"\bottomrule")
    lines.extend([r"\end{tabular}", r"\end{table*}", ""])
    (OUT_DIR / f"tab3_benchmark{VARIANT}.tex").write_text(apply_wording("\n".join(lines)))
    print(f"  wrote tab3_benchmark{VARIANT}.tex")

    # Markdown version
    md = ["# Table 3 — Cross-FM benchmark", "",
          "| Basin | Regime | Pipeline | A ROC | B ROC | ΔROC | fold-level 95% CI | pooled boot. 95% CI |",
          "|---|---|---|---|---|---|---|---|"]
    for b_slug, b_name, regime in BASINS:
        for fm_name, model, mod in FMS:
            d = load(b_slug, model, mod)
            deltas = [f["delta_roc"] for f in d["fold_results"]]
            m, lo, hi = fold_tci(deltas)
            fsig = "*" if not (lo <= 0 <= hi) else ""
            ci = d["delta_roc_ci_95"]
            bsig = "†" if not (ci[0] <= 0 <= ci[1]) else ""
            md.append(
                f"| {b_name} | {regime} | {fm_name} | "
                f"{d['A_roc_mean']:.3f} ± {d['A_roc_std']:.3f} | "
                f"{d['B_roc_mean']:.3f} ± {d['B_roc_std']:.3f} | "
                f"{m:+.3f} | "
                f"[{lo:+.3f}, {hi:+.3f}]{fsig} | "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}]{bsig} |"
            )
    md.append("\n*: fold-level 95% t-CI (df=4) excludes zero. †: pooled paired-bootstrap 95% CI excludes zero (anti-conservative under spatial autocorrelation).")
    (OUT_DIR / f"tab3_benchmark{VARIANT}.md").write_text(apply_wording("\n".join(md)))
    print(f"  wrote tab3_benchmark{VARIANT}.md")


if __name__ == "__main__":
    main()
