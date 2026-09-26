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
import os
from pathlib import Path

from scipy.stats import t as student_t

# See make_tab3_benchmark.py: suffix of the result files to read, empty for the
# primary 5-fold GroupKFold analysis.
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
    """Two-sided 95% t quantile for n folds, derived from the data."""
    _FOLD_COUNTS.add(int(n))
    return float(student_t.ppf(0.975, n - 1))


def load(basin, model, mod, init):
    init_s = "_randinit" if init == "rand" else ""
    if model == "terramind":
        mod_s = "" if mod == "dem" else "_dem+s2l2a"
        path = RESULTS / f"{basin}_terramind_linprobe_spatial{init_s}{mod_s}{VARIANT}.json"
    else:
        path = RESULTS / f"{basin}_prithvi-300m_linprobe_spatial{init_s}{VARIANT}.json"
    return json.loads(path.read_text())


def fold_tci(values):
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    h = t_crit(n) * sd / math.sqrt(n)
    return m, m - h, m + h


def deltas(d):
    return [r["delta_roc"] for r in d["fold_results"]]


def fmt_mtci(values, star="$^{*}$"):
    m, lo, hi = fold_tci(values)
    sig = star if not (lo <= 0 <= hi) else ""
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}", (m, lo, hi)


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
    (OUT_DIR / f"tab4_ablation{VARIANT}.tex").write_text(apply_wording("\n".join(lines)))
    print(f"  wrote tab4_ablation{VARIANT}.tex")

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
    (OUT_DIR / f"tab4_ablation{VARIANT}.md").write_text(apply_wording("\n".join(md)))
    print(f"  wrote tab4_ablation{VARIANT}.md")


if __name__ == "__main__":
    main()
