"""Table 6 — Point-probe controls: spectral-point and context-matched baselines.

Reads results/{basin}_point_probes{VARIANT}.json (produced by src/point_probes.py) and
reports AUC ROC of A, SPEC, A+SPEC and ACTX with fold-level paired t-intervals
of the difference against A, plus the paired difference of the stored FM runs
against ACTX (environment caveat noted in the manuscript).
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path

from scipy.stats import t as student_t

# See make_tab3_benchmark.py: suffix of the result files to read, empty for the
# primary 5-fold analysis.
VARIANT = os.environ.get("TABLE_VARIANT", "")

from paths import RESULTS
from paths import TABLES_DIR as OUT_DIR

BASINS = [("06_rio_huasco", "Huasco", "Semi-arid"),
          ("09_rio_maipo",  "Maipo",  "Mediterranean"),
          ("11_rio_maule",  "Maule",  "Temperate-humid")]

SETS = [("SPEC", "Spec.-point (8)"),
        ("A_SPEC", "A + spec.-point (25)"),
        ("ACTX", "A + terrain ctx (119)"),
        ("SCTX", "Spec. + ctx (56)"),
        ("AFULL", "A + full ctx (175)")]

_FOLD_COUNTS = set()


def t_crit(n):
    """Two-sided 95% t quantile for n folds, derived from the data."""
    _FOLD_COUNTS.add(int(n))
    return float(student_t.ppf(0.975, n - 1))


def tci(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    h = t_crit(n) * sd / math.sqrt(n)
    return m, m - h, m + h


def fmt_delta(deltas, star="$^{*}$"):
    m, lo, hi = tci(deltas)
    sig = star if not (lo <= 0 <= hi) else ""
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"


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
        r"\caption{Point-probe controls on the identical dataset and spatial folds of the benchmark. \textbf{A}: 17 geomorphometric features at the candidate pixel. \textbf{Spectral-point}: six HLS-equivalent bands plus NDVI and NBR sampled at the candidate pixel from the same $2023$ Sentinel-2 composite used by the FM pipelines (post-event scar-signal probe). Context variants augment the point features with nan-aware mean and standard deviation of each layer over $7\times7$, $37\times37$ and $111\times111$ pixel windows ($0.2$--$3.3$~km): \textbf{A + terrain context} extends the geomorphometric layers, \textbf{spectral point + context} the composite layers, and \textbf{A + full context} both --- the fully context-matched manual comparator to the FM pipelines. $\Delta$ROC is against A on identical folds with fold-level paired $95\%$ $t$-intervals (df $= 4$); $^{*}$: interval excludes zero. The last column reports the paired fold-level difference of the strongest FM pipeline in each basin against A + full context; FM embeddings were re-encoded in the same software environment as the controls, so all comparisons are within-environment and fold-paired.}",
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
        d = json.loads((RESULTS / f"{b_slug}_point_probes{VARIANT}.json").read_text())
        fm = d["fold_metrics"]
        A_roc, A_pr = fm["A"]["roc"], fm["A"]["pr"]
        a_mean = sum(A_roc) / len(A_roc)

        # strongest stored FM by mean roc
        best_key, best_mean = None, -1.0
        for key in ("TM_DEM_ENV", "TM_MM_ENV", "PRITHVI_ENV"):
            if key in fm:
                mmean = sum(fm[key]["roc"]) / len(fm[key]["roc"])
                if mmean > best_mean:
                    best_key, best_mean = key, mmean
        fmlab = {"TM_DEM_ENV": "TM+DEM", "TM_MM_ENV": "TM+DEM+S2", "PRITHVI_ENV": "Prithvi+S2"}.get(best_key, "?")
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
    (OUT_DIR / f"tab6_point_probes{VARIANT}.tex").write_text(apply_wording("\n".join(lines)))
    print(f"  wrote tab6_point_probes{VARIANT}.tex")

    md = ["# Table 6 — Point-probe controls", "",
          "| Basin | Probe | AUC ROC | ΔROC vs A | ΔPR vs A | best FM vs A+full context |",
          "|---|---|---|---|---|---|"] + md_rows + [
          "", "*: fold-level paired 95% t-CI (df=4) excludes zero."]
    (OUT_DIR / f"tab6_point_probes{VARIANT}.md").write_text(apply_wording("\n".join(md)))
    print(f"  wrote tab6_point_probes{VARIANT}.md")


if __name__ == "__main__":
    main()
