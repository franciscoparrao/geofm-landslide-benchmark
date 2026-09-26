"""Generate LaTeX tables for the CAGEO paper from result JSONs.

Outputs go to paper/tables/ as `tabN.tex` (input-ready snippets) plus a
`tabN.md` Markdown rendering for in-text preview.

Tables produced (initial set):
  Table 3 — Benchmark of TerraMind+DEM (pretrained) vs 17-feature baseline,
            spatial holdout, three basins.
  Table 4 — Random-initialization ablation: A vs B-pretrained vs B-random,
            same three basins.
"""
from __future__ import annotations

import json
from pathlib import Path

from paths import RESULTS
from paths import TABLES_DIR as OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASINS = ["06_rio_huasco", "09_rio_maipo", "11_rio_maule"]
SHORT = {"06_rio_huasco": "Huasco", "09_rio_maipo": "Maipo", "11_rio_maule": "Maule"}
REGIME = {"06_rio_huasco": "Semi-arid", "09_rio_maipo": "Mediterranean",
          "11_rio_maule": "Temperate-humid"}


def fmt_mean_std(m, s):
    return f"{m:.3f} $\\pm$ {s:.3f}"


def fmt_delta(mean, std):
    return f"{mean:+.3f} $\\pm$ {std:.3f}"


def fmt_ci(ci_low, ci_high):
    sig = "$^{*}$" if not (ci_low <= 0 <= ci_high) else ""
    return f"[{ci_low:+.3f}, {ci_high:+.3f}]{sig}"


# ============================================================================
# TABLE 3 — Benchmark A vs B (pretrained)
# ============================================================================
def table3():
    rows = []
    for b in BASINS:
        d = json.loads((RESULTS / f"{b}_terramind_linprobe_spatial.json").read_text())
        # fold-level delta std: stored in JSON if available, else compute from folds
        if "delta_roc_std" in d:
            d_roc_std = d["delta_roc_std"]
            d_pr_std = d["delta_pr_std"]
        else:
            droc = [f["delta_roc"] for f in d["fold_results"]]
            dpr = [f["delta_pr"] for f in d["fold_results"]]
            import statistics
            d_roc_std = statistics.pstdev(droc)
            d_pr_std = statistics.pstdev(dpr)
        rows.append({
            "basin": SHORT[b],
            "regime": REGIME[b],
            "a_roc": fmt_mean_std(d["A_roc_mean"], d["A_roc_std"]),
            "b_roc": fmt_mean_std(d["B_roc_mean"], d["B_roc_std"]),
            "d_roc": fmt_delta(d["delta_roc_mean"], d_roc_std),
            "ci_roc": fmt_ci(*d["delta_roc_ci_95"]),
            "a_pr":  fmt_mean_std(d["A_pr_mean"],  d["A_pr_std"]),
            "b_pr":  fmt_mean_std(d["B_pr_mean"],  d["B_pr_std"]),
            "d_pr":  fmt_delta(d["delta_pr_mean"], d_pr_std),
            "ci_pr":  fmt_ci(*d["delta_pr_ci_95"]),
        })

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Benchmark of TerraMind v1-tiny + DEM (pipeline B, 192-dim embedding) "
        r"against the 17-feature geomorphometric baseline (pipeline A) for landslide "
        r"susceptibility prediction in three Chilean basins. AUC values are 5-fold "
        r"spatial-holdout means $\pm$ standard deviation across folds. $\Delta$AUC is the "
        r"fold-mean difference $\pm$ across-fold standard deviation. The 95\% confidence "
        r"interval is estimated by paired bootstrap (N=1000) on the concatenated test-set "
        r"predictions, which is statistically more powerful than the fold-aggregate and "
        r"thus is the reference for significance testing. Asterisks denote intervals not "
        r"containing zero.}",
        r"\label{tab:benchmark}",
        r"\small",
        r"\begin{tabular}{llcccc@{\hspace{0.6em}}cccc}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{AUC ROC} & \multicolumn{4}{c}{AUC PR} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"Basin & Regime & A & B & $\Delta$ROC & 95\% CI & A & B & $\Delta$PR & 95\% CI \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['basin']} & {r['regime']} & "
            f"{r['a_roc']} & {r['b_roc']} & {r['d_roc']} & {r['ci_roc']} & "
            f"{r['a_pr']} & {r['b_pr']} & {r['d_pr']} & {r['ci_pr']} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    tex = "\n".join(lines)
    (OUT_DIR / "tab3_benchmark.tex").write_text(tex)
    print("  wrote tab3_benchmark.tex")

    md_lines = [
        "# Table 3 — Benchmark A vs TerraMind+DEM (pretrained), spatial holdout",
        "",
        "| Basin | Regime | A ROC | B ROC | ΔROC | 95% CI | A PR | B PR | ΔPR | 95% CI |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        def md(s):
            return (s.replace("$^{*}$", "*").replace("$\\pm$", "±")
                    .replace("{\\footnotesize ", "").replace("}", ""))
        md_lines.append(
            f"| {r['basin']} | {r['regime']} | {md(r['a_roc'])} | {md(r['b_roc'])} | "
            f"{md(r['d_roc'])} | {md(r['ci_roc'])} | "
            f"{md(r['a_pr'])} | {md(r['b_pr'])} | {md(r['d_pr'])} | {md(r['ci_pr'])} |"
        )
    md_lines.extend(["",
        "ΔAUC is the fold-mean ± across-fold standard deviation; 95% CI is from paired bootstrap (N=1000).",
        "*: 95% CI does not contain zero."])
    (OUT_DIR / "tab3_benchmark.md").write_text("\n".join(md_lines))
    print("  wrote tab3_benchmark.md")


# ============================================================================
# TABLE 4 — Random-init ablation
# ============================================================================
def table4():
    rows = []
    for b in BASINS:
        pre = json.loads((RESULTS / f"{b}_terramind_linprobe_spatial.json").read_text())
        rnd = json.loads((RESULTS / f"{b}_terramind_linprobe_spatial_randinit.json").read_text())
        gap_roc = pre["B_roc_mean"] - rnd["B_roc_mean"]
        gap_pr  = pre["B_pr_mean"]  - rnd["B_pr_mean"]
        rows.append({
            "basin": SHORT[b],
            "regime": REGIME[b],
            "a_roc":   fmt_mean_std(pre["A_roc_mean"], pre["A_roc_std"]),
            "bp_roc":  fmt_mean_std(pre["B_roc_mean"], pre["B_roc_std"]),
            "br_roc":  fmt_mean_std(rnd["B_roc_mean"], rnd["B_roc_std"]),
            "gap_roc": f"{gap_roc:+.3f}",
            "a_pr":    fmt_mean_std(pre["A_pr_mean"], pre["A_pr_std"]),
            "bp_pr":   fmt_mean_std(pre["B_pr_mean"], pre["B_pr_std"]),
            "br_pr":   fmt_mean_std(rnd["B_pr_mean"], rnd["B_pr_std"]),
            "gap_pr":  f"{gap_pr:+.3f}",
        })

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Random-initialization ablation. \textbf{A}: Random Forest on 17 "
        r"hand-engineered geomorphometric features. \textbf{B-pretrained}: Random "
        r"Forest on 192-dim embeddings from TerraMind v1-tiny with pretrained weights. "
        r"\textbf{B-random}: identical architecture and pipeline but with Xavier-"
        r"uniform random initialization (no pretraining). The \emph{gap} column "
        r"isolates the contribution of pretraining: pretrained $-$ random.}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{llcccc@{\hspace{0.6em}}cccc}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{AUC ROC} & \multicolumn{4}{c}{AUC PR} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"Basin & Regime & A & B-pretr. & B-rand. & gap & A & B-pretr. & B-rand. & gap \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['basin']} & {r['regime']} & "
            f"{r['a_roc']} & {r['bp_roc']} & {r['br_roc']} & {r['gap_roc']} & "
            f"{r['a_pr']}  & {r['bp_pr']}  & {r['br_pr']}  & {r['gap_pr']} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    tex = "\n".join(lines)
    (OUT_DIR / "tab4_ablation.tex").write_text(tex)
    print("  wrote tab4_ablation.tex")

    md_lines = [
        "# Table 4 — Random-initialization ablation",
        "",
        "| Basin | Regime | A ROC | B-pretr ROC | B-rand ROC | gap ROC | A PR | B-pretr PR | B-rand PR | gap PR |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_a_roc = r['a_roc'].replace("$\\pm$", "±")
        md_bp_roc = r['bp_roc'].replace("$\\pm$", "±")
        md_br_roc = r['br_roc'].replace("$\\pm$", "±")
        md_a_pr = r['a_pr'].replace("$\\pm$", "±")
        md_bp_pr = r['bp_pr'].replace("$\\pm$", "±")
        md_br_pr = r['br_pr'].replace("$\\pm$", "±")
        md_lines.append(
            f"| {r['basin']} | {r['regime']} | {md_a_roc} | {md_bp_roc} | {md_br_roc} | {r['gap_roc']} | "
            f"{md_a_pr} | {md_bp_pr} | {md_br_pr} | {r['gap_pr']} |"
        )
    (OUT_DIR / "tab4_ablation.md").write_text("\n".join(md_lines))
    print("  wrote tab4_ablation.md")


if __name__ == "__main__":
    print("Generating Table 3: benchmark A vs B")
    table3()
    print("Generating Table 4: random-init ablation")
    table4()
    print(f"\nTables saved to {OUT_DIR}")
