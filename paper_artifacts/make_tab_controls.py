"""Table: the Maule deficit under every control, from the released result files.

One row per design. Each cell is the fold-mean dROC of the foundation-model
pipeline against the manual comparator of that design, with the fold-level
paired 95% t-interval; * marks an interval that excludes zero.
"""
from __future__ import annotations

import json
import math

from scipy.stats import t as student_t

from paths import RESULTS, TABLES_DIR


def tci(v, k=None):
    """Fold mean, paired t-interval, and whether the Nadeau-Bengio corrected
    interval (variance factor 1/k + 1/(k-1)) also excludes zero."""
    n = len(v)
    k = k or n
    m = sum(v) / n
    s2 = sum((x - m) ** 2 for x in v) / (n - 1)
    tq = float(student_t.ppf(0.975, n - 1))
    h = tq * math.sqrt(s2 / n)
    h_nb = tq * math.sqrt((1 / k + 1 / (k - 1)) * s2)
    return m, m - h, m + h, not (m - h_nb <= 0 <= m + h_nb)


def cell(m, lo, hi, nb):
    star = "$^{*}$" if (lo > 0 or hi < 0) else ""
    dag = "$^{\\dagger}$" if nb else ""
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}{dag}"


def diff(a, b):
    return [x - y for x, y in zip(a, b)]


def load(name):
    return json.loads((RESULTS / name).read_text())


def bench(stem, suffix=""):
    return [f["delta_roc"] for f in load(f"11_rio_maule_{stem}{suffix}.json")["fold_results"]]


def main():
    rows = []
    rows.append(("Primary design (10 km blocks, GroupKFold)", "A",
                 tci(bench("terramind_linprobe_spatial")), tci(bench("prithvi-300m_linprobe_spatial"))))
    rows.append(("StratifiedGroupKFold, 5 folds", "A",
                 tci(bench("terramind_linprobe_spatial", "_sgkf")), tci(bench("prithvi-300m_linprobe_spatial", "_sgkf"))))
    rows.append(("StratifiedGroupKFold, 10 folds", "A",
                 tci(bench("terramind_linprobe_spatial", "_sgkf_k10")), tci(bench("prithvi-300m_linprobe_spatial", "_sgkf_k10"))))
    for buf, lab in (("33", "1 km"), ("17", "0.5 km"), ("0", "none")):
        rows.append((f"Negative-exclusion window {lab}", "A",
                     tci(bench("terramind_linprobe_spatial", f"_buf{buf}")),
                     tci(bench("prithvi-300m_linprobe_spatial", f"_buf{buf}"))))
    pp = load("11_rio_maule_point_probes.json")["fold_metrics"]
    rows.append(("Context-matched comparator", "A+full context",
                 tci(diff(pp["TM_DEM_ENV"]["roc"], pp["AFULL"]["roc"])),
                 tci(diff(pp["PRITHVI_ENV"]["roc"], pp["AFULL"]["roc"]))))
    pc = load("11_rio_maule_point_probes_constrained.json")["fold_metrics"]
    rows.append(("Negatives in the positives' slope envelope", "A",
                 tci(diff(pc["TM_DEM_ENV"]["roc"], pc["A"]["roc"])),
                 tci(diff(pc["PRITHVI_ENV"]["roc"], pc["A"]["roc"]))))
    gm = load("11_rio_maule_ground_motion_arm.json")["sets"]
    rows.append(("Ground motion given to both sides", "A+GM",
                 tci(diff(gm["TM+DEM+GM"]["roc"], gm["A+GM"]["roc"])),
                 tci(diff(gm["Prithvi+GM"]["roc"], gm["A+GM"]["roc"]))))
    ty = load("typology_arm.json")["11_rio_maule"]["single_type"]["sets"]
    rows.append(("Single movement type", "A",
                 tci(diff(ty["TM+DEM"]["roc"], ty["A"]["roc"])),
                 tci(diff(ty["Prithvi"]["roc"], ty["A"]["roc"]))))
    for fn, key, lab in (("geology_arm.json", "GEO", "Geology given to both sides"),
                         ("climate_arm.json", "CLIM", "Climate given to both sides")):
        r = load(fn)["11_rio_maule"]["roc"]
        rows.append((lab, f"A+{key.lower()}",
                     tci(diff(r[f"TM+DEM+{key}"], r[f"A+{key}"])),
                     tci(diff(r[f"Prithvi+{key}"], r[f"A+{key}"]))))
    fb = load("floor_benchmark_folds.json")["11_rio_maule"]["roc_folds"]
    rows.append(("Elevation given to the baseline", "A+elevation",
                 tci(diff(fb["TM+DEM"], fb["A+elev"])), tci(diff(fb["Prithvi"], fb["A+elev"]))))
    lp = load("11_rio_maule_linear_probe_arm.json")["readers"]
    for r, lab in (("Logistic", "Logistic-regression reader"), ("HistGradBoost", "Gradient-boosting reader")):
        c = lp[r]["cells"]
        rows.append((lab, "A (same reader)",
                     tci(diff(c["TM+DEM"]["roc"], c["A"]["roc"])),
                     tci(diff(c["Prithvi"]["roc"], c["A"]["roc"]))))

    lines = [r"\begin{table*}", r"\centering",
             r"\caption{The Maule deficit under every design and control. $\Delta$ROC of the "
             r"foundation-model pipeline against the manual comparator of each design, fold mean "
             r"with fold-level paired $95\%$ $t$-interval; $^{*}$: interval excludes zero; "
             r"$^{\dagger}$: the Nadeau--Bengio corrected interval also excludes zero. "
             r"All rows use the corrected inputs. Detailed specifications of each control are in "
             r"the Supplementary Material.}",
             r"\label{tab:controls}", r"\small",
             r"\begin{tabular}{llcc}", r"\toprule",
             r"Design or control & Comparator & TerraMind+DEM & Prithvi-EO-2.0+S2 \\", r"\midrule"]
    for name, comp, tm, pr in rows:
        lines.append(f"{name} & {comp} & {cell(*tm)} & {cell(*pr)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABLES_DIR / "tab_controls.tex").write_text("\n".join(lines) + "\n")
    print("[write] tab_controls.tex")
    for name, comp, tm, pr in rows:
        print(f"  {name:45s} TM {cell(*tm):35s} P {cell(*pr)}")


if __name__ == "__main__":
    main()
