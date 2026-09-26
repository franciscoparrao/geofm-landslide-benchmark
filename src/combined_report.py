"""Combined stability report across basins.

Reads {basin}_select_k.json and {basin}_bootstrap_stability.json for both
basins, generates a side-by-side panel of selection criteria + bootstrap
stability, and prints a ranked candidate table.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from config import BASINS as ALL_BASINS, RESULTS

BASINS = ALL_BASINS
COLORS = {
    "01_rio_lluta": "#e41a1c",     # red — hyperarid
    "06_rio_huasco": "#ff7f00",    # orange — semiarid
    "09_rio_maipo": "#984ea3",     # purple — mediterranean
    "11_rio_maule": "#377eb8",     # blue — temperate humid
    "13_rio_bueno": "#1b9e77",     # green — temperate rainy
}


def load_select_k(basin):
    p = RESULTS / f"{basin}_select_k.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_bootstrap(basin):
    p = RESULTS / f"{basin}_bootstrap_stability.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    select_data = {b: load_select_k(b) for b in BASINS}
    boot_data = {b: load_bootstrap(b) for b in BASINS}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    panels_select = [
        ("Silhouette (↑)", "silhouette", axes[0, 0]),
        ("Davies-Bouldin (↓)", "davies_bouldin", axes[0, 1]),
        ("GMM BIC (↓)", "gmm_bic", axes[0, 2]),
        ("Inertia / elbow", "inertia", axes[1, 0]),
    ]
    for title, key, ax in panels_select:
        for basin in BASINS:
            d = select_data[basin]
            if d is None:
                continue
            ks = [r["k"] for r in d["rows"]]
            ys = [r[key] for r in d["rows"]]
            ax.plot(ks, ys, marker="o", color=COLORS[basin], label=basin, lw=1.5)
        ax.set_xlabel("K"); ax.set_title(title); ax.grid(alpha=0.3)
        ax.set_xticks(range(2, 21, 2))
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    for basin in BASINS:
        d = boot_data[basin]
        if d is None:
            continue
        ks = [r["k"] for r in d["runs"]]
        means = np.array([r["ari_mean"] for r in d["runs"]])
        stds = np.array([r["ari_std"] for r in d["runs"]])
        ax.errorbar(ks, means, yerr=stds, marker="o", color=COLORS[basin],
                    label=basin, capsize=4, lw=1.5)
    ax.set_xlabel("K"); ax.set_ylabel("ARI"); ax.set_title("Bootstrap ARI mean ± std (↑)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(0, 1)

    ax = axes[1, 2]
    for basin in BASINS:
        d = boot_data[basin]
        if d is None:
            continue
        ks = [r["k"] for r in d["runs"]]
        means = [r["nmi_mean"] for r in d["runs"]]
        ax.plot(ks, means, marker="s", color=COLORS[basin], label=basin, lw=1.5)
    ax.set_xlabel("K"); ax.set_ylabel("NMI"); ax.set_title("Bootstrap NMI mean (↑)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(0, 1)

    fig.suptitle("Selección de K + Estabilidad bootstrap — Huasco vs Bueno", fontsize=14)
    out_png = RESULTS / "combined_stability_report.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[report] wrote {out_png}")

    print("\n=== Resumen rankings ===\n")
    for basin in BASINS:
        d = select_data[basin]
        if d is None:
            print(f"{basin}: no select_k data")
            continue
        rows = d["rows"]
        ks = [r["k"] for r in rows]
        sils = [r["silhouette"] for r in rows]
        dbs = [r["davies_bouldin"] for r in rows]
        bics = [r["gmm_bic"] for r in rows]
        top_sil = sorted(zip(sils, ks), reverse=True)[:3]
        top_db = sorted(zip(dbs, ks))[:3]
        top_bic = sorted(zip(bics, ks))[:3]
        print(f"--- {basin} ---")
        print(f"  top silhouette : {[(k, round(s, 3)) for s, k in top_sil]}")
        print(f"  top DB (low)   : {[(k, round(s, 3)) for s, k in top_db]}")
        print(f"  top BIC (low)  : {[(k, round(s, 0)) for s, k in top_bic]}")
        b = boot_data[basin]
        if b is not None:
            print(f"  bootstrap ARI  : "
                  f"{[(r['k'], round(r['ari_mean'], 3), round(r['ari_std'], 3)) for r in b['runs']]}")
        print()


if __name__ == "__main__":
    main()
