"""Phase 1 synthesis — cross-basin consolidation.

Reads bootstrap_stability, select_k, cluster_signatures, and the all-pairs
comparison JSON, and produces:
  1) Plot: bootstrap-optimal K vs basin (in latitude order).
  2) Universal morphotypes via transitive matching: build a graph where nodes
     are (basin, cluster_id) and edges are Hungarian matches with cos>=THR;
     connected components are shared morphotypes.
  3) Heatmap: morphotype share by cuenca for the top universal morphotypes.
  4) Consolidated JSON summary.

No new K-means runs needed; uses existing K=5 artifacts.
"""
from __future__ import annotations

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from config import RESULTS

# Order: north to south (climate gradient)
BASIN_ORDER = (
    "01_rio_lluta", "06_rio_huasco", "09_rio_maipo",
    "11_rio_maule", "13_rio_bueno",
)
SHORT = {
    "01_rio_lluta": "Lluta", "06_rio_huasco": "Huasco",
    "09_rio_maipo": "Maipo", "11_rio_maule": "Maule",
    "13_rio_bueno": "Bueno",
}
REGIME = {
    "01_rio_lluta": "Hyperarid", "06_rio_huasco": "Semiarid",
    "09_rio_maipo": "Mediterranean", "11_rio_maule": "Temperate-humid",
    "13_rio_bueno": "Temperate-rainy",
}
LATITUDE = {
    "01_rio_lluta": -18.0, "06_rio_huasco": -28.5,
    "09_rio_maipo": -33.5, "11_rio_maule": -35.5,
    "13_rio_bueno": -40.5,
}
K_FOR_MATCH = 5
EDGE_THRESHOLD = 0.7  # cos for transitive matching


# ---------- 1. Optimal K from bootstrap ----------
def gather_optimal_k():
    rows = []
    for basin in BASIN_ORDER:
        path = RESULTS / f"{basin}_bootstrap_stability.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        runs = d["runs"]
        best = max(runs, key=lambda r: r["ari_mean"])
        rows.append({
            "basin": basin, "regime": REGIME[basin], "lat": LATITUDE[basin],
            "k_opt": best["k"], "ari_at_k_opt": best["ari_mean"],
            "ari_std_at_k_opt": best["ari_std"],
            "all_runs": [(r["k"], r["ari_mean"], r["ari_std"]) for r in runs],
        })
    return rows


# ---------- 2. Universal morphotypes via union-find on all-pairs ----------
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def gather_universal_morphotypes():
    pairs_path = RESULTS / f"compare_all_pairs_K{K_FOR_MATCH:02d}.json"
    data = json.loads(pairs_path.read_text())
    basins = data["basins"]

    nodes = [(b, c) for b in basins for c in range(K_FOR_MATCH)]
    uf = UnionFind(nodes)
    edge_cos = {}
    for pair in data["pairs"]:
        ba, bb = pair["basin_a"], pair["basin_b"]
        for m in pair["matches"]:
            cos = m["cosine"]
            a = (ba, m["a_cluster"]); b = (bb, m["b_cluster"])
            if cos >= EDGE_THRESHOLD:
                uf.union(a, b)
                edge_cos[(a, b)] = cos

    groups = defaultdict(list)
    for n in nodes:
        groups[uf.find(n)].append(n)

    components = []
    for root, members in groups.items():
        cuencas = sorted({m[0] for m in members})
        components.append({
            "members": sorted(members),
            "n_basins": len(cuencas),
            "basins_present": cuencas,
        })
    components.sort(key=lambda c: -c["n_basins"])
    return components, basins


def cluster_shares(components, basins):
    shares_by_basin = {}
    for basin in basins:
        sig_path = RESULTS / f"{basin}_cluster_signatures.json"
        d = json.loads(sig_path.read_text())
        run = next(r for r in d["runs"] if r["k"] == K_FOR_MATCH)
        shares = {c["id"]: c.get("share", 0.0) for c in run["clusters"]}
        shares_by_basin[basin] = shares
    return shares_by_basin


def gather_signatures(components, basins):
    sigs_by_basin = {}
    feature_names = None
    for basin in basins:
        sig_path = RESULTS / f"{basin}_cluster_signatures.json"
        d = json.loads(sig_path.read_text())
        if feature_names is None:
            feature_names = d["feature_names"]
        run = next(r for r in d["runs"] if r["k"] == K_FOR_MATCH)
        sigs = {c["id"]: np.array(c["mean"]) for c in run["clusters"]
                if c.get("size", 0) > 0}
        sigs_by_basin[basin] = sigs
    return sigs_by_basin, feature_names


# ---------- 3. Plotting ----------
def plot_optimal_k(rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax = axes[0]
    lats = [r["lat"] for r in rows]
    ks = [r["k_opt"] for r in rows]
    aris = [r["ari_at_k_opt"] for r in rows]
    err = [r["ari_std_at_k_opt"] for r in rows]
    sc = ax.scatter(lats, ks, s=120 * np.array(aris), c=aris,
                    cmap="viridis", edgecolor="black", zorder=3)
    for r in rows:
        ax.annotate(SHORT[r["basin"]], (r["lat"], r["k_opt"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Latitude (°S, more south →)")
    ax.set_ylabel("Bootstrap-optimal K")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.set_title("K-óptimo (bootstrap ARI máx) vs latitud / régimen climático")
    fig.colorbar(sc, ax=ax, label="ARI at K_opt", fraction=0.04)

    ax = axes[1]
    K_VALUES = (3, 5, 7, 10, 12)
    cmap_basins = plt.colormaps["plasma"](np.linspace(0, 1, len(rows)))
    for i, r in enumerate(rows):
        ari_by_k = {k: a for k, a, _ in r["all_runs"]}
        std_by_k = {k: s for k, _, s in r["all_runs"]}
        ks = [k for k in K_VALUES if k in ari_by_k]
        ys = [ari_by_k[k] for k in ks]
        es = [std_by_k[k] for k in ks]
        ax.errorbar(ks, ys, yerr=es, marker="o", color=cmap_basins[i],
                    label=f"{SHORT[r['basin']]} ({REGIME[r['basin']]})",
                    capsize=3, lw=1.5)
        kopt = r["k_opt"]
        ax.scatter(kopt, ari_by_k[kopt], s=200, marker="*",
                   color=cmap_basins[i], edgecolor="black", zorder=5)
    ax.set_xlabel("K"); ax.set_ylabel("Bootstrap ARI")
    ax.set_title("Curvas ARI(K) — estrella = K óptimo")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xticks(K_VALUES)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_morphotype_shares(components, shares_by_basin, basins, out_path,
                           min_basins=3):
    universal = [c for c in components if c["n_basins"] >= min_basins]
    if not universal:
        print(f"[synthesis] no components with >= {min_basins} basins")
        return None

    matrix = np.zeros((len(universal), len(basins)))
    for i, comp in enumerate(universal):
        for basin, cluster_id in comp["members"]:
            j = basins.index(basin)
            matrix[i, j] += shares_by_basin[basin].get(cluster_id, 0.0)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(universal) + 2)),
                           constrained_layout=True)
    im = ax.imshow(matrix * 100, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=max(50, matrix.max() * 100))
    ax.set_xticks(range(len(basins)))
    ax.set_xticklabels([SHORT[b] for b in basins])
    ax.set_yticks(range(len(universal)))
    ax.set_yticklabels([
        f"M{i + 1}: {c['n_basins']}/{len(basins)} basins"
        for i, c in enumerate(universal)
    ])
    for i in range(len(universal)):
        for j in range(len(basins)):
            v = matrix[i, j] * 100
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v > 25 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="share %", fraction=0.04)
    ax.set_title(f"Universal morphotypes (cos≥{EDGE_THRESHOLD}, K={K_FOR_MATCH})")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return universal, matrix


def name_morphotype(component, sigs_by_basin, feature_names):
    feats = []
    for basin, c in component["members"]:
        sig = sigs_by_basin.get(basin, {}).get(c)
        if sig is not None:
            feats.append(sig)
    if not feats:
        return "unknown", {}
    avg = np.mean(feats, axis=0)
    abs_sorted = np.argsort(np.abs(avg))[::-1][:5]
    top_features = {feature_names[i]: float(avg[i]) for i in abs_sorted}

    slope_idx = feature_names.index("slope")
    mrvbf_idx = feature_names.index("mrvbf")
    mrrtf_idx = feature_names.index("mrrtf")
    twi_idx = feature_names.index("twi")

    if avg[mrvbf_idx] > 0.5 and avg[twi_idx] > 0.3:
        label = "valley_bottom"
    elif avg[mrrtf_idx] > 0.5:
        label = "ridgetop_flat"
    elif avg[slope_idx] > 0.4:
        label = "steep_slope"
    elif avg[slope_idx] < -0.3:
        label = "lowland_flat"
    else:
        label = "mid_slope"
    return label, top_features


def main() -> None:
    rows = gather_optimal_k()
    print("\n=== Optimal K by basin (bootstrap ARI max) ===")
    for r in rows:
        print(f"  {SHORT[r['basin']]:<8} {REGIME[r['basin']]:<18} "
              f"K_opt={r['k_opt']:>2}  ARI={r['ari_at_k_opt']:.3f}±{r['ari_std_at_k_opt']:.3f}")

    components, basins = gather_universal_morphotypes()
    sigs_by_basin, feature_names = gather_signatures(components, basins)
    shares_by_basin = cluster_shares(components, basins)

    print(f"\n=== Universal morphotypes (cos≥{EDGE_THRESHOLD}, K={K_FOR_MATCH}) ===")
    universal_named = []
    for i, c in enumerate(components):
        label, top = name_morphotype(c, sigs_by_basin, feature_names)
        members_str = ", ".join(f"{SHORT[b]}-c{cid}" for b, cid in c["members"])
        c["label"] = label
        c["top_features"] = top
        if c["n_basins"] >= 1:
            print(f"  M{i + 1} [{c['n_basins']}/5 basins, label={label}] {members_str}")
        if c["n_basins"] >= 3:
            universal_named.append(c)

    out_dir = RESULTS
    plot_optimal_k(rows, out_dir / "synthesis_k_optimal_vs_climate.png")
    print(f"\n[synthesis] wrote synthesis_k_optimal_vs_climate.png")

    res = plot_morphotype_shares(components, shares_by_basin, basins,
                                 out_dir / "synthesis_morphotype_shares.png",
                                 min_basins=3)
    if res is not None:
        print("[synthesis] wrote synthesis_morphotype_shares.png")

    summary = {
        "k_for_match": K_FOR_MATCH,
        "edge_threshold": EDGE_THRESHOLD,
        "basins": basins,
        "optimal_k_by_basin": rows,
        "components": [
            {**c, "label": c.get("label"),
             "top_features": c.get("top_features", {})}
            for c in components
        ],
    }
    out_json = out_dir / "synthesis_phase1.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[synthesis] wrote {out_json.name}")


if __name__ == "__main__":
    main()
