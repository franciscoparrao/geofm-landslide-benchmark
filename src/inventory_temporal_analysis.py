"""Temporal and trigger characterization of the landslide inventories.

For each benchmarked basin, reports the distribution of event years and
trigger types from the SERNAGEOMIN basin inventory, quantifying how many
events pre-date (a) the Sentinel-2 era (2015) and (b) the 2023 composite
used by the spectral pipelines. Supports the post-event-imagery caveat
(review Issue 1) with actual numbers.
"""
from __future__ import annotations
import csv
import json
from collections import Counter

from config import INVENTORY_BASE, ML_DATASET_BASE, RESULTS

BASINS = ["06_rio_huasco", "09_rio_maipo", "11_rio_maule"]


def main():
    summary = {}
    for basin in BASINS:
        years, triggers, types = [], Counter(), Counter()
        with open(INVENTORY_BASE / f"{basin}.csv") as f:
            for r in csv.DictReader(f):
                yr = r.get("year", "").strip()
                if yr:
                    try:
                        years.append(int(float(yr)))
                    except ValueError:
                        pass
                triggers[r.get("trigger", "?").strip() or "?"] += 1
                types[r.get("type", "?").strip() or "?"] += 1

        # sources actually present in the ml_dataset positives
        sources = Counter()
        ml = ML_DATASET_BASE / f"{basin}.csv"
        if ml.exists():
            with open(ml) as f:
                for r in csv.DictReader(f):
                    if r.get("label") == "1":
                        sources[r.get("source", "?")] += 1

        years.sort()
        n = len(years)
        stats = {
            "n_inventory_rows": sum(types.values()),
            "n_with_year": n,
            "year_min": years[0] if years else None,
            "year_max": years[-1] if years else None,
            "year_median": years[n // 2] if years else None,
            "pct_pre_2015": round(100 * sum(y < 2015 for y in years) / n, 1) if n else None,
            "pct_pre_2023": round(100 * sum(y < 2023 for y in years) / n, 1) if n else None,
            "triggers": dict(triggers.most_common()),
            "types": dict(types.most_common()),
            "ml_dataset_positive_sources": dict(sources.most_common()),
        }
        summary[basin] = stats
        print(f"\n=== {basin} ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    out = RESULTS / "inventory_temporal_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
