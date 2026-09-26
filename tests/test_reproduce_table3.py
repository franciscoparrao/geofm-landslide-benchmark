"""Regenerate Table 3 from the committed results and compare it to the paper.

This test exists because the package once claimed, in the manuscript, that
every number regenerated from released result files while shipping no result
files at all and hardcoding an absolute home directory in every generator. A
clean clone failed on the first read and nobody noticed until peer review. One
test that runs a generator against the committed data and diffs the output would
have caught the whole failure class on the first commit, so here it is.

What it checks, in order of how badly it would have failed before:

1. The committed results/ actually contains the per-fold files the generators
   read. (Previously: zero files.)
2. A generator runs from a checkout with no access to any private tree, using
   only the environment overrides the README documents. (Previously: it silently
   read the author's home directory, or raised FileNotFoundError elsewhere.)
3. The regenerated table matches the committed one row for row. (This is what
   makes "every number regenerates" a checkable statement rather than a promise.)
4. The headline cells carry the values the manuscript reports.

Run:  python -m pytest tests/ -v
  or: python tests/test_reproduce_table3.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COMMITTED_TABLE = ROOT / "paper_artifacts" / "tab3_benchmark.tex"
GENERATOR = ROOT / "paper_artifacts" / "make_tab3_benchmark.py"

# The two cells the manuscript's abstract and conclusions rest on. If a
# dependency bump moves these, the paper's claims move with them and the
# authors need to know before a reader does.
HEADLINE = {
    ("Maipo", "Prithvi-EO-2.0+S2"): "+0.124",
    ("Maule", "Prithvi-EO-2.0+S2"): "-0.085",
}


def _rows(tex: str) -> dict[tuple[str, str], str]:
    """Map (basin, pipeline) -> the Delta ROC column, from a LaTeX table body."""
    out: dict[tuple[str, str], str] = {}
    basin = None
    for line in tex.splitlines():
        if "&" not in line or line.lstrip().startswith("%"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 6:
            continue
        if cells[0] and not cells[0].startswith("\\"):
            basin = cells[0]
        pipeline = cells[2]
        delta = cells[5]
        if basin and pipeline and re.match(r"^[+-]?\d", delta):
            out[(basin, pipeline)] = delta
    return out


def test_results_are_committed() -> None:
    jsons = sorted(RESULTS.glob("*.json"))
    assert jsons, (
        f"{RESULTS} contains no result files. The manuscript claims every "
        "number regenerates from them; without them that claim is false and "
        "no table below can be rebuilt."
    )
    assert len(jsons) >= 100, f"only {len(jsons)} result files; expected the full set"


def test_table3_regenerates_and_matches() -> None:
    assert GENERATOR.exists(), f"missing generator: {GENERATOR}"
    assert COMMITTED_TABLE.exists(), (
        f"missing committed table: {COMMITTED_TABLE}. Commit the generated "
        ".tex so the regenerated one has something to be compared against."
    )

    with tempfile.TemporaryDirectory() as tmp:
        env = {
            **os.environ,
            "GEOFM_RESULTS_DIR": str(RESULTS),
            "GEOFM_TABLES_DIR": tmp,
            "MPLBACKEND": "Agg",
        }
        proc = subprocess.run(
            [sys.executable, str(GENERATOR)],
            cwd=GENERATOR.parent, env=env,
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, (
            "the generator failed from a plain checkout — this is the failure "
            f"a reader hits on a fresh clone:\n{proc.stderr[-1500:]}"
        )

        produced = Path(tmp) / "tab3_benchmark.tex"
        assert produced.exists(), (
            f"generator wrote nothing to GEOFM_TABLES_DIR. If it ignored the "
            f"override it may have written elsewhere — check for hardcoded "
            f"paths.\nstdout:\n{proc.stdout[-800:]}"
        )

        got = _rows(produced.read_text())
        want = _rows(COMMITTED_TABLE.read_text())
        assert got, "parsed no data rows from the regenerated table"

        drifted = {k: (want[k], got[k]) for k in want if k in got and want[k] != got[k]}
        assert not drifted, (
            "regenerated Table 3 disagrees with the committed one:\n"
            + "\n".join(f"  {b} / {p}: committed {w}, regenerated {g}"
                        for (b, p), (w, g) in drifted.items())
        )

        for key, expected in HEADLINE.items():
            assert key in got, f"headline cell missing from regenerated table: {key}"
            assert got[key] == expected, (
                f"{key[0]} / {key[1]}: manuscript reports {expected}, "
                f"regenerated {got[key]}"
            )


if __name__ == "__main__":
    test_results_are_committed()
    test_table3_regenerates_and_matches()
    print("ok — Table 3 regenerates from the committed results and matches")
