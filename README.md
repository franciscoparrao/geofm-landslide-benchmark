# geofm-landslide-benchmark

Reproducibility package for:

> **Where do geospatial foundation models help? Cross-model evaluation of
> TerraMind and Prithvi-EO-2.0 for landslide susceptibility across the Chilean
> climate gradient.** Parra, F., Gil-Costa, V., Bonacic, C., Marín, M.
> Submitted to *Computers & Geosciences*.

The package implements a cross-FM linear-probe benchmark with a
random-initialization ablation and point-probe controls (context-matched
manual baselines; post-event spectral scar probe; constrained-negative
sensitivity), evaluated under spatial-block cross-validation with a
fold-level paired *t*-inference.

## Layout

```
src/                      benchmark pipeline
  terramind_linprobe.py   cross-FM linear probe (--encoder terramind_v1_tiny | prithvi-300m,
                          --cv spatial, --init pretrained|random, --modalities DEM S2L2A,
                          --folds N, --splitter group|stratified-group, --encode-only)
  point_probes.py         point-probe controls (SPEC / A+context / A+full context),
                          constrained-negative sensitivity (--negatives constrained),
                          within-environment FM re-encoding (--with-fms all|terramind),
                          --folds N / --splitter as above
  pre_post_event_test.py  pre-/post-event composite test of the scar caveat (Huasco):
                          a 2016 composite predates the 2017+2020 events, so swapping
                          only the composite isolates post-event signal
  build_stack.py          17-layer geomorphometric feature stack -> {basin}_stack.npz
  inventory_temporal_analysis.py  trigger/temporal composition of the inventories
  pca_kmeans_baseline.py, umap_hdbscan.py, select_k.py,
  bootstrap_stability.py, compare_basins.py   unsupervised characterization
paper_artifacts/          table (make_tab*.py), figure (make_fig*.py) and
                          supplementary fold-ladder (make_fold_ladder.py) generators;
                          paths.py resolves input/output locations,
                          Sentinel-2 L2A composite builder (download_s2_composite.py:
                          --year / --months / --harmonize for season-matched,
                          radiometrically harmonized composites)
data/                     see data/README.md for how to obtain all inputs
```

## Requirements

Python ≥ 3.12, CPU-only (no GPU required). Install:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Model weights download automatically from HuggingFace on first use
(TerraMind: `ibm-esa-geospatial/TerraMind-1.0-tiny` via `terratorch`;
Prithvi: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`, place under a local
directory and pass `--prithvi-path`). Both are Apache-2.0 licensed by their
original authors.

Note: on CPU, Prithvi-EO-2.0-300M encodes at roughly 6–7 s/patch in fp32
(bf16 is slower on CPUs without native support); a full basin takes hours.
TerraMind v1-tiny encodes a basin in about one minute.

## Reproducing the paper

### Tables and figures, without re-running anything

`results/` ships the per-fold result files the paper reports (121 JSON, under
1 MB), so every table and figure regenerates offline, with no imagery download,
no GPU and no model weights:

```bash
GEOFM_RESULTS_DIR=results GEOFM_TABLES_DIR=out \
  python paper_artifacts/make_tab3_benchmark.py
```

This is the fastest way to check a number in the manuscript against its source.
The heavy artifacts — patch memmaps, embedding caches, composites and
susceptibility rasters — are not versioned here; the steps below rebuild them
from the raw inputs.

### From raw inputs

1. Obtain inputs (see `data/README.md`). Every location defaults to a path
   inside this package, so placing or symlinking the inputs under `data/` needs
   no code edits; alternatively point the environment variables in
   `data/README.md` at trees held elsewhere.
2. Build the feature stacks: `python src/build_stack.py --basin 06_rio_huasco`
   (repeat per basin).
3. Build the Sentinel-2 composites:
   `python paper_artifacts/download_s2_composite.py --basin 06_rio_huasco`.
4. Benchmark (Tables 3–5, Figs. 4–6), per basin × encoder × init:
   ```bash
   python src/terramind_linprobe.py --basin 06_rio_huasco --cv spatial
   python src/terramind_linprobe.py --basin 06_rio_huasco --cv spatial --modalities DEM S2L2A
   python src/terramind_linprobe.py --basin 06_rio_huasco --cv spatial --encoder prithvi-300m
   # add --init random for the ablation arm
   ```
5. Point-probe controls (Tables 6–7):
   ```bash
   python src/point_probes.py --basin 06_rio_huasco                  # uniform negatives
   python src/point_probes.py --basin 06_rio_huasco --negatives constrained
   # add --with-fms all to re-encode the FM pipelines in the same environment
   ```
6. Pre-/post-event scar test (Huasco). Build the matched composites first:
   ```bash
   python paper_artifacts/download_s2_composite.py --basin 06_rio_huasco \
       --year 2016 --months 1-6 --max-items 60 --harmonize --suffix _matched
   python paper_artifacts/download_s2_composite.py --basin 06_rio_huasco \
       --year 2023 --months 1-6 --max-items 60 --harmonize --suffix _matched
   python src/pre_post_event_test.py
   ```
   Note: Sentinel-2 processing baseline 04.00 (January 2022) introduced a
   1000 DN BOA_ADD_OFFSET. `--harmonize` removes it so that pre- and post-2022
   acquisitions share a radiometric scale; compositing across that boundary
   without it produces a spurious brightness difference between years.
7. Regenerate tables and figures: run the `paper_artifacts/make_*.py` scripts.
8. Supplementary fold-count ladder. The primary inference uses 5 spatial folds
   with `GroupKFold`; the supplement reports a 10-fold variant, and a 5-fold
   `StratifiedGroupKFold` rung in between so that the change of splitter and
   the change of fold count are not confounded. Both are post-hoc.
   ```bash
   # bridge rung (splitter only) and variant rung (splitter + fold count)
   python src/terramind_linprobe.py --basin 06_rio_huasco --cv spatial \
       --splitter stratified-group --folds 5
   python src/terramind_linprobe.py --basin 06_rio_huasco --cv spatial \
       --splitter stratified-group --folds 10
   python paper_artifacts/make_fold_ladder.py     # -> fold_ladder.md
   # supplementary tables read the variant through an env var
   TABLE_VARIANT=_sgkf_k10 python paper_artifacts/make_tab3_benchmark.py
   ```
   `--splitter stratified-group` is required at 10 folds: Huasco's events
   concentrate in few 10 km blocks, so plain `GroupKFold` yields a
   positive-free test fold and an undefined AUC. Results carry a `_sgkf`
   and/or `_k<folds>` suffix, so no variant can overwrite the primary run.
   `make_fold_ladder.py` also reports how much of the narrowing at 10 folds is
   the mechanical `t(df)/sqrt(n)` factor rather than added information.

All seeds are fixed (42; per-fold seeds 42+fold). Outputs are standardized
JSON files per (basin, encoder, initialization) cell.

Patch embeddings are cached under `results/_embcache/`, keyed by a fingerprint
of the sampling parameters and the sampled coordinates. A re-run that changes
only the cross-validation scheme reuses them; anything that moves the points
invalidates the cache and re-encodes. Use `--encode-only` to populate the cache
without running cross-validation. This matters because Prithvi-EO-2.0-300M
takes hours per basin on CPU.

## Extending to another foundation model

`terramind_linprobe.py` isolates the encoder behind `build_encoder()` /
`encode_patches()`; adding a HuggingFace-released model requires an adapter
of ~30 lines returning `(model, kind, embed_dim, norm_stats)` and an
encode function mapping patches to per-point embeddings.

## License

MIT (see `LICENSE`). Model weights and external datasets keep their original
licenses and access conditions.
