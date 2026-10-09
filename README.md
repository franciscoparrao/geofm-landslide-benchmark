# geofm-landslide-benchmark

Reproducibility package for:

> **What survives an audit: evaluating geospatial foundation models for
> landslide susceptibility against a geomorphometric baseline.** Parra, F.,
> Gil-Costa, V., Bonacic, C., Marín, M. In preparation for *Earth Science
> Informatics*.

The package implements a cross-FM linear-probe benchmark with a
random-initialization ablation and point-probe controls (context-matched
manual baselines; post-event spectral scar probe; constrained-negative
sensitivity), evaluated under spatial-block cross-validation with a
fold-level paired *t*-inference and the Nadeau–Bengio correction. Since v1.4.0
it also ships the audit re-analyses: a positional floor on the benchmark folds
for every pipeline, the range of spatial dependence against the block size, the
variance of the negative draw and of the block-grid placement, and readers
tuned by nested spatial cross-validation.

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
  floor_benchmark_folds.py  positional floor (coordinates-only forest) on the benchmark
                          folds for every pipeline, plus the A+elev arm
  dependence_range.py     variograms of residuals, labels and covariates (range vs block)
  resampling_variance.py  offsets: 9 placements of the block grid;
                          draws: fresh negatives (--n-draws, --encode, --basin)
  tuned_reader.py         RF / gradient boosting / logistic tuned by nested spatial CV
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

`results/` ships the per-fold result files the paper reports (191 JSON, about
1.3 MB), so every table and figure regenerates offline, with no imagery download,
no GPU and no model weights:

```bash
GEOFM_RESULTS_DIR=results GEOFM_TABLES_DIR=out \
  python paper_artifacts/make_tab3_benchmark.py
```

This is the fastest way to check a number in the manuscript against its source.
The heavy artifacts — patch memmaps, embedding caches, composites and
susceptibility rasters — are not versioned here; the steps below rebuild them
from the raw inputs.

To verify that the claim holds rather than taking it on trust:

```bash
pip install "scipy==1.16.3"        # the whole dependency for this path
python tests/test_reproduce_table3.py
```

The test regenerates Table 3 from `results/` and diffs it against the committed
`paper_artifacts/tab3_benchmark.tex`, row by row, failing with the offending
cell named. It runs in CI on every push (`.github/workflows/reproduce.yml`) on a
machine that has never seen the authors' filesystem — which is the case that
matters, since the defect this guards against reproduced perfectly on the
machine that produced it and not at all anywhere else.

`requirements.txt` pins exact versions. Relaxing them is fine; re-run the test
afterwards and it will tell you whether any published number moved.

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

9. Audit re-analyses (Tables of the positional floor and the dependence range,
   the Maule controls table, the audit figures, Supplement S5). All run on CPU
   from the cached embeddings, except `--encode`, which encodes the new
   negatives of each draw (about one point per second for Prithvi on CPU).
   ```bash
   python src/floor_benchmark_folds.py                # -> floor_benchmark_folds.json
   python src/dependence_range.py                     # -> dependence_range.json
   python src/resampling_variance.py offsets          # -> resampling_offsets.json
   python src/resampling_variance.py draws --n-draws 20
   python src/resampling_variance.py draws --n-draws 10 --encode --basin 11_rio_maule
   python src/tuned_reader.py --basin 11_rio_maule    # -> 11_rio_maule_tuned_reader.json
   python paper_artifacts/make_tab_floor.py
   python paper_artifacts/make_tab_range.py
   python paper_artifacts/make_tab_controls.py
   python paper_artifacts/make_fig_audit.py
   ```
   `GEOFM_N_JOBS` and `GEOFM_TORCH_THREADS` cap the cores the forests and the
   encoders use. `make_fig_audit.py` also reads the per-fold files of the three
   earlier input-pipeline stages, shipped under `results/_globaldraw/`,
   `results/_unstandardized/` and `results/_prithvi_offset/`.

   The transformer-block sweep, low-rank adaptation, the pre-/post-event test
   and Prithvi's susceptibility surfaces were computed before Prithvi's 1000 DN
   input offset was removed. Their result files are kept for traceability, but
   the manuscript does not report them until they are regenerated on corrected
   inputs. The baseline surfaces and the positional references are reported.

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
