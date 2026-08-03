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
                          --cv spatial, --init pretrained|random, --modalities DEM S2L2A)
  point_probes.py         point-probe controls (SPEC / A+context / A+full context),
                          constrained-negative sensitivity (--negatives constrained),
                          within-environment FM re-encoding (--with-fms all|terramind)
  build_stack.py          17-layer geomorphometric feature stack -> {basin}_stack.npz
  inventory_temporal_analysis.py  trigger/temporal composition of the inventories
  pca_kmeans_baseline.py, umap_hdbscan.py, select_k.py,
  bootstrap_stability.py, compare_basins.py   unsupervised characterization
paper_artifacts/          table (make_tab*.py) and figure (make_fig*.py) generators,
                          Sentinel-2 L2A composite builder (download_s2_composite.py)
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

1. Obtain inputs (see `data/README.md`) and edit the paths at the top of
   `src/config.py` and `src/terramind_linprobe.py`.
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
6. Regenerate tables and figures: run the `paper_artifacts/make_*.py` scripts.

All seeds are fixed (42; per-fold seeds 42+fold). Outputs are standardized
JSON files per (basin, encoder, initialization) cell.

## Extending to another foundation model

`terramind_linprobe.py` isolates the encoder behind `build_encoder()` /
`encode_patches()`; adding a HuggingFace-released model requires an adapter
of ~30 lines returning `(model, kind, embed_dim, norm_stats)` and an
encode function mapping patches to per-point embeddings.

## License

MIT (see `LICENSE`). Model weights and external datasets keep their original
licenses and access conditions.
