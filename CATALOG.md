# File catalog

Inventory of every tracked file in this repository with a one-line description and a status label.

**Status labels:**

- **active** — currently in use / canonical version. Edit this file.
- **superseded by X** — older variant of an active file. Kept for git history; do not edit.
- **duplicate of X** — bit-identical to another tracked file (verified by SHA-256).
- **scratch** — unfinished sandbox or test notebook. No production output depends on it.
- **out-of-scope** — committed from an unrelated project; leave alone or move out.
- **utility** — shared helper, not a top-level entry point but actively imported.

---

## Active production scripts

| File | Description | Status |
|---|---|---|
| `image_sharpening.py` | The `FocusDiversePhaseRetrieval`, `mft_rev`, `InstrumentConfiguration` package. Imported by every FDPR runner. Originally from the Keck-2 AO bench. | **utility** |
| `propagation.py` | Wrappers around prysm propagation routines used by older scripts. | **utility** |
| `processing.py` | Light image-processing helpers (background subtraction, centroiding). | **utility** |
| `Dense_FDPR_Grid_Analysis.py` | Canonical dense FDPR runner; 80×150 sweep, 150 iterations, checkpointed every 10 rows. Produces `fdpr_intermediate_row*.npz`. | **active** |
| `OTF_FINAL.py` | Canonical OTF sideband heatmap generator. Writes `OTF_heatmap_data_new_block.npz`. | **active** |
| `MC_FINAL.py` | Canonical Monte Carlo runner for the FDPR residual heatmaps in §3.5 of the paper. Writes `MC_RMS_heatmap_CORRECTED_dz.npz`. | **active** |
| `dean_bowers.py` | Two-line implementation of equation [9] of Dean & Bowers 2003. | **active** |
| `dean_bowers_graph.py` | Plots the Dean-Bowers sideband envelope against simulated OTFs (fig5 in the paper). | **active** |
| `april28_test.ipynb` | Single-operating-point diagnostic notebook: α/δ decomposition, iteration sweep, frequency-cutoff sweep, photon sweep, φ₀ knob tests, multi-seed reproducibility. Markdown cells include interpretation guidelines for every output. | **active** |

## Active sweep / analysis scripts (added during paper preparation)

| File | Description | Status |
|---|---|---|
| `replot_paper_figures.py` | Regenerates every paper figure from production-resolution `.npz` files. Drop-in replacement for the in-Overleaf figures. | **active** |
| `resume_dense_fdpr.py` | Resumes a checkpointed `Dense_FDPR_Grid_Analysis` run from `fdpr_intermediate_row*.npz`. | **active** |
| `optimize_diversity_config.py` | Searches `(N, {Δz_i})` for the diversity configuration that minimizes FDPR residual RMS at fixed total photon budget. Run-ID-labelled outputs in `fdpr_optimization/optdz_<UTC>/`. | **active** |
| `fdpr_injectable.py` | Drop-in FDPR wrapper that exposes the forward model as a callable; used by the `_mc` scripts below. | **active** |
| `fdpr_mc.py` | Single-amplitude Monte Carlo sweep over `(Δz, ν₀)` using `fdpr_injectable`. | **active** |
| `fdpr_mc_pm.py` | Paired-magnitude (`±Δz`) variant of `fdpr_mc.py`. | **active** |
| `fdpr_mc_pm_sinamp.py` | Paired-magnitude sweep over sinusoid amplitude rather than frequency. | **active** |
| `fdpr_mc_pm_zernike.py` | Paired-magnitude sweep over Zernike-coma amplitude. | **active** |
| `analyze_zernike_sweep.py` | Post-processes the output of `fdpr_mc_pm_zernike.py` into the figures in `paper_figures/`. | **active** |
| `dz_plotting.py` | Two-line plotting helper used during interactive exploration of `MC_RMS_heatmap_*.npz`. | **active** |
| `Paper_Figures_OTF.py` | Older OTF-only figure generator. Largely subsumed by `replot_paper_figures.py` but still callable. | **active** |

## Notebook generation / edit scripts (auto-generated)

| File | Description | Status |
|---|---|---|
| `_apply_notebook_edits.py` | Reproducible source-of-truth for the markdown-cell enrichment applied to `april28_test.ipynb`. Idempotent. | **utility** |
| `_insert_output_guides.py` | Inserts the "### Reading the …" interpretation cells after each output-producing code cell in `april28_test.ipynb`. Idempotent. | **utility** |

## Superseded (older snapshots, kept for git history)

| File | Description | Status |
|---|---|---|
| `FDPRRetryAug 12.py` | Aug-2025 snapshot of the dense FDPR runner. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `FDPRno_phase_error.py` | Earlier Aug snapshot with `# FDPR Retry Aug 12` header (suggests fork-of-fork). | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `Jul21FDPR.py` | July-21 snapshot of the FDPR runner; pep8-style. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `jul30FDPR.py` | July-30 snapshot; pep8-style. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `FDPR_Monte_Carlo_Final.py` | Earlier "final" MC runner. The current canonical is `MC_FINAL.py`. | **superseded by** `MC_FINAL.py` |
| `FDPR_Noise.py` | Standalone noise-injection study; functionality now in `Dense_FDPR_Grid_Analysis.py`. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `FDPR_phase_analysis_degen.py` | Filename literally contains `degen`; phase-degeneracy study. | **superseded by** `april28_test.ipynb` |
| `FDPR_sinusoidal.py` | Older sinusoidal-aberration study. | **superseded by** `OTF_Sinusoidal_Paper.py` (which itself is on its way out -- see below) |
| `OTF_Sinusoidal_Paer.py` | **Misspelled** filename (`Paer` -> `Paper`); near-identical contents to the correctly-spelled file. | **duplicate of** `OTF_Sinusoidal_Paper.py` |
| `OTF_Sinusoidal_Paper.py` | Mid-stage OTF sinusoid analysis. Functionality migrated into `OTF_FINAL.py`. | **superseded by** `OTF_FINAL.py` |
| `MCoptimizer_degen.py` | Filename contains `degen`; earlier MC optimization attempt. | **superseded by** `optimize_diversity_config.py` |
| `montecarlo_12_15_25.py` | Dec-15-2025 dated snapshot of the MC runner. | **superseded by** `MC_FINAL.py` |
| `time_scale_FDPR_montecarlo.py` | Wall-clock profiling variant. | **superseded by** `MC_FINAL.py` |
| `Focus_Diversity_Single_Cell_Corrected.py` | "Corrected" single-cell driver, predates the dense grid runner. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `focus_diversity_processing.py` | Earlier single-cell processing pipeline. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `focus_diversity-Copy1.py` | "Copy 1" of an older driver. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `focus_diverse-Phase_retrieval_class.py` | Class-style wrapper that predates `image_sharpening.py`. | **superseded by** `image_sharpening.py` |
| `Revised_Focus_Diversity.py` | "Revised" pep8 driver; still callable but pre-`Dense_FDPR_Grid_Analysis`. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `Revised_Focus_Diversity copy.py` | macOS-style "copy" of the above. | **duplicate of** `Revised_Focus_Diversity.py` |
| `Revised_Focus_Diversity` | Zero-byte file (empty stub). | **scratch** |
| `dean_bowers_graph_highres_otf.py` | Higher-resolution variant of the Dean-Bowers plot script. Largely the same outputs as `dean_bowers_graph.py`. | **superseded by** `dean_bowers_graph.py` (or vice-versa; review before deleting) |
| `deanbowersrun.py` | pep8-style runner for the Dean-Bowers plot. | **superseded by** `dean_bowers_graph.py` |
| `figures_for_otf_paper.py` | Earlier paper-figure generator. | **superseded by** `replot_paper_figures.py` |
| `plotting_SNR_NSR.py` | SNR/NSR heatmap helper; subsumed. | **superseded by** `replot_paper_figures.py` |
| `make_fdpr_pptx.py` | Generates an SPIE-style PowerPoint of the FDPR pipeline. Peripheral. | **active** but orthogonal to the paper |
| `messingwithphase.py` | Early phase-retrieval exploration. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `neat_phase_diversity.py` | "Neat" rewrite from an earlier iteration. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `neat_phase_diversity` | Same code, no `.py` extension. | **duplicate of** `neat_phase_diversity.py` |
| `neat_phase_2.py` | "Neat" v2 rewrite. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `neat_and_original.py` | Side-by-side "neat" vs "original" comparison driver. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `neattesting.py` | Test harness for the "neat" series. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `workingneat_phase.py` | Working-copy of the "neat" rewrite. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `workingneat_jul8.py` | July-8 snapshot of the "neat" runner. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `workingneat_jul9.py` | July-9 snapshot. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `workingneat_jul17.py` | July-17 snapshot. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `single_example_run.py` | Single-cell driver used for SPIE-poster figures. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `single_run.py` | Effectively the same as `single_example_run.py`. | **duplicate of** `single_example_run.py` |
| `testing_functions.py` | Header `# TODO`; unit-test stubs. | **scratch** |
| `otf_sinusoidal_wip.py` | Explicitly marked WIP in filename. | **superseded by** `OTF_FINAL.py` |
| `fdpr_vs_AS.py` | Comparison of FDPR against Gerchberg-Saxton "AS" (alternating projection). | **active** |
| `fdpr_vs_AS_save.py` | Pickled-output variant. **Bit-identical to** `Toggle_Phase_WIP`. | **duplicate of** `fdpr_vs_AS.py` |
| `Toggle_Phase_WIP` | No-extension file; SHA-256 identical to `fdpr_vs_AS_save.py`. | **duplicate of** `fdpr_vs_AS_save.py` |
| `OTF_analysis` | No-extension file; older OTF analysis script from the remote repo. | **superseded by** `OTF_FINAL.py` |
| `Focus_diversity_v2.ipynb` | v2 of the focus-diversity notebook. | **superseded by** `april28_test.ipynb` |
| `Focus_diversity_v2_wrong_output.ipynb` | Filename literally says `wrong_output`. | **superseded by** `Focus_diversity_v2.ipynb` |
| `Original_Focus_diversity.ipynb` | The earliest version of the focus-diversity notebook. | **superseded by** `april28_test.ipynb` |
| `Testing_Original.ipynb` | Test-harness around `Original_Focus_diversity.ipynb`. | **superseded by** `april28_test.ipynb` |
| `Work for Phase Diversity.ipynb` | Earliest exploration; identifies the basic setup. | **superseded by** `april28_test.ipynb` |
| `focus_diversity.ipynb` | Mid-stage exploration. | **superseded by** `april28_test.ipynb` |
| `focus_diversity_futzing.ipynb` | Filename says `futzing`. | **scratch** |
| `focus_diversity_futzing-3.ipynb` | v3 of the same scratch. | **scratch** |
| `futzing_vs_single_cell shapes.ipynb` | Compares the two earlier styles. | **scratch** |
| `Focus Diversity Single Cell.ipynb` | Single-cell driver. | **superseded by** `april28_test.ipynb` |
| `Focus Diversity Single Cell-Copy1.ipynb` | Copy-1 of above. | **duplicate of** `Focus Diversity Single Cell.ipynb` |
| `Focus Diversity Single Cell processing.ipynb` | Single-cell driver with image-processing front-end. | **superseded by** `april28_test.ipynb` |
| `Focus Diversity Single Cell processing-Copy1.ipynb` | Copy-1. | **duplicate of** the above |
| `Focus Diversity Single Cell processing-Copy2.ipynb` | Copy-2. | **duplicate of** the above |
| `Focus Diversity Single Cell skimage.ipynb` | Same driver with `skimage`-based resize. | **superseded by** `Dense_FDPR_Grid_Analysis.py` |
| `Focus Diversity Single Cell skimage-copy.ipynb` | Copy of the skimage notebook. | **duplicate of** `Focus Diversity Single Cell skimage.ipynb` |
| `apr28tesiting2.ipynb` | April-28 "tesiting2" (misspelled); diagnostic run with SEAL connection. | **superseded by** `april28_test.ipynb` |
| `april28testing1.ipynb` | April-28 "testing1"; very large (20 MB) with embedded outputs. | **superseded by** `april28_test.ipynb` |
| `april28_test_verbose.ipynb` | Verbose-print variant of `april28_test.ipynb`. | **superseded by** `april28_test.ipynb` |
| `april28_test.html` | HTML export of `april28_test.ipynb`. Kept for offline reading. | **active** (export) |
| `jan27testing.ipynb` | Jan-27 sandbox. | **scratch** |
| `neat-testing.ipynb` | "Neat" series test notebook. | **scratch** |
| `suggested fix for focused phase function .ipynb` | Proposed fix sketched in a notebook. | **scratch** |
| `Sandbox.ipynb` | Literally named "Sandbox". | **scratch** |
| `Untitled-1.ipynb` | macOS Jupyter scratch (note the hyphen). | **scratch** |
| `Untitled.ipynb` | Jupyter scratch. **Bit-identical to** `Untitled1.ipynb`. | **duplicate of** `Untitled1.ipynb` |
| `Untitled1.ipynb` | Jupyter scratch. | **scratch** |
| `Untitled2.ipynb` | Jupyter scratch. | **scratch** |
| `Untitled3.ipynb` | Jupyter scratch. | **scratch** |
| `hwhelp.py.ipynb` | Filename ends `.py.ipynb`; appears to be coursework helper. | **out-of-scope** |
| `my_great_lib.py` | Joke filename "lets make libs great again"; not imported anywhere. | **scratch** |
| `mypythontestin` | 290-byte file, no extension, single test snippet. | **scratch** |

## Out-of-scope (unrelated projects committed in the same directory)

| File | Description | Status |
|---|---|---|
| `cred2_test/test.py` | Keck KTL keyword test; for the CRED2 detector, not the FDPR paper. | **out-of-scope** |
| `cred2_test/test_nl.py` | Same, non-linear variant. | **out-of-scope** |
| `lab2photometrystuff.py` | Coursework photometry script. | **out-of-scope** |
| `variable_star_photometry.ipynb` | Coursework variable-star photometry notebook. | **out-of-scope** |
| `.claude/skills/gitnexus/*.md` | Editor-side Claude skill files; not part of the science code. | **out-of-scope** |
| `Jul_9.jpg`, `row_array.ps`, `defocus_conversion.pdf` | Plot exports from older runs. | **out-of-scope** (consider gitignore) |

## Data / figure outputs (committed only because they predate `.gitignore`)

| Pattern | Status |
|---|---|
| `FDPRNotebooks/fig*.png` | Older paper figures. | **superseded by** `paper_figures_real/*` (gitignored) |
| `fdpr_analysis/fig_*.png` | Plot outputs from an early analysis pass. | **superseded** |
| `*.png` in root (`fdpr_alpha_vs_photons.png`, `fdpr_residual_decomp.png`, `mc_optimization_*.png`, etc.) | Diagnostic-run plot outputs from `april28_test.ipynb` and the early MC sweeps. | **scratch** (kept for now; gitignore will catch new ones) |
| `dean_heatmap.png`, `example_heatmap.png`, `heatmap_plot_out.png`, `heatmap_rms_nm.png`, `snr_comparison.png`, `spotcheck_dz1_amp7.png`, `zernike_sweep_*.png` | Older diagnostic plots. | **scratch** |

---

## Recommended cleanup pass

If you want to slim the repo (not required, but recommended before linking it from the paper's data-availability statement):

1. `git rm` every row labelled `duplicate of …` (15 files).
2. `git rm` every row labelled `scratch` that you no longer reference (notebooks named `Untitled*`, `Sandbox`, `*futzing*`, `my_great_lib.py`, `mypythontestin`, `testing_functions.py`).
3. Move every row labelled `out-of-scope` to a separate repo (or `git rm` if not needed).
4. Move the `superseded by …` files into a top-level `legacy/` directory so the active set is obvious at first glance.

After cleanup, the canonical set is ~12 active scripts + 1 notebook + the two `_apply_*` helpers, which is enough to reproduce every figure in the paper.
