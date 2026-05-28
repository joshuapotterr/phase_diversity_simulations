# File catalog

Inventory of every tracked file in this repository after the 2026-05-28 cleanup pass.  Earlier history (duplicates, superseded snapshots, scratch notebooks) is preserved in git but no longer in the working tree; recover any of them with `git log --diff-filter=D --summary` followed by `git checkout <commit>~1 -- <path>`.

**Status labels:**

- **active** — currently in use / canonical version.  Edit this file.
- **utility** — shared helper, not a top-level entry point but actively imported.
- **historical** — kept for context (earliest exploration), not edited.
- **out-of-scope** — committed from an unrelated project; leave alone or move out.

---

## Active production scripts

| File | Description | Status |
|---|---|---|
| `image_sharpening.py` | The `FocusDiversePhaseRetrieval`, `mft_rev`, `InstrumentConfiguration` package.  Imported by every FDPR runner.  Originally from the Keck-2 AO bench. | **utility** |
| `propagation.py` | Wrappers around prysm propagation routines used by older scripts. | **utility** |
| `processing.py` | Light image-processing helpers (background subtraction, centroiding). | **utility** |
| `Dense_FDPR_Grid_Analysis.py` | Canonical dense FDPR runner; 80×150 sweep, 150 iterations, checkpointed every 10 rows.  Produces `fdpr_intermediate_row*.npz` and `fdpr_dense_completed.npz`. | **active** |
| `OTF_FINAL.py` | Canonical OTF sideband heatmap generator.  Writes `OTF_heatmap_data_new_block.npz`. | **active** |
| `MC_FINAL.py` | Canonical Monte Carlo runner for the FDPR residual heatmaps in §3.5 of the paper.  Writes `MC_RMS_heatmap_CORRECTED_dz.npz`. | **active** |
| `dean_bowers.py` | Two-line implementation of equation (9) of Dean & Bowers 2003. | **active** |
| `dean_bowers_graph.py` | Plots the Dean-Bowers sideband envelope against simulated OTFs (Figure 5 in the paper). | **active** |
| `april28_test.ipynb` | Single-operating-point diagnostic notebook: α/δ decomposition, iteration sweep, frequency-cutoff sweep, photon sweep, φ₀ knob tests, multi-seed reproducibility.  Markdown cells include interpretation guidelines for every output. | **active** |
| `april28_test.html` | HTML export of `april28_test.ipynb` for offline reading. | **active** (export) |

## Active sweep / analysis scripts

| File | Description | Status |
|---|---|---|
| `replot_paper_figures.py` | Regenerates every paper figure from production-resolution `.npz` files.  Drop-in replacement for the in-Overleaf figures.  Axis-bug fix applied 2026-05-19. | **active** |
| `resume_dense_fdpr.py` | Resumes a checkpointed `Dense_FDPR_Grid_Analysis` run from `fdpr_intermediate_row*.npz`.  Note: RNG seed differs from the original runner, which produces a seam at the resume row. | **active** |
| `optimize_diversity_config.py` | Searches `(N, {Δz_i})` for the diversity configuration that minimizes FDPR residual RMS at fixed total photon budget.  Run-ID-labelled outputs in `fdpr_optimization/optdz_<UTC>/`.  Signal-scale labelling clarified 2026-05-19. | **active** |
| `diagnose_dense_transition.py` | Four-panel diagnostic of the â≈12-wave simulator-limit transition in the dense FDPR run (Figure 17). | **active** |
| `compare_fdpr_otf_ridges.py` | Per-ν₀ FDPR-optimum vs OTF-optimum ridge comparison restricted to the reliable Δz≤145 mm regime (Figure 18). | **active** |
| `fdpr_injectable.py` | Drop-in FDPR wrapper that exposes the forward model as a callable; used by the `_mc` scripts below. | **active** |
| `fdpr_mc.py` | Single-amplitude Monte Carlo sweep over `(Δz, ν₀)` using `fdpr_injectable`. | **active** |
| `fdpr_mc_pm.py` | Paired-magnitude (`±Δz`) variant of `fdpr_mc.py`. | **active** |
| `fdpr_mc_pm_sinamp.py` | Paired-magnitude sweep over sinusoid amplitude rather than frequency. | **active** |
| `fdpr_mc_pm_zernike.py` | Paired-magnitude sweep over Zernike-coma amplitude. | **active** |
| `analyze_zernike_sweep.py` | Post-processes the output of `fdpr_mc_pm_zernike.py` into the figures in `paper_figures/`. | **active** |
| `dz_plotting.py` | Two-line plotting helper used during interactive exploration of `MC_RMS_heatmap_*.npz`. | **active** |
| `Paper_Figures_OTF.py` | OTF-only figure generator.  Subsumed by `replot_paper_figures.py` for the paper but still callable for ad-hoc plots. | **active** |
| `fdpr_vs_AS.py` | Comparison of FDPR against Gerchberg-Saxton "AS" (alternating projection). | **active** |
| `make_fdpr_pptx.py` | Generates an SPIE-style PowerPoint of the FDPR pipeline.  Orthogonal to the paper. | **active** |

## Notebook generation / edit scripts

| File | Description | Status |
|---|---|---|
| `_apply_notebook_edits.py` | Reproducible source-of-truth for the markdown-cell enrichment applied to `april28_test.ipynb`.  Idempotent. | **utility** |
| `_insert_output_guides.py` | Inserts the "### Reading the …" interpretation cells after each output-producing code cell in `april28_test.ipynb`.  Idempotent. | **utility** |

## Kept for context

| File | Description | Status |
|---|---|---|
| `Original_Focus_diversity.ipynb` | The earliest version of the focus-diversity exploration notebook.  Retained as a historical record of how the project started; not used by any current pipeline. | **historical** |
| `focus_diversity_futzing.ipynb` | Early focus-diversity sandbox.  Retained alongside `Original_Focus_diversity.ipynb` for context; not used by any current pipeline. | **historical** |

## Out-of-scope (unrelated projects committed in the same directory)

These belong to other projects and are left in place pending a separate decision.

| File | Description |
|---|---|
| `cred2_test/test.py`, `cred2_test/test_nl.py` | Keck KTL keyword test for the CRED2 detector; not the FDPR paper. |
| `lab2photometrystuff.py` | Coursework photometry script. |
| `variable_star_photometry.ipynb` | Coursework variable-star photometry notebook. |
| `.claude/skills/gitnexus/*.md` | Editor-side Claude skill files; not science code. |
| `Jul_9.jpg`, `row_array.ps`, `defocus_conversion.pdf` | Plot exports from older runs.  Consider gitignore. |

## Data / figure outputs (root)

Older `.png` / `.pdf` outputs predating `.gitignore` are still in the tree but the gitignore now catches new ones.  These are not authoritative; the paper uses figures regenerated via `replot_paper_figures.py`, `diagnose_dense_transition.py`, and `compare_fdpr_otf_ridges.py`.

---

## Cleanup history

**2026-05-28** — removed 70 deprecated files (duplicates, superseded snapshots, scratch notebooks, wrong-physics variants) from the working tree.  Earlier state still recoverable from git history; the canonical set listed above is enough to reproduce every figure in the paper.
