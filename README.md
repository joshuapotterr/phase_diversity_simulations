# Phase Diversity Simulations

Simulation and analysis code for the focus-diverse phase retrieval (FDPR) work on the Santa Cruz Extreme Adaptive Optics Laboratory (SEAL) testbed. Companion code to the PASP submission *Optimization of Sequential Phase Diversity: Defocus-Magnitude Selection for Focal-Plane Wavefront Sensing on the SEAL Testbed*.

## Repository layout

- **Forward model + retrieval.** `Dense_FDPR_Grid_Analysis.py`, `FDPR_Monte_Carlo_Final.py`, `MC_FINAL.py`, `Jul21FDPR.py`. The Misell-style amplitude-projection FDPR loop, wired up to the HCIPy Fraunhofer propagator with the SEAL optical parameters.
- **OTF analysis.** `OTF_FINAL.py`, `OTF_Sinusoidal_Paper.py`, `dean_bowers_graph_highres_otf.py`, `Paper_Figures_OTF.py`. Code for the sideband-amplitude heatmaps over the $(\nu_0, \Delta z)$ parameter space.
- **Optimization sweeps.** `optimize_diversity_config.py`. Searches $(N, \{\Delta z_i\})$ for the diversity configuration that minimizes FDPR residual RMS at fixed total photon budget.
- **Background resume.** `resume_dense_fdpr.py`. Picks up a checkpointed dense FDPR sweep from `fdpr_intermediate_row50.npz` and runs the remaining rows.
- **Paper figure generation.** `replot_paper_figures.py`. Loads the production-resolution `.npz` datasets and emits the figures that appear in the manuscript.
- **Diagnostic notebook.** `april28_test.ipynb`. The single-operating-point FDPR diagnostic that produced the $\alpha = 0.86$ pipeline-ceiling result and the $\nu_0 \approx 15$ cyc/aperture cutoff. Markdown cells include interpretation guidelines for every output.

## Run identification

All long-running sweeps produce timestamped output directories (e.g.\ `fdpr_optimization/optdz_20260513T223008Z/`) with a `config.json` capturing the operating point, a `runlog.txt`, and `.npz` outputs that embed the `run_id` for cross-referencing.

## Data products

Large `.npz` data files (sweep outputs, OTF heatmaps, residual maps) are not version-controlled in this repository; see the corresponding paper for citation. A Zenodo deposit is planned at paper acceptance.

## Dependencies

- `numpy`, `scipy`, `matplotlib`, `scikit-image`
- [`hcipy`](https://docs.hcipy.org) for the Fraunhofer propagator and Zernike basis
- An internal `image_sharpening` package providing `FocusDiversePhaseRetrieval`, `mft_rev`, and `InstrumentConfiguration`

## License

MIT (see `LICENSE`).
