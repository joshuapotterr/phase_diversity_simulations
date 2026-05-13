"""
Resume the partial 80x150 dense FDPR grid.  Loads
fdpr_dense_results/fdpr_intermediate_row50.npz, runs rows 50..79
(0-indexed; rows 51..80 in the script's 1-indexed reporting), and writes
the completed grid to fdpr_dense_results/fdpr_dense_completed.npz.

Config matches the original run that produced row50:
    grid_density='dense' (80 x 150)
    n_iterations=150
    n_trials=1
    read_noise_e=5.0
    m_waves=0.1

Estimated wall-clock at ~3 s/point: 30 * 150 = 4500 points -> ~3.75 hours.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from Dense_FDPR_Grid_Analysis import (
    DenseGridConfig,
    setup_optics,
    run_single_fdpr,
)

CFG = DenseGridConfig(
    grid_density="dense",
    n_iterations=150,
    n_trials=1,
    read_noise_e=5.0,
    m_waves=0.1,
)

CHECKPOINT_IN = os.path.join(HERE, "fdpr_dense_results", "fdpr_intermediate_row50.npz")
OUT_DIR = os.path.join(HERE, "fdpr_dense_results")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    if not os.path.exists(CHECKPOINT_IN):
        sys.exit(f"checkpoint missing: {CHECKPOINT_IN}")
    chk = np.load(CHECKPOINT_IN)
    dz_values = chk["dz_values"]
    v0_values = chk["v0_values"]
    rms_results = chk["rms_results"].copy()
    completed_rows = int(chk["completed_rows"])

    n_dz, n_v0, n_trials = rms_results.shape
    assert n_dz == 80 and n_v0 == 150 and n_trials == 1, rms_results.shape

    print(f"Resuming from row {completed_rows} (0-indexed: {completed_rows - 1}).")
    print(f"Grid: {n_dz} x {n_v0} x {n_trials}; rows {completed_rows}..{n_dz - 1} remain.")

    optics = setup_optics(CFG)
    rng = np.random.default_rng(54321)  # fresh seed for the resume half

    t_start = time.time()
    for i in range(completed_rows, n_dz):
        row_start = time.time()
        dz = float(dz_values[i])
        for j, v0 in enumerate(v0_values):
            for t in range(n_trials):
                rms_results[i, j, t] = run_single_fdpr(dz, float(v0), CFG, optics, rng)
        row_t = time.time() - row_start
        elapsed = time.time() - t_start
        done = i - completed_rows + 1
        total = n_dz - completed_rows
        eta_min = (elapsed / done) * (total - done) / 60.0
        valid = int(np.isfinite(rms_results[i]).sum())
        print(f"row {i + 1}/{n_dz}  dz={dz:6.1f} mm  valid={valid}/{n_v0}  "
              f"{row_t:6.1f}s  ETA {eta_min:6.1f} min", flush=True)

        if (i + 1) % 5 == 0:
            np.savez(os.path.join(OUT_DIR, f"fdpr_resume_row{i + 1}.npz"),
                     dz_values=dz_values, v0_values=v0_values,
                     rms_results=rms_results, completed_rows=i + 1)

    rms_mean = np.nanmean(rms_results, axis=2)
    rms_std = np.nanstd(rms_results, axis=2)
    out = os.path.join(OUT_DIR, "fdpr_dense_completed.npz")
    np.savez(out, dz=dz_values, v0=v0_values,
             rms_all=rms_results, rms_mean=rms_mean, rms_std=rms_std,
             config={"grid_density": CFG.grid_density,
                     "n_iterations": CFG.n_iterations,
                     "n_trials": CFG.n_trials,
                     "read_noise_e": CFG.read_noise_e,
                     "m_waves": CFG.m_waves})
    print(f"\nDone.  Wrote {out}")
    print(f"  rms_mean range: {np.nanmin(rms_mean):.1f} - {np.nanmax(rms_mean):.1f} nm")


if __name__ == "__main__":
    main()
