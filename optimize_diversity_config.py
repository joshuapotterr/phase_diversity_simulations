"""
Search for the FDPR diversity configuration that minimizes residual RMS.

Optimization variables:
    N              number of focal-plane inputs (incl. the focused frame)
    {Delta z_i}    magnitudes of the defocused inputs (mm)

Fixed quantities (so comparisons are fair across N):
    N_gamma_total  total photon budget shared across the N frames
    sigma_r        per-pixel read-noise RMS
    aberration     a fixed test phase (sinusoid by default, then coma)
    n_trials       FDPR Monte Carlo trials per configuration (different RNG seeds)
    n_iterations   FDPR iteration budget

For each (N, config) pair we run FDPR n_trials times and record per-trial
residual RMS plus mean and std across trials.

Run identification:
    Every invocation of this script generates a RUN_ID of the form
    "optdz_<UTC_timestamp>" that prefixes the output directory and is
    embedded in every npz / log line.  Two runs of this script with
    different parameters can therefore coexist without overwriting and
    can be cross-referenced by RUN_ID.

Output (under fdpr_optimization/<RUN_ID>/):
    config.json                static metadata about this run
    sweep_results.npz          all per-config rows, with full provenance
    best_per_N.npz             top-1 per N
    runlog.txt                 line-by-line log, prefixed with RUN_ID

The npz files store:
    run_id, started_at_utc, aberration, A_waves, nu0,
    N_gamma_total, sigma_r, n_trials, n_iter, dz_grid, results

Each row in `results` carries: N, config_label, dz_list, n_per_frame,
rms_mean_nm, rms_std_nm, conv_rate, trials_nm (length-n_trials list).
Designed to run in the background.
"""
from __future__ import annotations
import os
import sys
import time
import json
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from Dense_FDPR_Grid_Analysis import (
    DenseGridConfig,
    setup_optics,
)
from hcipy import (
    Wavefront, Field, make_zernike_basis,
)
from image_sharpening import FocusDiversePhaseRetrieval, mft_rev
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Run identification + output paths
# ---------------------------------------------------------------------------
RUN_ID = "optdz_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
ROOT = os.path.join(HERE, "fdpr_optimization", RUN_ID)
os.makedirs(ROOT, exist_ok=True)
LOG = open(os.path.join(ROOT, "runlog.txt"), "a", buffering=1)


def log(msg):
    line = f"[{RUN_ID}] [{time.strftime('%H:%M:%S')}]  {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


# Numerical constants
WAVELENGTH_M = 650e-9
D_M = 10.12e-3
F_M = 500e-3
PUPIL_NPIX = 256
RAD_TO_NM = WAVELENGTH_M * 1e9 / (2 * np.pi)
N_ITERATIONS = 150
N_TRIALS = 5
N_GAMMA_TOTAL = 3e6                # total photons split across N frames
SIGMA_R = 11.0

# Test aberration: sinusoid at the operating point used elsewhere
A_WAVES = 0.10
NU0 = 10.0

# Defocus magnitude grid (mm).  Sampled densely enough that the first
# Dean-Bowers ridge for v0=10 (a_hat=25 waves, dz~318 mm) is reachable.
DZ_GRID = np.array([10, 20, 30, 50, 70, 100, 140, 180, 220, 260], dtype=float)

# Per-N exploration of configurations.  Each "config" is a list of dz values
# in mm; the focused image (dz=0) is always included implicitly.
def configs_for_N(N: int):
    """Yield (label, [dz, ...]) tuples of length N-1 (focused is implicit)."""
    if N == 2:
        for dz in DZ_GRID:
            yield (f"asym_{dz:.0f}", [dz])
    elif N == 3:
        # Symmetric pair (paper baseline)
        for dz in DZ_GRID:
            yield (f"sym_pm{dz:.0f}", [+dz, -dz])
    elif N == 4:
        # Three defocused frames: one symmetric pair + one extra
        for dz1, dz2 in itertools.product(DZ_GRID[::2], DZ_GRID[::2]):
            if dz1 == dz2:
                continue
            yield (f"sym_pm{dz1:.0f}_plus{dz2:.0f}", [+dz1, -dz1, +dz2])
    elif N == 5:
        # Two symmetric pairs
        for i, dz1 in enumerate(DZ_GRID[::2]):
            for dz2 in DZ_GRID[::2][i + 1:]:  # upper-triangular: avoid duplicates
                yield (f"sym_pm{dz1:.0f}_pm{dz2:.0f}", [+dz1, -dz1, +dz2, -dz2])
    elif N == 7:
        # Three symmetric pairs, coarser sampling
        coarse = np.array([20, 70, 140, 220], dtype=float)
        for combo in itertools.combinations(coarse, 3):
            dz1, dz2, dz3 = combo
            yield (f"sym_pm{dz1:.0f}_pm{dz2:.0f}_pm{dz3:.0f}",
                   [+dz1, -dz1, +dz2, -dz2, +dz3, -dz3])


# ---------------------------------------------------------------------------
# FDPR with a user-supplied list of defocus values
# ---------------------------------------------------------------------------
def run_fdpr_general(dz_list_mm, n_per_frame, optics, rng, A_waves=A_WAVES, nu0=NU0,
                     n_iter=N_ITERATIONS, aberration="sinusoid"):
    """Return residual RMS (nm) for the given diversity list, or NaN if it fails."""
    try:
        pupil_grid = optics["pupil_grid"]
        telescope_pupil = optics["telescope_pupil"]
        pupil_mask = optics["pupil_mask"]
        prop = optics["propagator"]
        unit_defocus = optics["unit_defocus"]
        fdpr_conf = optics["fdpr_conf"]

        # Truth phase
        if aberration == "sinusoid":
            phi = Field(A_waves * 2 * np.pi *
                        np.sin(2 * np.pi * nu0 * pupil_grid.x / D_M),
                        pupil_grid)
        elif aberration == "coma":
            zerns = make_zernike_basis(num_modes=10, D=D_M, grid=pupil_grid)
            phi = A_waves * 2 * np.pi * zerns[7]      # Z8, x-coma
        else:
            raise ValueError(aberration)

        truth = np.asarray(phi.shaped) * np.asarray(telescope_pupil.shaped)

        def make_psf(phase_field):
            wf = Wavefront(telescope_pupil, wavelength=WAVELENGTH_M)
            wf.electric_field = telescope_pupil * np.exp(1j * phase_field)
            return np.asarray(prop(wf).intensity.shaped)

        def defocus_phase_mm(dz_mm):
            dz_m = dz_mm * 1e-3
            defocus_p2v_m = -dz_m / (8 * (F_M / D_M) ** 2)
            phase_p2v = defocus_p2v_m * (2 * np.pi / WAVELENGTH_M)
            return unit_defocus * phase_p2v

        psfs = [make_psf(phi)]
        for dz in dz_list_mm:
            psfs.append(make_psf(phi + defocus_phase_mm(dz)))

        # Divided-photon-budget noise model.  Raw HCIPy PSFs have relative-
        # intensity units; the working pipeline (Dense_FDPR_Grid_Analysis)
        # uses them directly with sigma_r ~ 5-11.  To make N comparable
        # across configs at fixed total exposure, we scale each frame's
        # signal by (n_per_frame / n_per_frame_baseline), where the
        # baseline is the per-frame budget that the production sweeps use
        # (N_GAMMA_TOTAL / 3).  Read noise sigma stays fixed; the SNR per
        # frame falls as 1/N because the signal drops, not because the
        # noise rises -- which is the physically correct way to model
        # adding a frame at fixed total exposure.
        n_per_frame_baseline = N_GAMMA_TOTAL / 3.0
        signal_scale = n_per_frame / n_per_frame_baseline
        noisy = []
        for p in psfs:
            scaled = p * signal_scale
            n = scaled + rng.normal(0.0, SIGMA_R, scaled.shape)
            noisy.append(np.maximum(n, 0.0))

        dz_um = [float(dz) * 1e3 for dz in dz_list_mm]
        # image_dx hardcoded to 2.0071 um inside setup_optics; the FDPR
        # interface wants one image_dx per defocused frame.
        image_dx_um = 2.0071
        mp = FocusDiversePhaseRetrieval(
            noisy, WAVELENGTH_M * 1e6,
            [image_dx_um] * len(dz_um), dz_um)
        for _ in range(n_iter):
            rec = mp.step()

        raw = np.angle(mft_rev(rec, fdpr_conf))
        recon = resize(raw, (PUPIL_NPIX, PUPIL_NPIX), preserve_range=True) * \
                np.asarray(telescope_pupil.shaped)

        truth_ms = truth[pupil_mask] - np.median(truth[pupil_mask])
        recon_ms = recon[pupil_mask] - np.median(recon[pupil_mask])
        residual = truth_ms - recon_ms
        if not np.any(np.isfinite(residual)):
            return np.nan
        return float(np.sqrt(np.nanmean(residual ** 2)) * RAD_TO_NM)
    except Exception as e:
        log(f"  FDPR FAIL: {e!r}")
        return np.nan


# ---------------------------------------------------------------------------
# Build the optics state once, reuse across all configs
# ---------------------------------------------------------------------------
def build_optics():
    base = DenseGridConfig(grid_density="dense", n_iterations=N_ITERATIONS,
                           n_trials=N_TRIALS, read_noise_e=SIGMA_R, m_waves=A_WAVES)
    return setup_optics(base)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Write a static config.json so every output file in this run can be
    # cross-referenced back to its operating point in one place.
    config = dict(
        run_id=RUN_ID,
        started_at_utc=started_at,
        script="optimize_diversity_config.py",
        objective="minimize FDPR residual RMS",
        optimization_variables=["N", "{Delta z_i in mm}"],
        aberration="sinusoid",
        A_waves=A_WAVES,
        nu0_cyc_per_aperture=NU0,
        N_gamma_total_relative=N_GAMMA_TOTAL,
        sigma_r_e=SIGMA_R,
        n_trials=N_TRIALS,
        n_iterations=N_ITERATIONS,
        dz_grid_mm=DZ_GRID.tolist(),
        N_sweep=[2, 3, 4, 5, 7],
        notes=("'N_gamma_total' is a RELATIVE SIGNAL SCALE, not a literal "
               "photon count.  The script scales raw HCIPy PSF intensity by "
               "n_per_frame/(N_gamma_total/3) and adds Gaussian read noise; "
               "this reproduces the divided-photon-budget SNR penalty across "
               "N at fixed total exposure but the absolute n_per_frame number "
               "is dimensionless (ratio to the N=3 baseline).  Relative "
               "rankings of (N, dz_list) configurations are correct.  See "
               "the code-audit memo for the explanation."),
    )
    with open(os.path.join(ROOT, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    log("=" * 70)
    log("DIVERSITY-CONFIG OPTIMIZATION SWEEP")
    log(f"RUN_ID = {RUN_ID}")
    log(f"output dir = {ROOT}")
    log(f"aberration = sinusoid  v0={NU0} cyc/ap  A={A_WAVES} waves")
    log(f"N_gamma_total={N_GAMMA_TOTAL:.0e}  sigma_r={SIGMA_R}  "
        f"n_trials={N_TRIALS}  n_iter={N_ITERATIONS}")
    log("=" * 70)

    optics = build_optics()
    # One RNG seed for the whole run; recorded so the sweep is reproducible.
    rng_seed = 20260512
    rng = np.random.default_rng(rng_seed)
    config["rng_seed"] = rng_seed
    with open(os.path.join(ROOT, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    results = []
    t0 = time.time()
    for N in [2, 3, 4, 5, 7]:
        cfgs = list(configs_for_N(N))
        log(f"\n--- N={N} : {len(cfgs)} configurations ---")
        n_per_frame = N_GAMMA_TOTAL / N
        for label, dz_list in cfgs:
            cfg_id = f"{RUN_ID}__N{N}__{label}"
            trial_rms = []
            for t in range(N_TRIALS):
                rms = run_fdpr_general(dz_list, n_per_frame, optics, rng,
                                       aberration="sinusoid")
                trial_rms.append(rms)
            arr = np.asarray(trial_rms, dtype=float)
            mean = float(np.nanmean(arr))
            std = float(np.nanstd(arr))
            conv = float(np.isfinite(arr).sum() / N_TRIALS)
            results.append(dict(
                run_id=RUN_ID,
                config_id=cfg_id,
                N=N,
                label=label,
                dz_list=list(dz_list),
                n_per_frame=n_per_frame,
                rms_mean_nm=mean,
                rms_std_nm=std,
                conv_rate=conv,
                trials_nm=arr.tolist(),
                aberration="sinusoid",
                A_waves=A_WAVES,
                nu0=NU0,
                sigma_r=SIGMA_R,
                N_gamma_total=N_GAMMA_TOTAL,
                n_iterations=N_ITERATIONS,
            ))
            elapsed = (time.time() - t0) / 60
            log(f"  N={N}  {label:32s}  RMS = {mean:6.1f} +/- {std:5.1f} nm   "
                f"conv={conv:.2f}   sig_scale={n_per_frame/(N_GAMMA_TOTAL/3.0):.3f}   "
                f"elapsed={elapsed:.1f} min   id={cfg_id}")
            # Checkpoint after every config so a crash never loses more than
            # one configuration's worth of work.
            np.savez(os.path.join(ROOT, "sweep_results.npz"),
                     run_id=RUN_ID,
                     started_at_utc=started_at,
                     config_json=json.dumps(config),
                     results_json=json.dumps(results))

    by_N = {}
    for r in results:
        if r["conv_rate"] < 0.6:
            continue
        if r["N"] not in by_N or r["rms_mean_nm"] < by_N[r["N"]]["rms_mean_nm"]:
            by_N[r["N"]] = r

    log("\n=== best per N (run_id=%s) ===" % RUN_ID)
    for N in sorted(by_N):
        r = by_N[N]
        sig_scale = r['n_per_frame'] / (N_GAMMA_TOTAL / 3.0)
        log(f"  N={N}: {r['label']:30s}  dz={r['dz_list']}  "
            f"RMS={r['rms_mean_nm']:.1f}+/-{r['rms_std_nm']:.1f} nm   "
            f"signal scale per frame = {sig_scale:.3f} (rel. to N=3 baseline)   "
            f"id={r['config_id']}")

    np.savez(os.path.join(ROOT, "best_per_N.npz"),
             run_id=RUN_ID,
             started_at_utc=started_at,
             config_json=json.dumps(config),
             best_json=json.dumps(list(by_N.values())))

    log("\nDone.  Outputs in %s" % ROOT)


if __name__ == "__main__":
    main()
