"""
Regenerate paper figures from the production-resolution .npz datasets.

Reads:
  - OTF_heatmap_data_new_block.npz    (80 x 150 OTF, focus + defocus)
  - OTF_heatmap_data_fixed.npz        (80 x 150 OTF with H_detection + H_photometry)
  - MC_RMS_heatmap_CORRECTED_dz.npz   (25 x 40 x 5 trials FDPR sweep)

Writes paper-ready PNGs into paper_figures_real/.  Each output filename matches
the \\includegraphics target in .context/overleaf/aastex701-1/sample701.tex so
files can be copied straight in.

SEAL geometry: lambda = 650 nm, D = 10.12 mm, f = 500 mm (F/49.4).
Normalized defocus  a_hat = D^2 * dz / (8 * lambda * f^2)   [waves P-V].
Dean-Bowers maximum-diversity defocus locus:  a_hat_rev_max = nu_0^2 / 4.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# SEAL geometry  (Jensen+ 2025 / Section 2.2 of the paper)
# ---------------------------------------------------------------------------
LAMBDA_M = 650e-9
D_M = 10.12e-3
F_M = 500e-3


def ahat_from_dz_mm(dz_mm: np.ndarray) -> np.ndarray:
    """Dean-Bowers normalized defocus a_hat (waves P-V) for a given dz [mm]."""
    return (D_M ** 2) * (dz_mm * 1e-3) / (8.0 * LAMBDA_M * F_M ** 2)


def dz_mm_from_ahat(ahat: np.ndarray) -> np.ndarray:
    return ahat * (8.0 * LAMBDA_M * F_M ** 2) / (D_M ** 2) / 1e-3


def nu0_dean_bowers_envelope_dz(nu0: np.ndarray) -> np.ndarray:
    """dz (mm) along the a_hat = nu0^2 / 4 Dean-Bowers max-diversity ridge."""
    ahat = (nu0 ** 2) / 4.0
    return dz_mm_from_ahat(ahat)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper_figures_real")
os.makedirs(OUT, exist_ok=True)


def _save(fig, name: str) -> str:
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Common loader: production OTF sideband heatmap (aperture-photometry metric,
# masked by H_valid_photometry).  OTF_heatmap_data_new_block.npz stores
# unbackground-subtracted OTF magnitude (includes DC) and is unsuitable here;
# the _fixed file applies the sideband window + DC mask.
# ---------------------------------------------------------------------------
def _load_otf_photometry():
    d = np.load(os.path.join(HERE, "OTF_heatmap_data_fixed.npz"))
    dzs = d["fixed_dz_heatmap"]
    v0s = d["v0_heatmap"]
    H = d["H_photometry"].copy()
    mask = d["H_valid_photometry"].astype(bool)
    H[~mask] = np.nan
    return dzs, v0s, H


# ---------------------------------------------------------------------------
# Figure: OTF heatmap with dual axis (replaces fig6_heatmap_dual_axis.png)
# ---------------------------------------------------------------------------
def fig_heatmap_dual_axis():
    dzs, v0s, H = _load_otf_photometry()
    Hn = H / np.nanmax(H)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    pcm = ax.pcolormesh(v0s, dzs, Hn, shading="auto", cmap="magma", vmin=0, vmax=1)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.13, fraction=0.05)
    cbar.set_label("OTF sideband amplitude (normalized)")

    # Dean-Bowers max-diversity ridge
    ridge = nu0_dean_bowers_envelope_dz(v0s)
    mask = ridge <= dzs.max()
    ax.plot(v0s[mask], ridge[mask], "r--", lw=1.2,
            label=r"$\hat a_{\rm rev,max}=\nu_0^2/4$")
    ax.set_xlabel(r"Spatial frequency $\nu_0$ (cycles/aperture)")
    ax.set_ylabel(r"Defocus $\Delta z$ (mm)")
    ax.set_xlim(v0s.min(), v0s.max())
    ax.set_ylim(dzs.min(), dzs.max())
    ax.legend(loc="upper right", framealpha=0.9)

    # right axis: a_hat in waves P-V
    ax2 = ax.twinx()
    ax2.set_ylim(ahat_from_dz_mm(dzs.min()), ahat_from_dz_mm(dzs.max()))
    ax2.set_ylabel(r"Normalized defocus $\hat a$ (waves P-V)")

    return _save(fig, "fig6_heatmap_dual_axis.png")


# ---------------------------------------------------------------------------
# Figure: OTF slices at fixed dz  (replaces fig2_slices_fixed_dz.png)
# Figure: OTF slices at fixed nu0 (replaces fig3_slices_fixed_v0.png)
# ---------------------------------------------------------------------------
def fig_slices():
    dzs, v0s, H = _load_otf_photometry()
    Hn = H / np.nanmax(H)

    # Light 3-point boxcar smoothing along v0 to suppress per-pixel jitter
    # without erasing the Dean-Bowers quasi-periodic structure.
    def smooth(y, k=3):
        kernel = np.ones(k) / k
        y2 = np.where(np.isfinite(y), y, np.nan)
        # nan-aware moving average
        mask = np.isfinite(y2)
        y2 = np.where(mask, y2, 0)
        return np.convolve(y2, kernel, mode="same") / np.convolve(mask.astype(float), kernel, mode="same")

    # Slices at fixed dz
    target_dz = [24, 52, 101, 151, 200]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for tdz in target_dz:
        i = int(np.argmin(np.abs(dzs - tdz)))
        ax.plot(v0s, smooth(Hn[i]), lw=1.6, label=fr"$\Delta z = {dzs[i]:.0f}$\,mm")
    ax.set_xlabel(r"Spatial frequency $\nu_0$ (cycles/aperture)")
    ax.set_ylabel("OTF sideband amplitude (normalized)")
    ax.set_xlim(v0s.min(), v0s.max())
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    path_dz = _save(fig, "fig2_slices_fixed_dz.png")

    # Slices at fixed nu0
    target_v0 = [5, 10, 20, 40, 60]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for tv in target_v0:
        j = int(np.argmin(np.abs(v0s - tv)))
        ax.plot(dzs, smooth(Hn[:, j]), lw=1.6, label=fr"$\nu_0 = {v0s[j]:.1f}$\,cyc/ap")
    ax.set_xlabel(r"Defocus $\Delta z$ (mm)")
    ax.set_ylabel("OTF sideband amplitude (normalized)")
    ax.set_xlim(dzs.min(), dzs.max())
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    path_v0 = _save(fig, "fig3_slices_fixed_v0.png")

    return path_dz, path_v0


# ---------------------------------------------------------------------------
# Figure: optimal dz vs nu0 (replaces fig4_optimal_defocus.png)
# ---------------------------------------------------------------------------
def fig_optimal_defocus():
    dzs, v0s, H = _load_otf_photometry()
    # nanargmax fails on all-NaN columns; mask those out before lookup
    col_has_valid = np.any(np.isfinite(H), axis=0)
    dz_star = np.full(H.shape[1], np.nan)
    if col_has_valid.any():
        idx = np.nanargmax(H[:, col_has_valid], axis=0)
        dz_star[col_has_valid] = dzs[idx]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw=dict(width_ratios=[1.4, 1.0]))

    finite = np.isfinite(dz_star)
    axL.scatter(v0s[finite], dz_star[finite], s=14, color="C0", label="measured")
    v0_ridge = np.linspace(v0s.min(), v0s.max(), 400)
    ridge_dz = nu0_dean_bowers_envelope_dz(v0_ridge)
    axL.plot(v0_ridge, ridge_dz, "r-", lw=1.2,
             label=r"$\hat a = \nu_0^2/4$  (Dean & Bowers 2003)")
    axL.axhline(dzs.max(), color="0.5", ls=":", lw=1)
    axL.set_xlabel(r"Spatial frequency $\nu_0$ (cycles/aperture)")
    axL.set_ylabel(r"Optimal defocus $\Delta z^{*}$ (mm)")
    axL.set_ylim(0, dzs.max() * 1.05)
    axL.set_xlim(v0s.min(), v0s.max())
    axL.legend(loc="lower right", fontsize=9)
    axL.set_title("(a)")

    axR.hist(dz_star[finite], bins=20, color="C0", edgecolor="white")
    axR.axvline(dzs.max(), color="0.5", ls=":", lw=1, label="grid limit")
    axR.set_xlabel(r"Optimal defocus $\Delta z^{*}$ (mm)")
    axR.set_ylabel("Count")
    axR.set_title("(b)")
    axR.legend(loc="upper left", fontsize=9)

    return _save(fig, "fig4_optimal_defocus.png")


# ---------------------------------------------------------------------------
# Figure: peak vs aperture-photometry comparison
# (replaces fig7_method_comparison.png  -- now from the same frames!)
# ---------------------------------------------------------------------------
def fig_method_comparison():
    d = np.load(os.path.join(HERE, "OTF_heatmap_data_fixed.npz"))
    Hp = d["H_detection"]    # peak detection
    Ha = d["H_photometry"]   # aperture photometry
    # AXIS BUG FIX: use the actual axis arrays from the file rather than
    # assuming np.linspace(0.5, 80) (which is off by ~2.5 cyc/ap at low v0).
    dzs = d["fixed_dz_heatmap"]
    v0s = d["v0_heatmap"]

    # use valid-pixel masks if provided
    mp = d["H_valid_detection"].astype(bool) if "H_valid_detection" in d.files else np.ones_like(Hp, bool)
    ma = d["H_valid_photometry"].astype(bool) if "H_valid_photometry" in d.files else np.ones_like(Ha, bool)

    Hp_n = Hp / np.nanmax(Hp)
    Ha_n = Ha / np.nanmax(Ha)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    plt.subplots_adjust(wspace=0.45)
    for ax, M, title in zip(axs[:2], [Hp_n, Ha_n], ["(a) peak detection", "(b) aperture photometry"]):
        pcm = ax.pcolormesh(v0s, dzs, M, shading="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xlabel(r"$\nu_0$ (cyc/ap)")
        ax.set_title(title)
        cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("amplitude (norm.)", fontsize=9)
    axs[0].set_ylabel(r"$\Delta z$ (mm)")

    valid = mp & ma & np.isfinite(Hp) & np.isfinite(Ha)
    x = Hp_n[valid]
    y = Ha_n[valid]
    r_p, _ = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    ax = axs[2]
    ax.scatter(x, y, s=6, alpha=0.4, color="C0")
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="1:1")
    ax.set_xlabel("peak detection (normalized)")
    ax.set_ylabel("aperture photometry (normalized)")
    ax.set_title(rf"(c) Pearson $r$={r_p:+.3f},  Spearman $r$={r_s:+.3f}, $p$={p_s:.1e}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

    print(f"  method comparison: Pearson r={r_p:+.4f}  Spearman r={r_s:+.4f}  p={p_s:.3e}  "
          f"on n={valid.sum()} valid cells")

    return _save(fig, "fig7_method_comparison.png")


# ---------------------------------------------------------------------------
# Figure: FDPR Monte Carlo 25 x 40 sweep
# (replaces sinusoid_amp0p2_heatmaps.png with production data)
# ---------------------------------------------------------------------------
def fig_mc_sin_production():
    d = np.load(os.path.join(HERE, "MC_RMS_heatmap_CORRECTED_dz.npz"))
    dzs = d["dzs_mc"]
    v0s = d["v0s_mc"]
    mean = d["rms_mean_nm"].copy()
    std = d["rms_std_nm"].copy()
    conv = d["convergence_rate"]
    sigma_e = float(d["sigma_e"])
    m_waves = float(d["m_waves"])
    n_trials = int(d["N_trials"])

    # Two rows show row-systematic elevated RMS not present in their immediate
    # neighbours: i=14 (dz=148 mm, a_hat=11.65 waves) and i=24 (dz=250 mm,
    # the propagation-grid edge).  Trial-to-trial std at these rows is 3-4x
    # the typical row std, suggesting a propagation/phase-wrap edge case
    # rather than a noise realization.  Flag them visually rather than
    # silently averaging them away.
    bad = [14, 24]

    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.4))
    plt.subplots_adjust(wspace=0.35)

    # (a) mean RMS, with bad rows visibly outlined
    pcm = axs[0].pcolormesh(v0s, dzs, mean, shading="auto", cmap="viridis")
    cb = fig.colorbar(pcm, ax=axs[0], fraction=0.046, pad=0.04)
    cb.set_label("mean residual RMS (nm)")
    axs[0].set_title(f"(a) mean RMS  ({n_trials} trials/cell)")
    axs[0].set_xlabel(r"$\nu_0$ (cyc/ap)")
    axs[0].set_ylabel(r"$\Delta z$ (mm)")
    for i in bad:
        axs[0].axhspan(dzs[i] - (dzs[1] - dzs[0]) / 2,
                       dzs[i] + (dzs[1] - dzs[0]) / 2,
                       fill=False, edgecolor="red", lw=1.0, ls="--")

    pcm = axs[1].pcolormesh(v0s, dzs, std, shading="auto", cmap="viridis")
    cb = fig.colorbar(pcm, ax=axs[1], fraction=0.046, pad=0.04)
    cb.set_label("RMS std (nm)")
    axs[1].set_title("(b) trial-to-trial std")
    axs[1].set_xlabel(r"$\nu_0$ (cyc/ap)")

    pcm = axs[2].pcolormesh(v0s, dzs, conv, shading="auto", cmap="cividis", vmin=0, vmax=1)
    cb = fig.colorbar(pcm, ax=axs[2], fraction=0.046, pad=0.04)
    cb.set_label("convergence rate")
    axs[2].set_title("(c) convergence rate")
    axs[2].set_xlabel(r"$\nu_0$ (cyc/ap)")

    # Dean-Bowers ridge overlay on (a) only -- keeps (b)(c) uncluttered.
    ridge = nu0_dean_bowers_envelope_dz(v0s)
    rmask = ridge <= dzs.max()
    axs[0].plot(v0s[rmask], ridge[rmask], "r--", lw=1.0, alpha=0.9,
                label=r"$\hat a=\nu_0^2/4$")
    axs[0].legend(loc="upper left", fontsize=8, framealpha=0.85)

    fig.suptitle(rf"Sinusoid sweep, $A = {m_waves}$ waves, $\sigma_r = {sigma_e:.0f}\,e^-$,  "
                 rf"grid {mean.shape[0]}$\times${mean.shape[1]} (5 trials/cell). "
                 r"Red-dashed rows are flagged anomalies; see caption.",
                 y=1.02, fontsize=10)
    return _save(fig, "sinusoid_amp0p1_heatmaps_production.png")


# ---------------------------------------------------------------------------
# Joint OTF <-> FDPR scatter at matched resolution
# (replaces otf_vs_rms_regen.png with the 25x40 FDPR cross-referenced against
# the 80x150 OTF, downsampled to the FDPR grid)
# ---------------------------------------------------------------------------
def fig_otf_vs_rms():
    dzs_otf, v0s_otf, H = _load_otf_photometry()
    H = H / np.nanmax(H)

    # Canonical FDPR data for the paper: 25x40 production sweep at sigma_r=11,
    # 5 trials/cell.  Match fig_mc_sin_production so both FDPR figures share
    # one source.  The 80x150 dense run (sigma_r=5, 1 trial) is plotted
    # separately by fig_fdpr_dense (below) because it is at a different
    # noise operating point.
    mc = np.load(os.path.join(HERE, "MC_RMS_heatmap_CORRECTED_dz.npz"))
    dzs_mc = mc["dzs_mc"]
    v0s_mc = mc["v0s_mc"]
    rms = mc["rms_mean_nm"]
    n_trials_mc = int(mc["N_trials"])
    sigma_r_mc = float(mc["sigma_e"])
    fdpr_label = f"{rms.shape[0]}x{rms.shape[1]}, {n_trials_mc} trials/cell, $\\sigma_r={sigma_r_mc:.0f}\\,e^-$"

    # Sample the OTF grid at the FDPR grid points (no-op when they share axes)
    H_at_mc = np.empty_like(rms)
    for i, dz in enumerate(dzs_mc):
        for j, v in enumerate(v0s_mc):
            ii = int(np.argmin(np.abs(dzs_otf - dz)))
            jj = int(np.argmin(np.abs(v0s_otf - v)))
            H_at_mc[i, j] = H[ii, jj]

    x_flat = H_at_mc.flatten()
    y_flat = rms.flatten()
    dz_mesh = np.repeat(dzs_mc, rms.shape[1])
    v0_mesh = np.tile(v0s_mc, rms.shape[0])
    finite_all = np.isfinite(x_flat) & np.isfinite(y_flat) & (x_flat > 1e-4)

    # The Dean-Bowers sideband expansion only describes FDPR behaviour
    # outside the low-v0 anomaly where a sinusoid looks like a low-order
    # Zernike that FDPR handles independently of OTF brightness.
    # Restrict to the regime in which the prediction is physically meaningful.
    regime = finite_all & (v0_mesh >= 5.0) & (dz_mesh <= 100.0)
    r_s, p_s = spearmanr(x_flat[regime], y_flat[regime])
    r_p_log, _ = pearsonr(np.log10(x_flat[regime]), np.log10(y_flat[regime]))
    r_g, p_g = spearmanr(x_flat[finite_all], y_flat[finite_all])
    grey = finite_all & ~regime
    r_grey, p_grey = spearmanr(x_flat[grey], y_flat[grey])

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6),
                             gridspec_kw=dict(width_ratios=[1.1, 1.0, 1.2]))
    plt.subplots_adjust(wspace=0.42, left=0.06, right=0.99, top=0.90, bottom=0.13)

    pcm = axs[0].pcolormesh(v0s_otf, dzs_otf, H, shading="auto", cmap="magma", vmin=0, vmax=1)
    cb = fig.colorbar(pcm, ax=axs[0], fraction=0.046, pad=0.04)
    cb.set_label("OTF sideband (norm.)", fontsize=9)
    axs[0].set_title("(a) OTF sideband (80$\\times$150)", fontsize=11)
    axs[0].set_xlabel(r"$\nu_0$ (cyc/ap)")
    axs[0].set_ylabel(r"$\Delta z$ (mm)")
    axs[0].axhline(100, color="cyan", lw=1.2, ls="--")
    axs[0].axvline(5, color="cyan", lw=1.2, ls="--")

    pcm = axs[1].pcolormesh(v0s_mc, dzs_mc, rms, shading="auto", cmap="viridis")
    cb = fig.colorbar(pcm, ax=axs[1], fraction=0.046, pad=0.04)
    cb.set_label("residual RMS (nm)", fontsize=9)
    axs[1].set_title(f"(b) FDPR residual ({fdpr_label})", fontsize=11)
    axs[1].set_xlabel(r"$\nu_0$ (cyc/ap)")
    axs[1].set_ylabel(r"$\Delta z$ (mm)")
    axs[1].axhline(100, color="cyan", lw=1.2, ls="--")
    axs[1].axvline(5, color="cyan", lw=1.2, ls="--")

    axs[2].loglog(x_flat[grey], y_flat[grey],
                  "o", ms=4, alpha=0.30, color="0.55",
                  label=rf"outside regime: $r_s={r_grey:+.2f}$, $p={p_grey:.1e}$")
    axs[2].loglog(x_flat[regime], y_flat[regime], "o", ms=4, alpha=0.70, color="C3",
                  label=rf"$\nu_0\geq 5$, $\Delta z\leq 100$ mm: $r_s={r_s:+.2f}$, $p={p_s:.1e}$")
    axs[2].set_xlabel("OTF sideband (normalized)")
    axs[2].set_ylabel("FDPR residual RMS (nm)")
    axs[2].set_title("(c) regime-conditioned correlation", fontsize=11)
    axs[2].grid(True, which="both", alpha=0.25, lw=0.4)
    axs[2].legend(fontsize=9, loc="lower left", framealpha=0.92)

    print(f"  joint OTF/FDPR (global):   n={finite_all.sum():4d}  Spearman r={r_g:+.4f}  p={p_g:.3e}")
    print(f"  joint OTF/FDPR (regime):   n={regime.sum():4d}  Spearman r={r_s:+.4f}  p={p_s:.3e}  "
          f"log-log Pearson r={r_p_log:+.4f}")
    print(f"  joint OTF/FDPR (outside):  n={grey.sum():4d}  Spearman r={r_grey:+.4f}  p={p_grey:.3e}")

    return _save(fig, "otf_vs_rms_regen.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Regenerating paper figures from production .npz datasets...")
    fig_heatmap_dual_axis()
    fig_slices()
    fig_optimal_defocus()
    fig_method_comparison()
    fig_mc_sin_production()
    fig_otf_vs_rms()
    print("\nDone.  Output PNGs in:", OUT)
