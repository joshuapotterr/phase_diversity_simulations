"""
Diagnose the sharp RMS transition at Δz ≈ 145-160 mm in the dense
80x150 FDPR sweep (fdpr_dense_completed.npz, σ_r=5, A=0.1 waves).

Produces paper_figures_real/dense_transition_diagnostic.png with:
    (a) the heatmap with the transition Δz outlined
    (b) row-mean RMS vs Δz, marking the jump
    (c) a 1-D RMS vs Δz slice at a representative ν₀
    (d) â (normalized defocus) vs Δz with the
        Nyquist-spillover and propagation-grid limits annotated

Numerical claim: the transition coincides with â ≈ 12 waves P-V.  At
that defocus the geometric blur radius (Δz / 2F#) is ≈ 0.79 of the
focal-plane half-extent, so the blur is still INSIDE the grid (the
spillover hypothesis is rejected).  The mechanism is therefore not
PSF cropping but more likely either (a) MFT focal-grid undersampling
of inner Fresnel rings at high â, or (b) a phase-wrap edge case
inside image_sharpening.mft_rev.  Definitive diagnosis requires
inspection of the package's MFT-reverse internals and is out of scope
here; the panel (d) plot reports the â value and the blur ratio as
the two relevant numbers without claiming a mechanism.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper_figures_real")
os.makedirs(OUT, exist_ok=True)

# SEAL geometry (matches replot_paper_figures.py and Dense_FDPR_Grid_Analysis.py)
LAMBDA_M = 650e-9
D_M = 10.12e-3
F_M = 500e-3
F_NUM = F_M / D_M
NUM_AIRY = 64       # from Dense_FDPR_Grid_Analysis.setup_optics (q=4, num_airy=64)
LAMBDA_F_OVER_D_UM = LAMBDA_M * F_M / D_M * 1e6   # one airy unit at focal plane (um)
HALF_GRID_UM = NUM_AIRY * LAMBDA_F_OVER_D_UM      # focal-plane half-width (um)


def ahat_from_dz_mm(dz_mm: np.ndarray) -> np.ndarray:
    return (D_M ** 2) * (dz_mm * 1e-3) / (8.0 * LAMBDA_M * F_M ** 2)


def geom_blur_radius_um(dz_mm: np.ndarray) -> np.ndarray:
    """Geometric defocus-blur radius at the focal plane, in micrometers."""
    return (dz_mm * 1e-3) / (2.0 * F_NUM) * 1e6


def main():
    d = np.load(os.path.join(HERE, "fdpr_dense_results", "fdpr_dense_completed.npz"))
    dz = d["dz"]
    v0 = d["v0"]
    rms = d["rms_mean"]
    print(f"Loaded {rms.shape} grid, RMS range {np.nanmin(rms):.1f}–{np.nanmax(rms):.1f} nm")

    row_mean = rms.mean(axis=1)
    row_p10 = np.nanpercentile(rms, 10, axis=1)
    row_p90 = np.nanpercentile(rms, 90, axis=1)

    # Find the largest single-step jump in row-mean RMS
    diffs = np.diff(row_mean)
    i_jump = int(np.argmax(diffs))
    dz_transition_lo = dz[i_jump]
    dz_transition_hi = dz[i_jump + 1]
    print(f"Largest row-mean RMS jump: dz {dz_transition_lo:.1f} -> {dz_transition_hi:.1f} mm"
          f"  (mean RMS {row_mean[i_jump]:.1f} -> {row_mean[i_jump+1]:.1f} nm)")

    ahat_lo = ahat_from_dz_mm(dz_transition_lo)
    ahat_hi = ahat_from_dz_mm(dz_transition_hi)
    print(f"Corresponds to â = {ahat_lo:.2f} -> {ahat_hi:.2f} waves P-V")

    # Sampling check at the transition
    blur_at_transition = geom_blur_radius_um(dz_transition_hi)
    print(f"Geometric blur radius at transition: {blur_at_transition:.1f} um  "
          f"(focal-plane half-width: {HALF_GRID_UM:.1f} um)")
    print(f"  Blur / half-grid ratio: {blur_at_transition / HALF_GRID_UM:.2f}")

    truth_rms_nm = (0.1 / np.sqrt(2)) * (LAMBDA_M * 1e9)  # A=0.1 wave sinusoid -> RMS in nm
    print(f"Truth sinusoid RMS (A=0.1 waves): {truth_rms_nm:.1f} nm")

    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1.0], hspace=0.35, wspace=0.32,
                          left=0.07, right=0.97, top=0.92, bottom=0.08)

    # ----- (a) full heatmap with transition band -----
    ax_a = fig.add_subplot(gs[0, :2])
    pcm = ax_a.pcolormesh(v0, dz, rms, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax_a, fraction=0.035, pad=0.02, label="mean residual RMS (nm)")
    # transition band
    ax_a.axhspan(dz_transition_lo, dz_transition_hi, fill=False,
                 edgecolor="red", lw=1.5, ls="--",
                 label=fr"transition: $\Delta z = {dz_transition_lo:.0f}\to{dz_transition_hi:.0f}$ mm  "
                       fr"($\hat a={ahat_lo:.1f}\to{ahat_hi:.1f}$ waves)")
    # Dean–Bowers ridge
    ridge = (v0 ** 2 / 4) * 8 * LAMBDA_M * F_M ** 2 / D_M ** 2 * 1e3
    m = ridge <= dz.max()
    ax_a.plot(v0[m], ridge[m], "w--", lw=1.0, alpha=0.7,
              label=r"$\hat a=\nu_0^2/4$")
    ax_a.set_xlabel(r"$\nu_0$ (cyc/aperture)")
    ax_a.set_ylabel(r"$\Delta z$ (mm)")
    ax_a.set_title(f"(a) Dense 80$\\times$150 FDPR sweep (A=0.1 waves, $\\sigma_r$=5 e$^-$, 1 trial/cell)",
                   fontsize=11)
    ax_a.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # ----- (b) row-mean RMS vs Δz with error band -----
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.fill_betweenx(dz, row_p10, row_p90, alpha=0.25, color="C0",
                       label="10th–90th pctile")
    ax_b.plot(row_mean, dz, "o-", color="C0", ms=3, lw=1.2, label="row mean")
    ax_b.axhspan(dz_transition_lo, dz_transition_hi, color="red", alpha=0.18)
    ax_b.axvline(truth_rms_nm, color="0.5", ls=":", lw=1.0,
                 label=f"truth RMS = {truth_rms_nm:.0f} nm")
    ax_b.invert_yaxis()
    ax_b.set_xlabel("residual RMS (nm)")
    ax_b.set_ylabel(r"$\Delta z$ (mm)")
    ax_b.set_title("(b) RMS vs $\\Delta z$", fontsize=11)
    ax_b.legend(fontsize=8, loc="lower right")
    ax_b.grid(True, alpha=0.25, lw=0.5)
    ax_b.invert_yaxis()   # restore (axhspan + invert ordering)

    # ----- (c) representative ν₀ slice -----
    ax_c = fig.add_subplot(gs[1, 0])
    for tv in [5, 10, 20, 40, 60]:
        j = int(np.argmin(np.abs(v0 - tv)))
        ax_c.plot(dz, rms[:, j], lw=1.4, label=fr"$\nu_0={v0[j]:.1f}$")
    ax_c.axvspan(dz_transition_lo, dz_transition_hi, color="red", alpha=0.18)
    ax_c.axhline(truth_rms_nm, color="0.5", ls=":", lw=1.0)
    ax_c.set_xlabel(r"$\Delta z$ (mm)")
    ax_c.set_ylabel("residual RMS (nm)")
    ax_c.set_title("(c) RMS slices at fixed $\\nu_0$", fontsize=11)
    ax_c.legend(fontsize=8, ncol=2, loc="upper left")
    ax_c.grid(True, alpha=0.25, lw=0.5)

    # ----- (d) â and blur-vs-grid bookkeeping -----
    ax_d = fig.add_subplot(gs[1, 1:])
    ax_d2 = ax_d.twinx()

    ahat_vals = ahat_from_dz_mm(dz)
    blur_vals = geom_blur_radius_um(dz)
    ax_d.plot(dz, ahat_vals, "C2-", lw=2.0, label=r"$\hat a$ (waves P-V)")
    ax_d2.plot(dz, blur_vals / HALF_GRID_UM, "C3-", lw=2.0,
               label="geometric blur / focal-grid half-width")
    ax_d.axvspan(dz_transition_lo, dz_transition_hi, color="red", alpha=0.18,
                 label="FDPR transition")
    ax_d.axhline(ahat_hi, color="C2", ls=":", lw=1.0)
    ax_d2.axhline(1.0, color="C3", ls=":", lw=1.0)

    ax_d.set_xlabel(r"$\Delta z$ (mm)")
    ax_d.set_ylabel(r"normalized defocus $\hat a$ (waves P-V)", color="C2")
    ax_d2.set_ylabel("blur radius / focal-grid half-width", color="C3")
    ax_d.tick_params(axis='y', colors="C2")
    ax_d2.tick_params(axis='y', colors="C3")
    ax_d.set_title(r"(d) Bookkeeping at the transition  "
                   rf"($\hat a={ahat_hi:.1f}$ waves, blur ratio {blur_at_transition/HALF_GRID_UM:.2f})",
                   fontsize=10)
    ax_d.grid(True, alpha=0.25, lw=0.5)

    lines1, labels1 = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left",
                framealpha=0.9)

    # Honest annotation: the blur ratio at the transition is ~0.79 (well
    # below 1), so PSF spilling off the grid is NOT the cause.  The
    # transition coincides with a-hat ~ 12 waves; the mechanism (likely
    # MFT-reverse undersampling of inner Fresnel rings) is not pinned down
    # by this diagnostic alone and requires a deeper look at the
    # image_sharpening package.
    ax_d.text(0.02, 0.02,
              "Blur stays inside the grid (ratio<1); PSF spillover is not the cause.\n"
              "Mechanism: likely MFT-reverse undersampling at high â; needs deeper diagnosis.",
              transform=ax_d.transAxes, fontsize=7.5, va="bottom",
              bbox=dict(boxstyle="round", facecolor="0.95", edgecolor="0.7", alpha=0.9))

    out = os.path.join(OUT, "dense_transition_diagnostic.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
