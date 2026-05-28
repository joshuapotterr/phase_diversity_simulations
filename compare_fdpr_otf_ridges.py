"""
Compare the FDPR-optimum and OTF-optimum ridges in the dense 80x150 sweep.

Code-audit fix #4: this version restricts BOTH ridges to the same dz range
(<=145 mm, the simulator's good regime), reports the rail-limited fraction,
and adds the "neither rail-limited" sub-regime (v0 < ~24 cyc/ap) where the
comparison is most meaningful.

Caveat we are honest about:
  * The OTF "ridge" (argmax_dz of H_photometry per v0) is rail-limited at
    the upper dz edge in ~2/3 of the v0 cells; for v0 > 24 cyc/ap the
    Dean-Bowers prediction nu0^2/4 exceeds 145 mm so the OTF wants
    bigger dz than the simulator's good regime allows.  The reported
    anti-correlation is meaningful in the v0 in [5, 24] regime; outside
    it the OTF ridge is dominated by the boundary.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper_figures_real")
os.makedirs(OUT, exist_ok=True)

LAMBDA_M = 650e-9
D_M = 10.12e-3
F_M = 500e-3
DZ_GOOD_MAX_MM = 145.0       # FDPR simulator good-regime cutoff


def dz_from_ahat(a):
    return a * 8 * LAMBDA_M * F_M ** 2 / D_M ** 2 * 1e3


def main():
    fdpr = np.load(os.path.join(HERE, "fdpr_dense_results", "fdpr_dense_completed.npz"))
    dz_f, v0_f, rms = fdpr["dz"], fdpr["v0"], fdpr["rms_mean"]

    otf = np.load(os.path.join(HERE, "OTF_heatmap_data_fixed.npz"))
    H = otf["H_photometry"].copy()
    mask = otf["H_valid_photometry"].astype(bool)
    H[~mask] = np.nan
    dz_o, v0_o = otf["fixed_dz_heatmap"], otf["v0_heatmap"]

    # Matched dz range
    good_f = dz_f <= DZ_GOOD_MAX_MM
    good_o = dz_o <= DZ_GOOD_MAX_MM
    dz_f_g, rms_g = dz_f[good_f], rms[good_f]
    dz_o_g, H_g = dz_o[good_o], H[good_o]

    # Per-v0 optima
    fdpr_opt = dz_f_g[np.argmin(np.where(np.isfinite(rms_g), rms_g, np.inf), axis=0)]
    otf_opt_raw = np.full(H_g.shape[1], np.nan)
    for j in range(H_g.shape[1]):
        col = H_g[:, j]
        if np.any(np.isfinite(col)):
            otf_opt_raw[j] = dz_o_g[np.nanargmax(col)]
    otf_opt = np.interp(v0_f, v0_o, otf_opt_raw)

    finite = np.isfinite(fdpr_opt) & np.isfinite(otf_opt)
    rs_all, ps_all = spearmanr(fdpr_opt[finite], otf_opt[finite])

    # v0 cutoff above which the OTF ridge (nu0^2/4) exceeds the matched range
    v0_rail = np.sqrt(4 * DZ_GOOD_MAX_MM * D_M ** 2 / (8 * LAMBDA_M * F_M ** 2 * 1e3))
    # Sub-regime where neither ridge is rail-limited
    sub = finite & (v0_f >= 5.0) & (v0_f <= v0_rail)
    rs_sub, ps_sub = spearmanr(fdpr_opt[sub], otf_opt[sub])

    otf_railed = otf_opt >= (DZ_GOOD_MAX_MM - 5)
    fdpr_railed = fdpr_opt >= (DZ_GOOD_MAX_MM - 5)

    print(f"=== Ridge comparison, both restricted to dz <= {DZ_GOOD_MAX_MM:.0f} mm ===")
    print(f"All v0:                              n={finite.sum()}  Spearman r={rs_all:+.3f}  p={ps_all:.2e}")
    print(f"v0 in [5, {v0_rail:.1f}] (neither rail-limited): n={sub.sum()}  "
          f"Spearman r={rs_sub:+.3f}  p={ps_sub:.2e}")
    print(f"OTF optimum at upper grid edge: {otf_railed.sum()}/{finite.sum()} cells  "
          f"({100*otf_railed.mean():.0f}%)")
    print(f"FDPR optimum at upper grid edge: {fdpr_railed.sum()}/{finite.sum()} cells  "
          f"({100*fdpr_railed.mean():.0f}%)")

    fig, axs = plt.subplots(1, 2, figsize=(14.5, 5.3))
    plt.subplots_adjust(wspace=0.30, left=0.06, right=0.99, top=0.91, bottom=0.11)

    # (a) heatmap with both ridges
    ax = axs[0]
    pcm = ax.pcolormesh(v0_f, dz_f_g, rms_g, shading="auto", cmap="viridis",
                        vmin=33, vmax=110)
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label="FDPR residual RMS (nm)")
    ax.scatter(v0_f, fdpr_opt, s=14, color="C3", edgecolor="black", lw=0.5,
               label="FDPR per-$\\nu_0$ optimum")
    ax.plot(v0_o, otf_opt_raw, "o", ms=3, color="lightblue", alpha=0.55,
            label="OTF per-$\\nu_0$ optimum")
    ax.plot(v0_f, dz_from_ahat(v0_f ** 2 / 4), "w--", lw=1.5,
            label=r"$\hat a=\nu_0^2/4$ (Dean-Bowers)")
    ax.axvline(v0_rail, color="orange", lw=1.0, ls=":",
               label=fr"$\nu_0={v0_rail:.1f}$: OTF ridge meets grid edge")
    ax.set_xlabel(r"$\nu_0$ (cyc/aperture)")
    ax.set_ylabel(r"$\Delta z$ (mm)")
    ax.set_xlim(v0_f.min(), v0_f.max())
    ax.set_ylim(0, DZ_GOOD_MAX_MM)
    ax.set_title(f"(a) FDPR heatmap (dz<={DZ_GOOD_MAX_MM:.0f} mm) with both per-$\\nu_0$ optima",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    # (b) ridge-vs-ridge scatter, both restricted to matched range
    ax = axs[1]
    ax.plot([0, DZ_GOOD_MAX_MM], [0, DZ_GOOD_MAX_MM], "0.7", ls=":", lw=1, label="1:1")
    sc1 = ax.scatter(otf_opt[finite & ~sub], fdpr_opt[finite & ~sub],
                     c="0.55", s=22, alpha=0.4,
                     label=fr"all $\nu_0$ ($r_s={rs_all:+.2f}$, n={finite.sum()})")
    sc2 = ax.scatter(otf_opt[sub], fdpr_opt[sub],
                     c=v0_f[sub], s=32, cmap="plasma", edgecolor="black", lw=0.4,
                     label=rf"$\nu_0\in[5,{v0_rail:.0f}]$ neither rail-limited "
                           rf"($r_s={rs_sub:+.2f}$, n={sub.sum()})")
    fig.colorbar(sc2, ax=ax, label=r"$\nu_0$ (cyc/ap, sub-regime)")
    ax.set_xlim(0, DZ_GOOD_MAX_MM); ax.set_ylim(0, DZ_GOOD_MAX_MM)
    ax.set_xlabel("OTF-ridge optimal $\\Delta z$ (mm)")
    ax.set_ylabel("FDPR-ridge optimal $\\Delta z$ (mm)")
    ax.set_title("(b) ridge-vs-ridge, matched dz range", fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92)

    out = os.path.join(OUT, "fdpr_otf_ridge_comparison.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
