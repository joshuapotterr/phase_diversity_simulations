"""
OTF Sideband Heatmap Analysis for Sequential Phase Diversity
=============================================================

Single unified analysis of OTF sideband structure across the (ν₀, Δz)
parameter space, with three pair-selection criteria:

  1. Total power       — aggregate sideband amplitude (amplitude-only)
  2. Signed correlation — Pearson r of response profiles (signed)
  3. Complementarity    — IQI-weighted (R1−R2)² score  (signed, broadband)

All metrics operate on a single data pipeline with signed OTF responses.
Where amplitude-only quantities are needed (e.g. total power, IQI 1/R),
|R| is taken explicitly and labeled as such.

The complementarity score I_eff = (R1−R2)² is a heuristic motivated by
the observation (Dean & Bowers 2003, Eq. 8) that the sideband modulation
sin(πν̂₀²/8â) produces contrast maxima and minima at defocus values
given by Eqs. (9) and (11).  Pairs whose maxima/minima patterns are
interleaved across ν₀ yield large (R1−R2)², which we take as a proxy
for complementary spectral information content.

This is *not* the Fisher information of Dean & Bowers Eq. 18 (which
involves PSF-intensity derivatives normalized by the PSF itself), but it
captures the same structural intuition: diversity channels that sample
different parts of the sideband modulation provide less redundant
constraints on the wavefront.
"""

import matplotlib.pyplot as plt
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

HEATMAP_FILE     = "OTF_heatmap_data.npz"
FIXED_DEFOCUS_MM = [113.5, 160.1]   # slices for 1-D plots (mid-to-high ν₀ regime)

# Frequency bands — defined by physical value, converted to indices at load time
V0_MIN_CYC       = 3.0     # cycles/aperture  (analysis band lower bound)
V0_MAX_CYC       = 40.0    # cycles/aperture  (analysis band upper bound)
PAIR_V0_MIN_CYC  = 0.0     # cycles/aperture  (pair-metric band lower bound)
PAIR_V0_MAX_CYC  = 55.0    # cycles/aperture  (pair-metric band upper bound)

INSPECT    = False
DZ1_INSP   = 153.9   # mm
DZ2_INSP   = 153.9   # mm
COLLAPSE   = True     # compute 2-D cumulative heatmaps
N_TOP      = 5        # top complementary pairs to report

# Separate floors for distinct numerical roles
AMP_FLOOR    = 1e-6   # below this, sideband amplitude treated as unmeasured
DENOM_FLOOR  = 1e-12  # denominator regularization (correlation, normalization)
RECIP_FLOOR  = 1e-6   # reciprocal guarding (IQI, 1/R surfaces)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def freq_to_idx(v0_arr, v_min, v_max):
    """Convert physical frequency bounds to array index range."""
    i_lo = int(np.searchsorted(v0_arr, v_min, side='left'))
    i_hi = int(np.searchsorted(v0_arr, v_max, side='right'))
    return max(i_lo, 0), min(i_hi, len(v0_arr))


def safe_reciprocal(arr, floor=RECIP_FLOOR):
    """1/arr, returning NaN where |arr| < floor or non-finite."""
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr) & (np.abs(arr) > floor)
    out[valid] = 1.0 / arr[valid]
    return out


def safe_nanmax(arr):
    """np.nanmax that returns NaN instead of raising on all-NaN input."""
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.nan
    return float(np.nanmax(arr))


def safe_nanargmax(arr):
    """np.nanargmax that returns None instead of raising on all-NaN input."""
    if not np.any(np.isfinite(arr)):
        return None
    return int(np.nanargmax(arr))


def safe_normalize(arr):
    """Divide by nanmax; returns all-NaN array if max is zero or NaN."""
    m = safe_nanmax(arr)
    if np.isnan(m) or m < DENOM_FLOOR:
        return np.full_like(arr, np.nan, dtype=float)
    return arr / m


def show_heatmap(data, extent, title, xlabel, ylabel, clabel,
                 figsize=(9, 7), plo=5, phi=95, cmap='viridis'):
    """Plot a 2-D heatmap with NaN/inf/degenerate guards."""
    plot_data = np.where(np.isfinite(data), data, np.nan)
    n_fin = np.sum(np.isfinite(plot_data))
    if n_fin == 0:
        print(f"  [skip] {title}: no finite data to plot.")
        return
    vmin = np.nanpercentile(plot_data, plo)
    vmax = np.nanpercentile(plot_data, phi)
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5

    _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plot_data, aspect='auto', origin='lower', extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=clabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (single pipeline, never reloaded or overwritten)
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path):
    loaded   = np.load(path)
    H        = loaded["H"].astype(float)
    dz       = loaded["fixed_dz_heatmap"].astype(float)
    v0       = loaded["v0_heatmap"].astype(float)

    if H.shape[1] != len(v0):
        n = min(H.shape[1], len(v0))
        print(f"Shape mismatch: trimming to {n} frequency points.")
        H, v0 = H[:, :n], v0[:n]

    assert np.all(np.diff(v0) > 0), "v0_heatmap must be monotonically increasing"
    return H, dz, v0


H_raw, fixed_dz, v0_heatmap = load_data(HEATMAP_FILE)
Nd  = len(fixed_dz)
Nv0 = len(v0_heatmap)

# Convert physical frequency bounds → index ranges (issue #9)
V0_MIN_IDX, V0_MAX_IDX         = freq_to_idx(v0_heatmap, V0_MIN_CYC, V0_MAX_CYC)
PAIR_V0_MIN_IDX, PAIR_V0_MAX_IDX = freq_to_idx(v0_heatmap, PAIR_V0_MIN_CYC, PAIR_V0_MAX_CYC)

# The ONE signed response array everything derives from (issue #10)
# Non-finite and sub-floor amplitudes → NaN, sign preserved
R_all = H_raw.copy()
R_all[~np.isfinite(R_all)] = np.nan
R_all[np.abs(R_all) < AMP_FLOOR] = np.nan

# Convenient band slices — copies, not views, to prevent accidental R_all mutation
v0_band      = v0_heatmap[V0_MIN_IDX:V0_MAX_IDX]
v0_pair_band = v0_heatmap[PAIR_V0_MIN_IDX:PAIR_V0_MAX_IDX]
R_band       = R_all[:, V0_MIN_IDX:V0_MAX_IDX].copy()            # signed
R_pair       = R_all[:, PAIR_V0_MIN_IDX:PAIR_V0_MAX_IDX].copy()   # signed

# Guard against empty frequency bands
if len(v0_band) == 0:
    raise ValueError(
        f"Analysis band is empty: V0_MIN_CYC={V0_MIN_CYC}, V0_MAX_CYC={V0_MAX_CYC} "
        f"yielded indices [{V0_MIN_IDX}:{V0_MAX_IDX}] in v0_heatmap "
        f"(range {v0_heatmap[0]:.1f}–{v0_heatmap[-1]:.1f} cyc/ap).")
if len(v0_pair_band) == 0:
    raise ValueError(
        f"Pair band is empty: PAIR_V0_MIN_CYC={PAIR_V0_MIN_CYC}, "
        f"PAIR_V0_MAX_CYC={PAIR_V0_MAX_CYC} yielded indices "
        f"[{PAIR_V0_MIN_IDX}:{PAIR_V0_MAX_IDX}] in v0_heatmap "
        f"(range {v0_heatmap[0]:.1f}–{v0_heatmap[-1]:.1f} cyc/ap).")

dz_extent  = [fixed_dz.min(), fixed_dz.max(), fixed_dz.min(), fixed_dz.max()]
v0_extent  = [v0_heatmap.min(), v0_heatmap.max(), fixed_dz.min(), fixed_dz.max()]


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATION FUNCTIONS (no plotting — issue #16)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ns(R_signed):
    """Net Sensitivity: quadrature sum of |R| over ν₀ band."""
    return np.sqrt(np.nansum(R_signed**2, axis=1))


def compute_iqi_1d(R_signed):
    """1-D IQI effective response: 1/sqrt(Σ (1/|R|)²), per defocus row."""
    R_abs = np.abs(R_signed)
    R_inv = safe_reciprocal(R_abs)
    inv_sq_sum = np.nansum(R_inv**2, axis=1)
    inv_sq_sum[inv_sq_sum < DENOM_FLOOR] = np.nan
    return 1.0 / np.sqrt(inv_sq_sum)


def compute_pair_total_power(R_signed):
    """
    Total amplitude power per pair: sqrt(Σ_ν₀ (|R1|² + |R2|²)).

    This is purely an aggregate signal-strength metric; it does NOT
    measure complementarity.  A pair of identical defocus values will
    score high if both are individually strong.
    """
    R_abs = np.abs(R_signed)
    R_sq  = R_abs**2                             # (Nd, Nv)
    # Vectorized: for each pair (i,j), we need sum_k (Ri_k² + Rj_k²)
    # over jointly-finite k.
    X = np.where(np.isfinite(R_sq), R_sq, 0.0)  # NaN → 0
    M = np.isfinite(R_sq).astype(float)          # mask
    # sum_k Ri² where Rj is also finite:
    S_i_given_j = X @ M.T                        # (Nd, Nd)
    S_j_given_i = M @ X.T                        # (Nd, Nd)
    n_finite    = M @ M.T
    total_sq    = S_i_given_j + S_j_given_i
    total_sq[n_finite < 1] = np.nan
    return np.sqrt(total_sq)


def compute_pair_power_normalized(R_signed):
    """
    Pair total power normalized so that pair(i,i) = sqrt(2).

    NOTE: the self-pair is sqrt(2), not 1, because combining a channel
    with itself gives sqrt(2|R|²)/sqrt(|R|²) = sqrt(2).  This is a
    mathematical consequence, not a bug — see derivation in docstring.
    """
    R_abs   = np.abs(R_signed)
    NS_self = np.sqrt(np.nanmean(R_abs**2, axis=1))     # per-defocus RMS

    # Mean power per pair (need mean, not sum, for normalization)
    R_sq  = R_abs**2
    X = np.where(np.isfinite(R_sq), R_sq, 0.0)
    M = np.isfinite(R_sq).astype(float)
    S_ij = X @ M.T + M @ X.T
    N_ij = M @ M.T
    N_ij[N_ij < 1] = np.nan
    pair_mean_rms = np.sqrt(S_ij / N_ij)

    ns_outer = np.sqrt(np.outer(NS_self, NS_self))
    ns_outer[ns_outer < DENOM_FLOOR] = np.nan
    return pair_mean_rms / ns_outer


def compute_signed_correlation(R_signed):
    """
    Pearson correlation of signed response profiles across ν₀.

    Mean-centering is performed over the *jointly-finite* domain for
    each pair (i,j), not over each row's own finite elements.  This is
    achieved via the covariance identity:

        r_ij = (n·Σxy − Σx·Σy) / sqrt((n·Σx² − (Σx)²)(n·Σy² − (Σy)²))

    where all sums run over k ∈ F(i,j) = {k : both R_i[k] and R_j[k] finite}.
    Each of the five terms (n, Σxy, Σx, Σy, Σx², Σy²) is an (Nd, Nd) matrix
    computed as a single matrix product.

    Returns (Nd, Nd) array with values in [-1, +1].
    """
    X    = np.where(np.isfinite(R_signed), R_signed, 0.0)
    M    = np.isfinite(R_signed).astype(float)
    X_sq = X**2

    n      = M @ M.T            # jointly-finite count per pair
    sum_xy = X @ X.T            # Σ_F  xi·xj
    sum_x  = X @ M.T            # Σ_F  xi   (sum of row i restricted to j-finite)
    sum_y  = M @ X.T            # Σ_F  xj   (sum of row j restricted to i-finite)
    sum_x2 = X_sq @ M.T         # Σ_F  xi²
    sum_y2 = M @ X_sq.T         # Σ_F  xj²

    numer = n * sum_xy - sum_x * sum_y
    var_x = n * sum_x2 - sum_x**2
    var_y = n * sum_y2 - sum_y**2

    denom = np.sqrt(var_x * var_y)
    denom[denom < DENOM_FLOOR] = np.nan

    corr = np.where(n >= 3, numer / denom, np.nan)
    return corr


def compute_complementarity_iqi(R_signed):
    """
    IQI-weighted complementarity score for all defocus pairs.

    Per spatial frequency:
        I_eff(ν₀) = (R1(ν₀) − R2(ν₀))²

    This is large when one response is positive and the other negative
    (anti-correlated), and small when both are similar (redundant).

    Aggregated via harmonic mean:
        score = 1 / sqrt( Σ_ν₀  1/I_eff(ν₀) )

    This ensures the score is limited by the weakest frequency bin,
    favoring pairs with uniform broadband coverage over those that
    are strong in one regime but blind in another.

    DESIGN NOTE: this is a heuristic complementarity metric, not the
    Fisher information of Dean & Bowers (2003) Eq. 18.  It is motivated
    by the same structural observation — that the sideband modulation
    sin(πν̂₀²/8â) produces interleaved maxima/minima across defocus
    (Eqs. 9, 11) — but it operates on measured OTF amplitudes rather
    than PSF-intensity derivatives.
    """
    N = R_signed.shape[0]
    ui, uj = np.triu_indices(N, k=1)

    R1 = R_signed[ui, :]                         # (N_pairs, Nv)
    R2 = R_signed[uj, :]
    I_eff = (R1 - R2)**2
    I_eff[I_eff < RECIP_FLOOR**2] = np.nan

    n_fin   = np.sum(np.isfinite(I_eff), axis=1)
    inv_sum = np.nansum(1.0 / I_eff, axis=1)
    inv_sum[inv_sum < DENOM_FLOOR] = np.nan
    inv_sum[n_fin < 3]             = np.nan
    iqi_vals = 1.0 / np.sqrt(inv_sum)

    out = np.full((N, N), np.nan)
    out[ui, uj] = iqi_vals
    out[uj, ui] = iqi_vals
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL METRICS
# ═══════════════════════════════════════════════════════════════════════════════

# Single-defocus metrics (over analysis band)
NS_dz      = compute_ns(R_band)
NS_dz_norm = safe_normalize(NS_dz)
R_eff      = compute_iqi_1d(R_band)
R_eff_norm = safe_normalize(R_eff)

# Pair metrics (over pair band)
pair_power      = compute_pair_total_power(R_pair)
pair_power_norm = compute_pair_power_normalized(R_pair)
corr_signed     = compute_signed_correlation(R_pair)
iqi_pair        = compute_complementarity_iqi(R_pair)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

# ── Raw OTF heatmap ───────────────────────────────────────────────────────────
show_heatmap(H_raw, v0_extent,
             title="OTF side-peak amplitude heatmap vs (dz, v0)",
             xlabel="Spatial frequency v0 [cycles/aperture]",
             ylabel="Defocus dz [mm]",
             clabel="OTF side-peak amplitude (unnormalized)",
             plo=0, phi=100)


# ── OTF slices at selected defocus values ─────────────────────────────────────
row_idxs  = [int(np.argmin(np.abs(fixed_dz - dz))) for dz in FIXED_DEFOCUS_MM]
dz_snapto = [float(fixed_dz[i]) for i in row_idxs]

plt.figure(figsize=(11, 6))
for dz_fixed, i in zip(dz_snapto, row_idxs):
    row    = R_all[i, :]
    finite = np.isfinite(row)
    plt.plot(v0_heatmap[finite], row[finite], label=fr'dz = {dz_fixed:.1f} mm')
plt.legend()
plt.xlabel("Spatial frequency v0 [cycles/aperture]")
plt.ylabel("OTF side-peak amplitude")
plt.title("OTF amplitude vs spatial frequency for selected defocus values")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()


# ── Inverse sensitivity (proxy noise floor) ───────────────────────────────────
rows       = np.array([R_all[i, :] for i in row_idxs])
R_combined = np.sqrt(np.nansum(rows**2, axis=0))
inv_R      = safe_reciprocal(R_combined)
finite_inv = np.isfinite(inv_R)

if np.any(finite_inv):
    plt.figure(figsize=(9, 6))
    plt.plot(v0_heatmap[finite_inv], inv_R[finite_inv], color='k', lw=2)
    plt.yscale('log')
    plt.xlabel(r"Spatial frequency v0 [cycles/aperture]")
    plt.ylabel(r"Relative inverse sensitivity 1/|R|")
    plt.title(r"Inverse quadrature-combined response (proxy noise floor)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ── Pair total power heatmap ──────────────────────────────────────────────────
show_heatmap(pair_power, dz_extent,
             title=(f"Defocus-pair total amplitude power "
                    f"(v0 = {v0_pair_band[0]:.0f}–{v0_pair_band[-1]:.0f} cyc/ap)"),
             xlabel="Defocus dz1 [mm]", ylabel="Defocus dz2 [mm]",
             clabel="sqrt(Σ |R1|² + |R2|²)  [amplitude only, not complementarity]")


# ── Pair normalized power (diagonal = √2) ────────────────────────────────────
show_heatmap(pair_power_norm, dz_extent,
             title=(f"Defocus-pair normalized amplitude power "
                    r"(diagonal = $\sqrt{2}$, not 1)"),
             xlabel="Defocus dz1 [mm]", ylabel="Defocus dz2 [mm]",
             clabel="Normalized pair power", plo=0.5, phi=99.5)


# ── Signed correlation heatmap ────────────────────────────────────────────────
show_heatmap(corr_signed, dz_extent,
             title=(f"Signed Pearson correlation of OTF response profiles\n"
                    f"(v0 = {v0_pair_band[0]:.0f}–{v0_pair_band[-1]:.0f} cyc/ap)"),
             xlabel="dz1 [mm]", ylabel="dz2 [mm]",
             clabel="Pearson r  (blue = anti-correlated)",
             cmap='RdBu_r', plo=0, phi=100)


# ── Complementarity IQI heatmap ───────────────────────────────────────────────
show_heatmap(iqi_pair, dz_extent,
             title=("IQI complementarity score (heuristic)\n"
                    r"$1\,/\,\sqrt{\Sigma\, 1/(R_1 - R_2)^2}$  "
                    "(higher = better broadband coverage)"),
             xlabel="dz1 [mm]", ylabel="dz2 [mm]",
             clabel="IQI complementarity (design metric, not Fisher info)",
             cmap='magma')


# ── NS and IQI 1-D curves ────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(fixed_dz, R_eff_norm, lw=2, color='crimson', label='IQI Effective Response')
plt.scatter(fixed_dz, R_eff_norm, s=25, color='crimson', alpha=0.8)
plt.plot(fixed_dz, NS_dz_norm, '--k', alpha=0.6, label='Net Sensitivity NS(dz)')
plt.xlabel(r"Defocus dz [mm]")
plt.ylabel(r"Normalized response")
plt.title(f"IQI vs NS over v0 = {v0_band[0]:.1f}–{v0_band[-1]:.1f} cyc/ap")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

idx_ns  = safe_nanargmax(NS_dz_norm)
idx_iqi = safe_nanargmax(R_eff_norm)
if idx_ns  is not None: print(f"Optimal dz (NS)  = {fixed_dz[idx_ns]:.2f} mm")
if idx_iqi is not None: print(f"Optimal dz (IQI) = {fixed_dz[idx_iqi]:.2f} mm")


# ═══════════════════════════════════════════════════════════════════════════════
# PAIR SELECTION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

mask_ut = np.triu(np.ones((Nd, Nd), dtype=bool), k=1)

def find_best(metric, label, maximize=True):
    """Find and print the best pair under a given metric."""
    m = np.where(mask_ut & np.isfinite(metric), metric, np.nan)
    idx = safe_nanargmax(m) if maximize else safe_nanargmax(-m)
    if idx is None:
        print(f"  {label}: no valid pairs.")
        return None
    i, j = np.unravel_index(idx, m.shape)
    val  = metric[i, j]
    r    = corr_signed[i, j] if np.isfinite(corr_signed[i, j]) else float('nan')
    print(f"  {label}:  dz1 = {fixed_dz[i]:.1f},  dz2 = {fixed_dz[j]:.1f} mm  "
          f"(metric = {val:.4f},  r = {r:.3f})")
    return (i, j)

print(f"\n{'='*72}")
print("Best defocus pair under each criterion")
print(f"(pair band: {v0_pair_band[0]:.0f}–{v0_pair_band[-1]:.0f} cyc/ap)")
print(f"{'='*72}")
best_power = find_best(pair_power,  "Total amplitude power      ")
best_neg_r = find_best(corr_signed, "Most anti-correlated (min r)", maximize=False)
best_iqi   = find_best(iqi_pair,    "IQI complementarity (broadband)")
print(f"{'='*72}")
print("NOTE: These are different optimization criteria and generally yield")
print("different pairs.  Total power favors strong channels; min-r favors")
print("opposite sign patterns; IQI complementarity favors uniform broadband")
print("coverage limited by the weakest frequency bin.")


# ═══════════════════════════════════════════════════════════════════════════════
# TOP ANTI-CORRELATED PAIRS — overlay plots
# ═══════════════════════════════════════════════════════════════════════════════

corr_search = np.where(mask_ut & np.isfinite(corr_signed),
                        corr_signed, np.inf)
flat_rank   = np.argsort(corr_search.ravel())

print(f"\nTop {N_TOP} most anti-correlated pairs:")
print(f"{'Rank':<6} {'dz1 [mm]':<12} {'dz2 [mm]':<12} {'Pearson r':<12} {'IQI score':<12}")
print("-" * 54)

top_pairs = []
for rank, fi in enumerate(flat_rank[:N_TOP]):
    i, j = np.unravel_index(fi, corr_search.shape)
    r_val   = corr_signed[i, j]
    iqi_val = iqi_pair[i, j] if np.isfinite(iqi_pair[i, j]) else float('nan')
    print(f"{rank+1:<6} {fixed_dz[i]:<12.1f} {fixed_dz[j]:<12.1f} "
          f"{r_val:<12.4f} {iqi_val:<12.4f}")
    top_pairs.append((i, j, r_val))

# Guard bar_width for irregular or short v0 arrays (issue #8)
if len(v0_pair_band) >= 2:
    bar_width = np.median(np.diff(v0_pair_band)) * 0.8
else:
    bar_width = 1.0

for rank, (i, j, r_val) in enumerate(top_pairs[:3]):
    dz1, dz2 = fixed_dz[i], fixed_dz[j]
    r1, r2   = R_pair[i, :], R_pair[j, :]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})

    axes[0].plot(v0_pair_band, r1, lw=2, color='tab:blue',
                 label=fr'dz$_1$ = {dz1:.1f} mm')
    axes[0].plot(v0_pair_band, r2, lw=2, color='tab:red',
                 label=fr'dz$_2$ = {dz2:.1f} mm')
    axes[0].axhline(0, color='gray', ls=':', alpha=0.5)
    axes[0].set_ylabel('OTF sideband amplitude (signed)')
    axes[0].set_title(f'Anti-correlated pair #{rank+1}: '
                      fr'dz = {dz1:.1f} vs {dz2:.1f} mm  (r = {r_val:.3f})')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    product  = r1 * r2
    finite_p = np.isfinite(product)
    if np.any(finite_p):
        colors = np.where(product[finite_p] < 0, 'tab:green', 'tab:orange')
        axes[1].bar(v0_pair_band[finite_p], product[finite_p],
                    width=bar_width, color=colors, alpha=0.7)
    axes[1].axhline(0, color='gray', ls=':', alpha=0.5)
    axes[1].set_xlabel('Spatial frequency v0 [cycles/aperture]')
    axes[1].set_ylabel('R1 × R2')
    axes[1].set_title('Product (green = opposite sign = complementary)',
                      fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Best IQI complementary pair overlay ───────────────────────────────────────
if best_iqi is not None:
    i, j = best_iqi
    r1, r2 = R_pair[i, :], R_pair[j, :]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(v0_pair_band, r1, alpha=0.3, color='tab:blue')
    ax.fill_between(v0_pair_band, r2, alpha=0.3, color='tab:red')
    ax.plot(v0_pair_band, r1, lw=2, color='tab:blue',
            label=fr'dz$_1$ = {fixed_dz[i]:.1f} mm')
    ax.plot(v0_pair_band, r2, lw=2, color='tab:red',
            label=fr'dz$_2$ = {fixed_dz[j]:.1f} mm')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Spatial frequency v0 [cycles/aperture]')
    ax.set_ylabel('OTF sideband amplitude (signed)')
    ax.set_title(f'Best IQI complementary pair: dz = {fixed_dz[i]:.1f} vs '
                 f'{fixed_dz[j]:.1f} mm  '
                 f'(r = {corr_signed[i,j]:.3f}, '
                 f'IQI = {iqi_pair[i,j]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: inspect a specific pair
# ═══════════════════════════════════════════════════════════════════════════════

if INSPECT:
    i = int(np.argmin(np.abs(fixed_dz - DZ1_INSP)))
    j = int(np.argmin(np.abs(fixed_dz - DZ2_INSP)))
    print(f"\nInspecting dz1 = {fixed_dz[i]:.2f} mm, dz2 = {fixed_dz[j]:.2f} mm")
    if np.isfinite(pair_power[i, j]):
        print(f"  Total power     = {pair_power[i, j]:.4f}")
    if np.isfinite(corr_signed[i, j]):
        print(f"  Pearson r       = {corr_signed[i, j]:.4f}")
    if np.isfinite(iqi_pair[i, j]):
        print(f"  IQI complement. = {iqi_pair[i, j]:.4f}")

    r_combo = np.sqrt(R_pair[i, :]**2 + R_pair[j, :]**2)
    finite  = np.isfinite(r_combo)
    if np.any(finite):
        plt.figure(figsize=(8, 5))
        plt.plot(v0_pair_band[finite], r_combo[finite])
        plt.xlabel(r"Spatial frequency v0 [cycles/aperture]")
        plt.ylabel("Combined response amplitude |R|")
        plt.title(fr"Amplitude-combined response: dz = {fixed_dz[i]:.1f} vs {fixed_dz[j]:.1f} mm")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: 2-D cumulative heatmaps
# ═══════════════════════════════════════════════════════════════════════════════

if COLLAPSE:
    band_extent = [v0_band[0], v0_band[-1], fixed_dz.min(), fixed_dz.max()]

    R_abs_band = np.abs(R_band)
    R_abs_band[R_abs_band < AMP_FLOOR] = np.nan

    # Cumulative quadrature sum: sqrt(cumsum(|R|²))
    R_quad_2D = np.sqrt(np.nancumsum(R_abs_band**2, axis=1))
    show_heatmap(safe_normalize(R_quad_2D), band_extent,
                 title="Cumulative quadrature response (normalized)",
                 xlabel="Spatial frequency v0 [cycles/aperture]",
                 ylabel="Defocus dz [mm]",
                 clabel="Cumulative quadrature (normalized)")

    # Cumulative IQI: 1/sqrt(cumsum(1/|R|²))
    R_inv_sq      = safe_reciprocal(R_abs_band)**2
    cum_inv_sq    = np.nancumsum(R_inv_sq, axis=1)
    cum_inv_sq[cum_inv_sq < DENOM_FLOOR] = np.nan
    R_IQI_band_2D = 1.0 / np.sqrt(cum_inv_sq)

    show_heatmap(R_IQI_band_2D, band_extent,
                 title="Cumulative IQI effective response",
                 xlabel="Spatial frequency v0 [cycles/aperture]",
                 ylabel="Defocus dz [mm]",
                 clabel="Cumulative IQI R(dz, v0)")

    # 1/|R| surface
    show_heatmap(safe_reciprocal(R_abs_band), band_extent,
                 title="Inverse OTF response surface 1/|R|(dz, v0)",
                 xlabel="Spatial frequency v0 [cycles/aperture]",
                 ylabel="Defocus dz [mm]",
                 clabel="1 / |R(dz, v0)|")