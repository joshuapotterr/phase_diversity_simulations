# Paper Figures from Actual Simulation Data
# ==========================================
# Uses OTF_heatmap_data.npz to create publication-quality figures

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Tuple
import os

# Set publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'dejavuserif',
})

# Color scheme (colorblind-friendly)
COLORS = {
    'blue': '#0072B2',
    'orange': '#D55E00', 
    'green': '#009E73',
    'pink': '#CC79A7',
    'yellow': '#F0E442',
    'cyan': '#56B4E9',
    'red': '#E69F00',
}


# =============================================================================
# LOAD DATA
# =============================================================================

def load_otf_data(filepath: str) -> dict:
    """Load OTF heatmap data from NPZ file."""
    data = np.load(filepath)
    
    return {
        'H': data['H'],
        'H_photometry': data['H_photometry'],
        'dz': data['fixed_dz_heatmap'],
        'v0': data['v0_heatmap'],
        'm_waves': float(data['m_waves']),
    }


# =============================================================================
# PHYSICS HELPERS
# =============================================================================

# SEAL optical parameters
F_M = 0.5           # Focal length [m]
D_M = 10.12e-3      # Pupil diameter [m]
LAM_M = 650e-9      # Wavelength [m]
F_NUMBER = F_M / D_M


def dz_to_a_hat(dz_mm: np.ndarray) -> np.ndarray:
    """Convert mechanical defocus (mm) to normalized defocus â (waves P-V)."""
    dz_m = dz_mm * 1e-3
    return (dz_m * D_M**2) / (8 * F_M**2 * LAM_M)


def a_hat_to_dz(a_hat: np.ndarray) -> np.ndarray:
    """Convert â (waves P-V) to mechanical defocus (mm)."""
    dz_m = 8 * a_hat * F_M**2 * LAM_M / D_M**2
    return dz_m * 1e3


def optimal_a_hat(v0: np.ndarray) -> np.ndarray:
    """Dean & Bowers optimal defocus: â = v₀²/4."""
    return v0**2 / 4


# =============================================================================
# FIGURE 1: OTF HEATMAP (PRIMARY RESULT)
# =============================================================================

def figure_otf_heatmap(data: dict, save_path: Optional[str] = None,
                       use_photometry: bool = True, show_theory: bool = True):
    """
    Main OTF heatmap figure showing side-peak amplitude vs (dz, v0).
    """
    dz = data['dz']
    v0 = data['v0']
    H = data['H_photometry'] if use_photometry else data['H']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    extent = [v0.min(), v0.max(), dz.min(), dz.max()]
    
    # Use percentile clipping for color scale
    valid = np.isfinite(H)
    if np.sum(valid) > 0:
        vmin, vmax = np.nanpercentile(H[valid], [2, 98])
    else:
        vmin, vmax = 0, 1
    
    im = ax.imshow(H, aspect='auto', origin='lower', extent=extent,
                   cmap='viridis', vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('OTF Side-Peak Amplitude [a.u.]')
    
    # Overlay Dean & Bowers theory
    if show_theory:
        v0_theory = np.linspace(v0.min(), v0.max(), 200)
        a_optimal = optimal_a_hat(v0_theory)
        dz_optimal = a_hat_to_dz(a_optimal)
        
        # Only plot where within range
        mask = dz_optimal <= dz.max()
        if np.any(mask):
            ax.plot(v0_theory[mask], dz_optimal[mask], 'r--', lw=2.5,
                    label=r'Theory: $\hat{a} = v_0^2/4$')
            ax.legend(loc='upper left', fontsize=11)
    
    ax.set_xlabel(r'Spatial Frequency $v_0$ [cycles/aperture]')
    ax.set_ylabel(r'Defocus $\Delta z$ [mm]')
    ax.set_title('OTF Side-Peak Amplitude')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# FIGURE 2: OTF SLICES AT FIXED DEFOCUS
# =============================================================================

def figure_otf_slices_fixed_dz(data: dict, dz_targets: list = [25, 50, 100, 150, 200],
                                save_path: Optional[str] = None):
    """
    Plot OTF amplitude vs spatial frequency at fixed defocus values.
    """
    dz = data['dz']
    v0 = data['v0']
    H = data['H_photometry']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(dz_targets)))
    
    for i, dz_target in enumerate(dz_targets):
        idx = np.argmin(np.abs(dz - dz_target))
        actual_dz = dz[idx]
        row = H[idx, :]
        
        valid = np.isfinite(row)
        ax.plot(v0[valid], row[valid], '-', color=colors[i], lw=2,
                label=f'Δz = {actual_dz:.0f} mm')
    
    ax.set_xlabel(r'Spatial Frequency $v_0$ [cycles/aperture]')
    ax.set_ylabel('OTF Side-Peak Amplitude [a.u.]')
    ax.set_title('OTF Amplitude vs Spatial Frequency')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(v0.min(), v0.max())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# FIGURE 3: OTF SLICES AT FIXED FREQUENCY
# =============================================================================

def figure_otf_slices_fixed_v0(data: dict, v0_targets: list = [5, 10, 20, 40, 60],
                                save_path: Optional[str] = None):
    """
    Plot OTF amplitude vs defocus at fixed spatial frequencies.
    Shows the oscillatory sin² behavior from Dean & Bowers.
    """
    dz = data['dz']
    v0 = data['v0']
    H = data['H_photometry']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], 
              COLORS['pink'], COLORS['cyan']]
    
    for i, v0_target in enumerate(v0_targets):
        idx = np.argmin(np.abs(v0 - v0_target))
        actual_v0 = v0[idx]
        col = H[:, idx]
        
        valid = np.isfinite(col)
        ax.plot(dz[valid], col[valid], '-', color=colors[i % len(colors)], lw=2,
                label=f'$v_0$ = {actual_v0:.0f} cyc/ap')
    
    ax.set_xlabel(r'Defocus $\Delta z$ [mm]')
    ax.set_ylabel('OTF Side-Peak Amplitude [a.u.]')
    ax.set_title('OTF Amplitude vs Defocus')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(dz.min(), dz.max())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# FIGURE 4: OPTIMAL DEFOCUS VS FREQUENCY
# =============================================================================

def figure_optimal_defocus(data: dict, save_path: Optional[str] = None):
    """
    Plot optimal defocus (argmax of OTF amplitude) vs spatial frequency.
    Compare to Dean & Bowers theory.
    """
    dz = data['dz']
    v0 = data['v0']
    H = data['H_photometry']
    
    # Find optimal dz for each v0
    optimal_dz = []
    optimal_amp = []
    valid_v0 = []
    
    for j in range(len(v0)):
        col = H[:, j]
        if np.any(np.isfinite(col)):
            i_max = np.nanargmax(col)
            optimal_dz.append(dz[i_max])
            optimal_amp.append(col[i_max])
            valid_v0.append(v0[j])
    
    optimal_dz = np.array(optimal_dz)
    valid_v0 = np.array(valid_v0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Optimal dz vs v0
    ax = axes[0]
    ax.scatter(valid_v0, optimal_dz, s=20, alpha=0.7, c=COLORS['blue'],
               label='Measured optimal')
    
    # Theory curve
    v0_theory = np.linspace(v0.min(), v0.max(), 200)
    dz_theory = a_hat_to_dz(optimal_a_hat(v0_theory))
    mask = dz_theory <= dz.max() * 1.1
    ax.plot(v0_theory[mask], dz_theory[mask], 'r-', lw=2.5,
            label=r'Theory: $\hat{a} = v_0^2/4$')
    
    ax.set_xlabel(r'Spatial Frequency $v_0$ [cycles/aperture]')
    ax.set_ylabel(r'Optimal Defocus $\Delta z$ [mm]')
    ax.set_title('(a) Optimal Defocus vs Spatial Frequency')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, v0.max())
    ax.set_ylim(0, dz.max() * 1.05)
    
    # Right: Histogram of optimal dz values
    ax = axes[1]
    ax.hist(optimal_dz, bins=20, alpha=0.7, color=COLORS['blue'],
            edgecolor='black', linewidth=1)
    ax.axvline(dz.max(), color='red', linestyle='--', lw=2,
               label=f'Grid limit ({dz.max():.0f} mm)')
    ax.set_xlabel(r'Optimal Defocus $\Delta z$ [mm]')
    ax.set_ylabel('Count (number of frequencies)')
    ax.set_title('(b) Distribution of Optimal Defocus')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# FIGURE 5: DEAN & BOWERS THEORY COMPARISON
# =============================================================================

def figure_dean_bowers_theory(data: dict, v0_select: float = 10.0,
                               save_path: Optional[str] = None):
    """
    Compare measured OTF amplitude to Dean & Bowers sin² theory.
    """
    dz = data['dz']
    v0 = data['v0']
    H = data['H_photometry']
    
    # Find closest v0
    idx_v0 = np.argmin(np.abs(v0 - v0_select))
    actual_v0 = v0[idx_v0]
    measured = H[:, idx_v0]
    
    # Compute â for each dz
    a_hat = dz_to_a_hat(dz)
    
    # Dean & Bowers theory: OTF amplitude ~ |sin(π v₀² / 8â)|
    # Avoid division by zero
    arg = np.pi * actual_v0**2 / (8 * np.maximum(a_hat, 0.01))
    theory = np.abs(np.sin(arg))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Plot vs â
    ax = axes[0]
    
    valid = np.isfinite(measured)
    measured_norm = measured / np.nanmax(measured)  # Normalize for comparison
    
    ax.plot(a_hat[valid], measured_norm[valid], 'o-', color=COLORS['blue'],
            lw=1.5, ms=4, alpha=0.7, label='Measured (normalized)')
    ax.plot(a_hat, theory, 'r-', lw=2, label=r'Theory: $|\sin(\pi v_0^2 / 8\hat{a})|$')
    
    ax.set_xlabel(r'Normalized Defocus $\hat{a}$ [waves P-V]')
    ax.set_ylabel('OTF Amplitude (normalized)')
    ax.set_title(f'(a) Comparison to Theory ($v_0$ = {actual_v0:.0f} cyc/ap)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, a_hat.max())
    ax.set_ylim(0, 1.1)
    
    # Right: Plot vs dz
    ax = axes[1]
    
    ax.plot(dz[valid], measured_norm[valid], 'o-', color=COLORS['blue'],
            lw=1.5, ms=4, alpha=0.7, label='Measured')
    ax.plot(dz, theory, 'r-', lw=2, label='Theory')
    
    # Mark theoretical maxima
    # Maxima occur when sin(arg) = ±1, i.e., arg = (2n-1)π/2
    # So: π v₀² / 8â = (2n-1)π/2  =>  â = v₀² / [4(2n-1)]
    for n in range(1, 6):
        a_max = actual_v0**2 / (4 * (2*n - 1))
        dz_max = a_hat_to_dz(a_max)
        if dz_max <= dz.max():
            ax.axvline(dz_max, color='gray', linestyle=':', alpha=0.5)
            ax.text(dz_max, 1.05, f'n={n}', ha='center', fontsize=9, color='gray')
    
    ax.set_xlabel(r'Defocus $\Delta z$ [mm]')
    ax.set_ylabel('OTF Amplitude (normalized)')
    ax.set_title(f'(b) Measured vs Theory ($v_0$ = {actual_v0:.0f} cyc/ap)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, dz.max())
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# FIGURE 6: DUAL-AXIS HEATMAP (dz and â)
# =============================================================================

def figure_heatmap_dual_axis(data: dict, save_path: Optional[str] = None):
    """
    OTF heatmap with both dz (mm) and â (waves) on y-axis.
    """
    dz = data['dz']
    v0 = data['v0']
    H = data['H_photometry']
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    extent = [v0.min(), v0.max(), dz.min(), dz.max()]
    
    valid = np.isfinite(H)
    vmin, vmax = np.nanpercentile(H[valid], [2, 98])
    
    im = ax1.imshow(H, aspect='auto', origin='lower', extent=extent,
                    cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Primary y-axis: dz in mm
    ax1.set_xlabel(r'Spatial Frequency $v_0$ [cycles/aperture]')
    ax1.set_ylabel(r'Defocus $\Delta z$ [mm]')
    
    # Secondary y-axis: â in waves
    ax2 = ax1.secondary_yaxis('right', functions=(dz_to_a_hat, a_hat_to_dz))
    ax2.set_ylabel(r'Normalized Defocus $\hat{a}$ [waves P-V]')
    
    # Theory overlay
    v0_theory = np.linspace(v0.min(), v0.max(), 200)
    dz_theory = a_hat_to_dz(optimal_a_hat(v0_theory))
    mask = dz_theory <= dz.max()
    ax1.plot(v0_theory[mask], dz_theory[mask], 'r--', lw=2.5,
             label=r'$\hat{a}_{opt} = v_0^2/4$')
    ax1.legend(loc='upper left')
    
    cbar = plt.colorbar(im, ax=ax1, pad=0.12)
    cbar.set_label('OTF Amplitude [a.u.]')
    
    ax1.set_title('OTF Side-Peak Amplitude Heatmap')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# FIGURE 7: METHOD COMPARISON (DETECTION VS PHOTOMETRY)
# =============================================================================

def figure_method_comparison(data: dict, save_path: Optional[str] = None):
    """
    Compare peak detection vs photometry methods.
    """
    dz = data['dz']
    v0 = data['v0']
    H_det = data['H']
    H_photo = data['H_photometry']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    extent = [v0.min(), v0.max(), dz.min(), dz.max()]
    
    # Left: Detection
    ax = axes[0]
    valid = np.isfinite(H_det)
    vmin, vmax = np.nanpercentile(H_det[valid], [2, 98])
    im = ax.imshow(H_det, aspect='auto', origin='lower', extent=extent,
                   cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, pad=0.02)
    ax.set_xlabel(r'$v_0$ [cycles/aperture]')
    ax.set_ylabel(r'$\Delta z$ [mm]')
    ax.set_title('(a) Peak Detection')
    
    # Middle: Photometry
    ax = axes[1]
    valid = np.isfinite(H_photo)
    vmin, vmax = np.nanpercentile(H_photo[valid], [2, 98])
    im = ax.imshow(H_photo, aspect='auto', origin='lower', extent=extent,
                   cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, pad=0.02)
    ax.set_xlabel(r'$v_0$ [cycles/aperture]')
    ax.set_ylabel(r'$\Delta z$ [mm]')
    ax.set_title('(b) Aperture Photometry')
    
    # Right: Correlation
    ax = axes[2]
    valid = np.isfinite(H_det) & np.isfinite(H_photo) & (H_det > 0) & (H_photo > 0)
    ax.scatter(H_det[valid], H_photo[valid], s=2, alpha=0.3, c=COLORS['blue'])
    
    # 1:1 line
    max_val = max(np.nanmax(H_det), np.nanmax(H_photo))
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='1:1')
    
    # Correlation coefficient
    corr = np.corrcoef(H_det[valid].flatten(), H_photo[valid].flatten())[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=11, va='top')
    
    ax.set_xlabel('Peak Detection Amplitude')
    ax.set_ylabel('Photometry Amplitude')
    ax.set_title('(c) Method Correlation')
    ax.legend(loc='lower right')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# GENERATE ALL FIGURES
# =============================================================================

def generate_all_paper_figures(npz_path: str, output_dir: str = "./FDPRNotebooks",
                                fmt: str = "png"):
    """Generate all paper figures from the NPZ data file."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING PAPER FIGURES FROM SIMULATION DATA")
    print("=" * 60)
    print(f"Input: {npz_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    data = load_otf_data(npz_path)
    print(f"Loaded: dz range [{data['dz'].min():.0f}, {data['dz'].max():.0f}] mm")
    print(f"        v0 range [{data['v0'].min():.0f}, {data['v0'].max():.0f}] cyc/ap")
    print(f"        m_waves = {data['m_waves']}")
    print()
    
    # Generate figures
    print("[1/7] OTF Heatmap (main result)...")
    figure_otf_heatmap(data, f"{output_dir}/fig1_otf_heatmap.{fmt}")
    
    print("\n[2/7] OTF slices at fixed defocus...")
    figure_otf_slices_fixed_dz(data, save_path=f"{output_dir}/fig2_slices_fixed_dz.{fmt}")
    
    print("\n[3/7] OTF slices at fixed frequency...")
    figure_otf_slices_fixed_v0(data, save_path=f"{output_dir}/fig3_slices_fixed_v0.{fmt}")
    
    print("\n[4/7] Optimal defocus vs frequency...")
    figure_optimal_defocus(data, save_path=f"{output_dir}/fig4_optimal_defocus.{fmt}")
    
    print("\n[5/7] Dean & Bowers theory comparison...")
    figure_dean_bowers_theory(data, v0_select=10, save_path=f"{output_dir}/fig5_dean_bowers.{fmt}")
    
    print("\n[6/7] Dual-axis heatmap...")
    figure_heatmap_dual_axis(data, save_path=f"{output_dir}/fig6_heatmap_dual_axis.{fmt}")
    
    print("\n[7/7] Method comparison...")
    figure_method_comparison(data, save_path=f"{output_dir}/fig7_method_comparison.{fmt}")
    
    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Path to your NPZ file
    npz_path = "/Users/joshuapotter/Documents/SEAL/FDPRNotebooks/OTF_heatmap_data.npz"
    
    # Generate all figures
    generate_all_paper_figures(npz_path, output_dir="./FDPRNotebooks", fmt="png")