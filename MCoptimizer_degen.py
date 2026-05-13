# Monte Carlo Defocus Optimizer
# Finds optimal defocus for wavefront sensing via SNR maximization

import numpy as np
from hcipy import *
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for Monte Carlo optimization."""
    # Defocus search range (mm)
    dz_min: float = 5.0
    dz_max: float = 250.0
    
    # Monte Carlo parameters
    n_coarse_samples: int = 50      # Initial random samples
    n_refinement_samples: int = 30  # Samples in refinement phase
    refinement_width: float = 0.2   # Fraction of range to refine around best
    n_refinement_iterations: int = 3
    
    # Aberration model: 'sinusoid' or 'random_zernike'
    aberration_mode: str = 'sinusoid'
    
    # Sinusoid parameters (when mode='sinusoid')
    v0_single: float = 20.0         # cycles/aperture
    m_waves_single: float = 0.10    # peak amplitude in waves
    
    # Random Zernike parameters (when mode='random_zernike')
    n_zernike_modes: int = 20       # Number of Zernike modes to include (starting from mode 4)
    zernike_rms_waves: float = 0.15 # Total RMS wavefront error in waves
    zernike_power_law: float = -2.0 # Power law for mode amplitude falloff
    
    # Noise model
    photon_flux: float = 1e6        # Total photons in PSF
    read_noise_e: float = 5.0       # Read noise in electrons
    dark_current_e: float = 0.1     # Dark current per pixel
    
    # Spatial frequency range for SNR integration
    v0_min: float = 5.0             # Min frequency to consider
    v0_max: float = 60.0            # Max frequency to consider
    n_freq_samples: int = 20        # Number of frequencies to sample
    
    # Monte Carlo noise realizations
    n_noise_realizations: int = 10  # Noise realizations per defocus evaluation
    
    # Reproducibility
    seed: Optional[int] = 42


@dataclass 
class OpticalConfig:
    """Optical system parameters."""
    wavelength_m: float = 650e-9
    pupil_diameter_m: float = 10.12e-3
    focal_length_m: float = 500e-3
    pupil_npix: int = 256
    focal_q: int = 4
    focal_num_airy: int = 64


# =============================================================================
# HELPER FUNCTIONS (from original script)
# =============================================================================

def mm_to_m(x_mm: float) -> float:
    return x_mm * 1e-3


def delta_to_p(delta: float, f: float, D: float) -> float:
    """Convert mechanical defocus to P-V wavefront error."""
    return -1 * delta / (8 * (f / D) ** 2)


def make_sinusoidal_phase_waves(pupil_grid, pupil_diameter_m: float, 
                                 cycles_per_aperture: float, m_waves: float) -> Field:
    """Create sinusoidal phase aberration along x-axis."""
    x = pupil_grid.x
    D = pupil_diameter_m
    phase_waves = m_waves * np.sin(2 * np.pi * cycles_per_aperture * (x / D))
    return Field(phase_waves, pupil_grid)


def make_random_zernike_aberration(zernike_basis, n_modes: int, rms_waves: float,
                                    power_law: float, rng: np.random.Generator) -> Field:
    """
    Create random wavefront from Zernike modes.
    
    Parameters
    ----------
    zernike_basis : ModeBasis
        HCIPy Zernike basis (should have at least n_modes + 4 modes)
    n_modes : int
        Number of modes to include (starting from mode index 4, i.e., excluding piston/tip/tilt/defocus)
    rms_waves : float
        Target RMS wavefront error in waves
    power_law : float
        Power law exponent for mode amplitude (e.g., -2 for Kolmogorov-like)
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    Field
        Phase aberration in waves
    """
    # Generate random coefficients with power-law falloff
    mode_indices = np.arange(4, 4 + n_modes)  # Skip piston, tip, tilt, defocus
    amplitudes = (mode_indices ** power_law)
    amplitudes /= np.sqrt(np.sum(amplitudes ** 2))  # Normalize
    
    # Random signs and small variations
    coeffs = amplitudes * rng.standard_normal(n_modes)
    
    # Scale to target RMS
    coeffs *= rms_waves / np.sqrt(np.sum(coeffs ** 2))
    
    # Build wavefront
    phase = np.zeros_like(zernike_basis[0])
    for i, c in enumerate(coeffs):
        phase += c * zernike_basis[4 + i]
    
    return phase


# =============================================================================
# OPTICAL SIMULATION ENGINE
# =============================================================================

class OpticalSimulator:
    """Handles PSF/OTF computation for the optical system."""
    
    def __init__(self, optical_config: OpticalConfig):
        self.config = optical_config
        self._setup_grids()
        self._setup_propagator()
        self._setup_zernike()
    
    def _setup_grids(self):
        """Initialize pupil and focal plane grids."""
        cfg = self.config
        self.pupil_grid = make_pupil_grid(cfg.pupil_npix, cfg.pupil_diameter_m)
        self.focal_grid = make_focal_grid(
            q=cfg.focal_q,
            num_airy=cfg.focal_num_airy,
            pupil_diameter=cfg.pupil_diameter_m,
            focal_length=cfg.focal_length_m,
            reference_wavelength=cfg.wavelength_m
        )
        aperture = make_circular_aperture(cfg.pupil_diameter_m)
        self.telescope_pupil = aperture(self.pupil_grid)
        self.pupil_mask = np.array(self.telescope_pupil.shaped, dtype=bool)
    
    def _setup_propagator(self):
        """Initialize Fraunhofer propagator."""
        cfg = self.config
        self.propagator = FraunhoferPropagator(
            self.pupil_grid, self.focal_grid, cfg.focal_length_m
        )
    
    def _setup_zernike(self):
        """Initialize Zernike basis for defocus and aberrations."""
        cfg = self.config
        self.zernike_basis = make_zernike_basis(
            num_modes=256,
            D=cfg.pupil_diameter_m,
            grid=self.pupil_grid
        )
        self.defocus_template = self.zernike_basis[3]
        
        # Precompute unit defocus (normalized to 1 wave P-V)
        defocus_shaped = self.defocus_template.shaped
        template_pv = defocus_shaped[self.pupil_mask].max() - defocus_shaped[self.pupil_mask].min()
        self.unit_defocus = self.defocus_template / template_pv
    
    def compute_defocus_phase(self, dz_mm: float) -> Field:
        """
        Compute defocus phase from mechanical distance.
        
        Parameters
        ----------
        dz_mm : float
            Defocus distance in mm
        
        Returns
        -------
        Field
            Defocus phase in radians
        """
        cfg = self.config
        dz_m = mm_to_m(dz_mm)
        defocus_pv_waves = delta_to_p(dz_m, cfg.focal_length_m, cfg.pupil_diameter_m) / cfg.wavelength_m
        phase_pv_rad = defocus_pv_waves * 2 * np.pi
        return self.unit_defocus * phase_pv_rad
    
    def compute_psf(self, aberration_rad: Field, defocus_rad: Optional[Field] = None) -> np.ndarray:
        """
        Compute PSF intensity from aberration.
        
        Parameters
        ----------
        aberration_rad : Field
            Aberration phase in radians
        defocus_rad : Field, optional
            Additional defocus phase in radians
        
        Returns
        -------
        np.ndarray
            PSF intensity (normalized to sum=1)
        """
        total_phase = aberration_rad
        if defocus_rad is not None:
            total_phase = total_phase + defocus_rad
        
        wf = Wavefront(self.telescope_pupil * np.exp(1j * total_phase), self.config.wavelength_m)
        psf = self.propagator(wf).intensity.shaped
        psf = np.asarray(psf)
        psf /= psf.sum()  # Normalize
        return psf
    
    def compute_otf(self, psf: np.ndarray) -> np.ndarray:
        """Compute OTF magnitude from PSF."""
        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        return np.abs(otf)
    
    def find_otf_sidepeaks(self, otf: np.ndarray, kill_core_pix: int = 9,
                           subpixel: bool = True) -> Tuple[float, float]:
        """
        Find side-peak amplitude and offset in OTF.
        
        Returns
        -------
        offset : float
            Average pixel distance from DC
        amplitude : float
            Average side-peak amplitude
        """
        row = otf[otf.shape[0] // 2, :].copy()
        c = len(row) // 2
        row[c - kill_core_pix:c + kill_core_pix + 1] = 0.0
        
        left, right = row[:c], row[c + 1:]
        il, ir = int(np.argmax(left)), int(np.argmax(right))
        amp_l, amp_r = float(left[il]), float(right[ir])
        
        def refine(i, a):
            if not subpixel or i <= 0 or i >= len(a) - 1:
                return float(i)
            y0, y1, y2 = a[i - 1], a[i], a[i + 1]
            d = y0 - 2 * y1 + y2
            return i + 0.5 * (y0 - y2) / d if d != 0 else float(i)
        
        pos_l = refine(il, left)
        pos_r = refine(ir, right) + c + 1
        off_l = -(c - pos_l)
        off_r = pos_r - c
        
        offset = 0.5 * (abs(off_l) + abs(off_r))
        amplitude = 0.5 * (amp_l + amp_r)
        
        return offset, amplitude


# =============================================================================
# NOISE MODEL
# =============================================================================

class NoiseModel:
    """Detector noise model for SNR computation."""
    
    def __init__(self, photon_flux: float, read_noise_e: float, dark_current_e: float):
        self.photon_flux = photon_flux
        self.read_noise = read_noise_e
        self.dark_current = dark_current_e
    
    def add_noise(self, psf: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Add realistic detector noise to PSF.
        
        Parameters
        ----------
        psf : np.ndarray
            Normalized PSF (sum=1)
        rng : np.random.Generator
            Random number generator
        
        Returns
        -------
        np.ndarray
            Noisy PSF in electrons
        """
        # Convert to photons
        signal_e = psf * self.photon_flux
        
        # Shot noise (Poisson)
        noisy = rng.poisson(signal_e).astype(float)
        
        # Dark current
        noisy += rng.poisson(self.dark_current, size=psf.shape)
        
        # Read noise (Gaussian)
        noisy += rng.normal(0, self.read_noise, size=psf.shape)
        
        return noisy
    
    def estimate_snr(self, psf: np.ndarray, otf_amplitude: float) -> float:
        """
        Estimate SNR for OTF side-peak measurement.
        
        This is a simplified model: SNR ≈ signal / sqrt(noise_variance)
        where signal is the OTF amplitude and noise comes from PSF noise
        propagated through the FFT.
        
        Parameters
        ----------
        psf : np.ndarray
            Normalized PSF
        otf_amplitude : float
            OTF side-peak amplitude
        
        Returns
        -------
        float
            Estimated SNR
        """
        # Signal in electrons
        signal_e = psf * self.photon_flux
        
        # Per-pixel variance
        var_shot = signal_e  # Poisson variance = mean
        var_dark = self.dark_current
        var_read = self.read_noise ** 2
        total_var = var_shot + var_dark + var_read
        
        # Noise in OTF (approximately)
        # FFT preserves total power, so noise spreads across OTF
        n_pix = psf.size
        otf_noise_var = np.sum(total_var) / n_pix
        otf_noise = np.sqrt(otf_noise_var)
        
        # SNR
        snr = otf_amplitude / otf_noise if otf_noise > 0 else 0.0
        return snr


# =============================================================================
# SNR OBJECTIVE FUNCTION
# =============================================================================

class SNRObjective:
    """
    Computes integrated SNR over spatial frequencies.
    
    This is the objective function we want to maximize.
    """
    
    def __init__(self, simulator: OpticalSimulator, noise_model: NoiseModel,
                 config: OptimizationConfig):
        self.sim = simulator
        self.noise = noise_model
        self.config = config
        
        # Precompute frequency samples
        self.v0_samples = np.linspace(config.v0_min, config.v0_max, config.n_freq_samples)
        
        # Random number generator
        self.rng = np.random.default_rng(config.seed)
    
    def _generate_aberration(self) -> Field:
        """Generate aberration based on config mode."""
        cfg = self.config
        
        if cfg.aberration_mode == 'sinusoid':
            phi_waves = make_sinusoidal_phase_waves(
                self.sim.pupil_grid,
                self.sim.config.pupil_diameter_m,
                cfg.v0_single,
                cfg.m_waves_single
            )
            return phi_waves * 2 * np.pi  # Convert to radians
        
        elif cfg.aberration_mode == 'random_zernike':
            phi_waves = make_random_zernike_aberration(
                self.sim.zernike_basis,
                cfg.n_zernike_modes,
                cfg.zernike_rms_waves,
                cfg.zernike_power_law,
                self.rng
            )
            return phi_waves * 2 * np.pi  # Convert to radians
        
        else:
            raise ValueError(f"Unknown aberration mode: {cfg.aberration_mode}")
    
    def evaluate(self, dz_mm: float, verbose: bool = False) -> float:
        """
        Evaluate objective function (integrated SNR) at given defocus.
        
        Parameters
        ----------
        dz_mm : float
            Defocus distance in mm
        verbose : bool
            Print debug info
        
        Returns
        -------
        float
            Integrated SNR (higher is better)
        """
        cfg = self.config
        defocus_rad = self.sim.compute_defocus_phase(dz_mm)
        
        snr_values = []
        
        if cfg.aberration_mode == 'sinusoid':
            # For single sinusoid, evaluate at that frequency
            aberration_rad = self._generate_aberration()
            
            for _ in range(cfg.n_noise_realizations):
                psf = self.sim.compute_psf(aberration_rad, defocus_rad)
                noisy_psf = self.noise.add_noise(psf, self.rng)
                noisy_psf = np.maximum(noisy_psf, 0)  # Clip negatives
                noisy_psf /= noisy_psf.sum()  # Renormalize
                
                otf = self.sim.compute_otf(noisy_psf)
                _, amplitude = self.sim.find_otf_sidepeaks(otf)
                snr = self.noise.estimate_snr(psf, amplitude)
                snr_values.append(snr)
        
        elif cfg.aberration_mode == 'random_zernike':
            # For random Zernike, we measure OTF quality differently
            # We look at the OTF envelope / MTF across frequencies
            aberration_rad = self._generate_aberration()
            
            for _ in range(cfg.n_noise_realizations):
                psf = self.sim.compute_psf(aberration_rad, defocus_rad)
                noisy_psf = self.noise.add_noise(psf, self.rng)
                noisy_psf = np.maximum(noisy_psf, 0)
                noisy_psf /= noisy_psf.sum()
                
                otf = self.sim.compute_otf(noisy_psf)
                
                # Sample OTF along radial direction (average azimuthally)
                center = np.array(otf.shape) // 2
                y, x = np.ogrid[:otf.shape[0], :otf.shape[1]]
                r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
                
                # Bin by radius and compute mean OTF
                r_max = min(center)
                n_bins = 50
                r_bins = np.linspace(0, r_max, n_bins + 1)
                otf_radial = np.zeros(n_bins)
                
                for i in range(n_bins):
                    mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
                    if np.any(mask):
                        otf_radial[i] = np.mean(otf[mask])
                
                # Compute integrated SNR (OTF amplitude / noise floor)
                # Skip DC peak
                otf_signal = otf_radial[5:].mean()
                snr = self.noise.estimate_snr(psf, otf_signal)
                snr_values.append(snr)
        
        mean_snr = np.mean(snr_values)
        
        if verbose:
            print(f"  dz={dz_mm:.1f} mm: SNR={mean_snr:.2f} (std={np.std(snr_values):.2f})")
        
        return mean_snr


# =============================================================================
# MONTE CARLO OPTIMIZER
# =============================================================================

class MonteCarloOptimizer:
    """
    Monte Carlo optimization with iterative refinement.
    
    Algorithm:
    1. Coarse phase: Random sampling across full search range
    2. Refinement phase: Focus sampling around best points
    3. Iterate refinement with shrinking search window
    """
    
    def __init__(self, objective: SNRObjective, config: OptimizationConfig):
        self.objective = objective
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        # Storage for results
        self.history = {
            'dz': [],
            'snr': [],
            'phase': []
        }
    
    def optimize(self, verbose: bool = True) -> Tuple[float, float]:
        """
        Run Monte Carlo optimization.
        
        Returns
        -------
        best_dz : float
            Optimal defocus in mm
        best_snr : float
            SNR at optimal defocus
        """
        cfg = self.config
        
        if verbose:
            print("=" * 60)
            print("Monte Carlo Defocus Optimization")
            print(f"  Search range: {cfg.dz_min:.1f} - {cfg.dz_max:.1f} mm")
            print(f"  Aberration mode: {cfg.aberration_mode}")
            print(f"  Coarse samples: {cfg.n_coarse_samples}")
            print(f"  Refinement iterations: {cfg.n_refinement_iterations}")
            print("=" * 60)
        
        # Phase 1: Coarse sampling
        if verbose:
            print("\nPhase 1: Coarse sampling...")
        
        dz_samples = self.rng.uniform(cfg.dz_min, cfg.dz_max, cfg.n_coarse_samples)
        
        t0 = time.time()
        for i, dz in enumerate(dz_samples):
            snr = self.objective.evaluate(dz, verbose=False)
            self.history['dz'].append(dz)
            self.history['snr'].append(snr)
            self.history['phase'].append('coarse')
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{cfg.n_coarse_samples}")
        
        if verbose:
            print(f"  Coarse phase: {time.time() - t0:.1f}s")
        
        # Find current best
        best_idx = np.argmax(self.history['snr'])
        best_dz = self.history['dz'][best_idx]
        best_snr = self.history['snr'][best_idx]
        
        if verbose:
            print(f"  Current best: dz={best_dz:.1f} mm, SNR={best_snr:.2f}")
        
        # Phase 2: Iterative refinement
        search_width = (cfg.dz_max - cfg.dz_min) * cfg.refinement_width
        
        for iteration in range(cfg.n_refinement_iterations):
            if verbose:
                print(f"\nPhase 2.{iteration + 1}: Refinement (width={search_width:.1f} mm)...")
            
            # Sample around best point
            dz_lo = max(cfg.dz_min, best_dz - search_width / 2)
            dz_hi = min(cfg.dz_max, best_dz + search_width / 2)
            dz_samples = self.rng.uniform(dz_lo, dz_hi, cfg.n_refinement_samples)
            
            t0 = time.time()
            for dz in dz_samples:
                snr = self.objective.evaluate(dz, verbose=False)
                self.history['dz'].append(dz)
                self.history['snr'].append(snr)
                self.history['phase'].append(f'refine_{iteration + 1}')
            
            if verbose:
                print(f"  Refinement {iteration + 1}: {time.time() - t0:.1f}s")
            
            # Update best
            best_idx = np.argmax(self.history['snr'])
            best_dz = self.history['dz'][best_idx]
            best_snr = self.history['snr'][best_idx]
            
            if verbose:
                print(f"  Current best: dz={best_dz:.1f} mm, SNR={best_snr:.2f}")
            
            # Shrink search window
            search_width *= 0.5
        
        return best_dz, best_snr
    
    def get_results_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (dz_array, snr_array) from history."""
        return np.array(self.history['dz']), np.array(self.history['snr'])
    
    def plot_optimization(self, save_path: Optional[str] = None):
        """Plot optimization history and results."""
        dz = np.array(self.history['dz'])
        snr = np.array(self.history['snr'])
        phases = np.array(self.history['phase'])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: All samples colored by phase
        ax = axes[0]
        colors = {'coarse': 'blue', 'refine_1': 'orange', 'refine_2': 'green', 'refine_3': 'red'}
        for phase in np.unique(phases):
            mask = phases == phase
            c = colors.get(phase, 'gray')
            ax.scatter(dz[mask], snr[mask], c=c, alpha=0.6, label=phase, s=20)
        ax.set_xlabel('Defocus (mm)')
        ax.set_ylabel('SNR')
        ax.set_title('Optimization Samples')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Convergence of best SNR
        ax = axes[1]
        best_so_far = np.maximum.accumulate(snr)
        ax.plot(best_so_far, 'b-', linewidth=2)
        ax.set_xlabel('Evaluation Number')
        ax.set_ylabel('Best SNR')
        ax.set_title('Convergence')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Histogram of samples
        ax = axes[2]
        ax.hist(dz, bins=30, alpha=0.7, edgecolor='black')
        best_idx = np.argmax(snr)
        ax.axvline(dz[best_idx], color='red', linestyle='--', linewidth=2, label=f'Best: {dz[best_idx]:.1f} mm')
        ax.set_xlabel('Defocus (mm)')
        ax.set_ylabel('Count')
        ax.set_title('Sample Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
        return fig


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_snr_vs_defocus_curve(simulator: OpticalSimulator, noise_model: NoiseModel,
                              config: OptimizationConfig, dz_range: np.ndarray,
                              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SNR vs defocus curve with error bars.
    
    Returns
    -------
    dz_array : np.ndarray
        Defocus values
    snr_mean : np.ndarray
        Mean SNR at each defocus
    snr_std : np.ndarray
        Standard deviation of SNR
    """
    objective = SNRObjective(simulator, noise_model, config)
    
    snr_mean = np.zeros(len(dz_range))
    snr_std = np.zeros(len(dz_range))
    
    if verbose:
        print(f"Computing SNR curve for {len(dz_range)} defocus values...")
    
    for i, dz in enumerate(dz_range):
        # Run multiple evaluations for statistics
        snrs = [objective.evaluate(dz) for _ in range(5)]
        snr_mean[i] = np.mean(snrs)
        snr_std[i] = np.std(snrs)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(dz_range)}")
    
    return dz_range, snr_mean, snr_std


def compare_aberration_modes(optical_config: OpticalConfig, 
                              dz_range: np.ndarray,
                              save_path: Optional[str] = None):
    """
    Compare optimization results for sinusoid vs random Zernike modes.
    """
    simulator = OpticalSimulator(optical_config)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, mode in enumerate(['sinusoid', 'random_zernike']):
        config = OptimizationConfig(aberration_mode=mode, seed=42)
        noise_model = NoiseModel(config.photon_flux, config.read_noise_e, config.dark_current_e)
        
        dz, snr_mean, snr_std = run_snr_vs_defocus_curve(
            simulator, noise_model, config, dz_range, verbose=True
        )
        
        ax = axes[idx]
        ax.fill_between(dz, snr_mean - snr_std, snr_mean + snr_std, alpha=0.3)
        ax.plot(dz, snr_mean, 'b-', linewidth=2)
        ax.axvline(dz[np.argmax(snr_mean)], color='red', linestyle='--', 
                   label=f'Optimal: {dz[np.argmax(snr_mean)]:.1f} mm')
        ax.set_xlabel('Defocus (mm)')
        ax.set_ylabel('SNR')
        ax.set_title(f'Aberration Mode: {mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MONTE CARLO DEFOCUS OPTIMIZER")
    print("=" * 70)
    
    # Initialize optical system
    optical_config = OpticalConfig()
    simulator = OpticalSimulator(optical_config)
    
    # Run optimization for both modes
    for aberration_mode in ['sinusoid', 'random_zernike']:
        print(f"\n{'=' * 70}")
        print(f"ABERRATION MODE: {aberration_mode.upper()}")
        print(f"{'=' * 70}")
        
        # Configure optimization
        opt_config = OptimizationConfig(
            aberration_mode=aberration_mode,
            n_coarse_samples=50,
            n_refinement_samples=30,
            n_refinement_iterations=3,
            n_noise_realizations=10,
            seed=42
        )
        
        # Initialize noise model
        noise_model = NoiseModel(
            opt_config.photon_flux,
            opt_config.read_noise_e,
            opt_config.dark_current_e
        )
        
        # Create objective and optimizer
        objective = SNRObjective(simulator, noise_model, opt_config)
        optimizer = MonteCarloOptimizer(objective, opt_config)
        
        # Run optimization
        best_dz, best_snr = optimizer.optimize(verbose=True)
        
        print(f"\n{'=' * 40}")
        print(f"RESULT: Optimal defocus = {best_dz:.2f} mm")
        print(f"        SNR at optimum = {best_snr:.2f}")
        print(f"{'=' * 40}")
        
        # Plot results
        optimizer.plot_optimization(save_path=f"mc_optimization_{aberration_mode}.png")
    
    # Optional: Run detailed SNR curve
    print("\nComputing detailed SNR vs defocus curves...")
    dz_range = np.linspace(5, 250, 50)
    compare_aberration_modes(optical_config, dz_range, save_path="snr_comparison.png")
    
    print("\nOptimization complete!")
