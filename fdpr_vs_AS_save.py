'''
Imports
'''
import time
import numpy as np
from skimage.transform import resize
from hcipy import *
from image_sharpening import FocusDiversePhaseRetrieval,mft_rev, InstrumentConfiguration
import matplotlib.pyplot as plt


'''
TODO:

Check in with EMiel about creating new fourier transform function post mp.step()


'''

'''
Configuration flags
'''
# Verbose: plot first and last trial for EVERY grid point
verbose = True

# Spot check: detailed 4x4 plot at a SINGLE (dz, v0) point (finds nearest)
do_spot_check = False

#if we want noise or if we want clean
if_noise=False

snapshot=True

'''
Toggle to decide which model to use for propagation
I found that since image_sharpening internally calls angular spectrum, it fucks up mft_rev
Necessary to run them respectively:
    pupil_zernike & mft_rev
    image_AS & hcipy_backward
    my_fdpr & fft
'''
DEFOCUS_MODEL = "pupil_zernike"   # you can choose: "pupil_zernike" OR "image_AS" OR "my_fdpr"
PUPIL_READOUT = "mft_rev"         # you can choose: "mft_rev" OR "hcipy_backward" OR "fft"

try:
    from image_sharpening import _angular_spectrum_transfer_function, _angular_spectrum_prop
    _HAS_AS = True
except Exception:
    _HAS_AS = False
'''
FDPR CLASS
'''
class SimpleFDPR:
    """
    Simple Focus-Diverse Phase Retrieval using Gerchberg-Saxton iteration.
    Uses FFT for pupil<->focal transforms (simpler than angular spectrum).
    """
        
    def __init__(self, psf_focused, defocus_list=None):
        """
        Parameters
           -
        psf_focused : numpy.ndarray
            Focused PSF (intensity image, i.e. what the camera sees)
        defocus_list : list of tuples, optional
            List of (psf_defocused, defocus_phase) tuples
            Can also add later with add_defocused_image()
        """
        # Camera measures intensity = |E|^2, but we need amplitude = |E|
        # so take sqrt to get the amplitude constraint
        self.amp_focused = np.sqrt(psf_focused)
        
        # Will hold all our defocused measurements
        # Each entry stores: amplitude, known defocus phase, and cost history
        self.defocused_data = []
        self.cost_functions = [[]]  # for compatibility with original FDPR class
        
        # add defocused images
        if defocus_list is not None:
            for psf_defoc, defoc_phase in defocus_list:
                self.add_defocused_image(psf_defoc, defoc_phase)
        
        self.phase_estimate = np.random.rand(*psf_focused.shape) * 2 * np.pi
        
        self.iter = 0
        self.cost_history = []
    
    def add_defocused_image(self, psf_defocused, defocus_phase):
        """
        Add a defocused image to the retrieval.
        
        More images = more constraints = better reconstruction.
        Sweet spot is usually 1-2 waves of defocus.
        
        Parameters
           -
        psf_defocused : numpy.ndarray
            Defocused PSF (intensity image)
        defocus_phase : numpy.ndarray  
            The known defocus phase in radians - this is what makes it "diverse"
            Needs to be same shape as PSF (gets applied after FFT to pupil)
        """
        amp_defocused = np.sqrt(psf_defocused)
        self.defocused_data.append({
            'amplitude': amp_defocused,
            'defocus_phase': defocus_phase,
            'cost': []
        })
        print(f"Added defocused image. Total: {len(self.defocused_data)} defocused images.")
    
    def _focal_to_pupil(self, focal_field):
        """
        FFT from focal plane back to pupil plane.
        
        Optics refresher: pupil and focal planes are Fourier pairs.
        The PSF is literally the Fourier transform of the pupil function.
        """
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(focal_field)))
    
    def _pupil_to_focal(self, pupil_field):
        """FFT from pupil plane to focal plane."""
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_field)))
    
    def _compute_mse(self, estimate, target):
        """How wrong are we? Want this to go down over iterations."""
        return np.mean((estimate - target)**2)
    
    def step(self):
        """
        One iteration of Gerchberg-Saxton with focus diversity.
        
        The core loop:
        1. Current focal estimate -> pupil (via FFT)
        2. Add defocus -> defocused focal plane
        3. FORCE amplitude to match measurement, keep phase
        4. Back to pupil -> remove defocus -> focused focal
        5. FORCE amplitude to match measurement, keep phase
        6. Repeat for each defocused image
        
        The phase evolves freely while we keep enforcing amplitude constraints.
        Eventually it converges to something consistent with ALL measurements.
        
        Returns
          -
        psf_intensity : numpy.ndarray
            Current PSF estimate (real-valued intensity, not complex field)
        """
        if len(self.defocused_data) == 0:
            raise ValueError("No defocused images added. Use add_defocused_image() first.")
        
        # Build our current best guess of the focal plane E-field
        # E = amplitude * exp(i * phase)
        focal_field = self.amp_focused * np.exp(1j * self.phase_estimate)
        
        # Process each defocused image - more images = tighter constraints
        for data in self.defocused_data:
            amp_defocused = data['amplitude']
            defocus_phase = data['defocus_phase']
            
            #Go to pupil plane  
            pupil_field = self._focal_to_pupil(focal_field)
            pupil_phase = np.angle(pupil_field)
            pupil_amp = np.abs(pupil_field)
            
            # Add defocus and go to defocused focal plane  
            pupil_defocused = pupil_amp * np.exp(1j * (pupil_phase + defocus_phase))
            focal_defocused = self._pupil_to_focal(pupil_defocused)
            
            # Track how well we match the measurement
            cost = self._compute_mse(np.abs(focal_defocused), amp_defocused)
            data['cost'].append(cost)
            
            # enforce amplitude constraint  
            # know the amplitude, so force it
            # DON'T know phase, so keep our current estimate
            focal_defocused_constrained = amp_defocused * np.exp(1j * np.angle(focal_defocused))
            
            #Back to pupil, remove the defocus added  
            pupil_field_2 = self._focal_to_pupil(focal_defocused_constrained)
            pupil_focused = np.abs(pupil_field_2) * np.exp(1j * (np.angle(pupil_field_2) - defocus_phase))
            
            #Back to focused focal plane 
            focal_field = self._pupil_to_focal(pupil_focused)
            
            #Enforce focused amplitude constraint  
            focal_field = self.amp_focused * np.exp(1j * np.angle(focal_field))
        
        # Save the updated phase estimate for next iteration
        self.phase_estimate = np.angle(focal_field)
        self.iter += 1
        
        # Return intensity (real), not complex field
        return np.abs(focal_field)**2
    
    def run(self, n_iterations=100, verbose=True):
        """
        Run multiple iterations. Usually converges in 50-200 iterations.
        
        """
        for i in range(n_iterations):
            focal_field = self.step()
            if verbose and (i+1) % 20 == 0:
                total_cost = sum(data['cost'][-1] for data in self.defocused_data)
                print(f"Iteration {i+1}/{n_iterations}, Total cost: {total_cost:.2e}")
        
        return focal_field
    
    def get_pupil_phase(self):
        """
        Get the reconstructed pupil phase, THE MAIN OUTPUT.
        
        This is the aberration/wavefront error
        Units are radians. To get OPD in meters: phase * wavelength / (2*pi)
        """
        focal_field = self.amp_focused * np.exp(1j * self.phase_estimate)
        pupil_field = self._focal_to_pupil(focal_field)
        return np.angle(pupil_field)
    
    def get_pupil_field(self):
        """Get full complex pupil field (amplitude shows aperture, phase shows aberration)"""
        focal_field = self.amp_focused * np.exp(1j * self.phase_estimate)
        return self._focal_to_pupil(focal_field)
    
    def plot_cost(self):
        """
        Plot convergence. Good sign: drops several orders of magnitude then plateaus.
        Bad sign: oscillates, goes up, or barely moves.
        """
        plt.figure(figsize=(8, 5))
        for i, data in enumerate(self.defocused_data):
            plt.semilogy(data['cost'], label=f'Defocus {i+1}')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('Cost Function History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

'''
Helper Functions

'''
def plot_cost_convergence(mp):
    plt.figure(figsize=(6,4))
    if hasattr(mp, 'defocused_data'):
        # SimpleFDPR
        for i, data in enumerate(mp.defocused_data):
            if len(data['cost']) > 0:
                plt.plot(data['cost'], label=f"Defocus {i+1}")
    else:
        # Original FDPR
        for i, cf in enumerate(mp.cost_functions):
            if len(cf) > 0:
                plt.plot(cf, label=f"PSF {i+1}")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Cost")
    plt.legend()
    plt.title("FDPR Cost Function Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pupil_convergence(snapshot_data, conf, telescope_pupil, use_fft=False):
    iters = sorted(snapshot_data.keys())
    n = len(iters)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))

    for ax, it in zip(axes, iters):
        Erec = snapshot_data[it]["psf_rec_complex"]

        if use_fft:
            # For SimpleFDPR - use FFT
            pupil_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Erec)))
            raw_pupil = np.angle(pupil_field)
        else:
            # For original FDPR - use mft_rev
            raw_pupil = np.angle(mft_rev(Erec, conf))
        
        pupil = resize(raw_pupil, (256,256), preserve_range=True) * telescope_pupil.shaped

        ax.imshow(pupil, cmap="RdBu_r")
        ax.set_title(f"Iter {it}")
        ax.axis("off")

    fig.suptitle("FDPR Pupil Phase Convergence", fontsize=14)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.real(snapshot_data[it]["psf_rec_complex"]))
    plt.title("Real part")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(np.imag(snapshot_data[it]["psf_rec_complex"]))
    plt.title("Imag part")
    plt.colorbar()
    plt.show()




def plot_psf_convergence(snapshot_data):
    iters = sorted(snapshot_data.keys())
    n = len(iters)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))

    for ax, it in zip(axes, iters):
        psf = snapshot_data[it]["psf_rec"]
        ax.imshow(np.log10(np.abs(psf) + 1e-10), cmap="inferno", vmin=-6)
        ax.set_title(f"Iter {it}")
        ax.axis("off")

    fig.suptitle("FDPR PSF Convergence (log10)", fontsize=14)
    plt.tight_layout()
    plt.show()

def mm_to_m(x_mm: float) -> float:
    return x_mm * 1e-3

def mm_to_um(x_mm: float) -> float:
    return x_mm * 1e3

def delta_to_p(delta, f, D):
    return -1 * delta / (8 * (f/D)**2)

def make_sinusoidal_phase_waves(pupil_grid, pupil_diameter_m, cycles_per_aperture, m_waves):
    """
    Single-frequency sinusoid along x. 'cycles/aperture' means cycles across diameter D.
    Returns HCIPy Field [waves] on pupil_grid.
    """
    x = pupil_grid.x
    D = pupil_diameter_m
    phase_waves = m_waves * np.sin(2 * np.pi * cycles_per_aperture * (x / D))
    return Field(phase_waves, pupil_grid)

def psf_from_wavefront(wf):
    """Returns PSF in photons/pixel."""
    I = prop_p2f(wf).power.shaped   # <-- change .intensity to .power
    return np.asarray(I)

def calculate_defocus_phase(seal_parameters, defocus_distance, telescope_pupil, defocus_template):
    """
    Calculate defocus phase from mechanical defocus distance.
    defocus_distance should be in MM.
    """
    mask = np.array(telescope_pupil.shaped, dtype=bool)
    defocus_template_s = defocus_template.shaped
    template_p2v = defocus_template_s[mask].max() - defocus_template_s[mask].min()
    unit_defocus = defocus_template / template_p2v
    dz_m = mm_to_m(defocus_distance)
    defocus_p2v = delta_to_p(
        delta=dz_m,
        f=seal_parameters['focal_length_meters'],
        D=seal_parameters['pupil_size']
    )
    phase_p2v = defocus_p2v * (2*np.pi/seal_parameters['wavelength_meter'])
    defocus_phase = unit_defocus * phase_p2v
    return defocus_phase

def add_read_noise(image_electron, sigma_e, rng):
    read_noise = rng.normal(scale=sigma_e, size=image_electron.shape)
    return image_electron + read_noise


def plot_verbose_trial(dz_mm, v0, trial_idx, phi_sine_rad, phi_def, 
                       psf_focus_clean, psf_defoc_clean, psf0, psfd,
                       psf_reconstruction, real_pupil, telescope_pupil, seal_parameters):
    """
    Verbose plotting for first/last trial at each grid point.
    Shows: injected phase, PSFs, reconstruction, and difference image.
    2x4 grid layout.
    """
    mask = np.array(telescope_pupil.shaped, dtype=bool)
    
    pupil_phase = real_pupil
    pupil_image_phase = phi_sine_rad.shaped
    
    med_subtracted = pupil_phase - np.median(pupil_phase[mask])
    difference_image = pupil_image_phase - med_subtracted
    check_error_region = difference_image[mask]
    
    median_error = np.median(check_error_region)
    rms_error = np.sqrt(np.nanmean(check_error_region**2))
    rms_error_nm = rms_error * seal_parameters['wavelength_meter'] * 1e9 / (2*np.pi)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(f'VERBOSE: dz={dz_mm:.1f} mm, v0={v0:.2f} cycles/ap, Trial {trial_idx}', fontsize=14)
    
    # Row 1: Inputs
    ax = axes[0, 0]
    im = ax.imshow(phi_sine_rad.shaped, cmap='RdBu_r')
    ax.set_title('Injected Sinusoid [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[0, 1]
    im = ax.imshow(np.log10(np.abs(psf_reconstruction) + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('PSF Reconstruction post FDPR, log10 scale')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[0, 2]
    im = ax.imshow(np.log10(psf0 + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Focused PSF (noisy, log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[0, 3]
    im = ax.imshow(np.log10(psfd + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Defocused PSF (noisy, log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    # Row 2: Reconstruction and difference
    ax = axes[1, 0]
    im = ax.imshow(pupil_phase, cmap='RdBu_r')
    ax.set_title('Reconstruction [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1, 1]
    im = ax.imshow(med_subtracted, cmap='RdBu_r')
    ax.set_title('Recon (median sub) [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1, 2]
    im = ax.imshow(pupil_image_phase, cmap='RdBu_r')
    ax.set_title('Truth [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1, 3]
    #vmax_diff = max(abs(np.nanmin(difference_image)), abs(np.nanmax(difference_image)), 0.01)
    im = ax.imshow(difference_image, cmap='RdBu_r')
    ax.set_title(f'Difference (truth - recon)\nRMS={rms_error_nm:.1f} nm')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"  [Verbose] dz={dz_mm:.1f}, v0={v0:.2f}, trial={trial_idx}: "
          f"Median error={median_error:.4f} rad, RMS={rms_error_nm:.1f} nm")


def plot_spot_check_full(dz_mm, v0, phi_sine_rad, phi_def, psf_focus_clean, psf_defoc_clean,
                         psf0, psfd, psf_reconstruction, real_pupil, 
                         telescope_pupil, seal_parameters, trial_idx=0):
    """
    Full detailed plotting for a single (dz, v0) point.
    
    trying to match the original futzing here:

        pupil_image.phase.shaped = phi_sine_rad.shaped (the truth/injected aberration)
        pupil_phase = real_pupil (the FDPR reconstruction)
    
    4x4 grid:
    Row 1: Injected sinusoid, defocus phase, combined phase, pupil mask
    Row 2: Focused PSF (clean), Defocused PSF (clean), Focused PSF (noisy), Defocused PSF (noisy)
    Row 3: FDPR output, Reconstruction, Reconstruction (median sub), Truth
    Row 4: Truth, Difference image, Central row slice, Difference histogram
    """
    mask = np.array(telescope_pupil.shaped, dtype=bool)

    # pupil_image.phase.shaped -> phi_sine_rad.shaped (truth)
    # pupil_phase -> real_pupil (reconstruction)
    
    pupil_phase = real_pupil  # reconstruction
    pupil_image_phase = phi_sine_rad.shaped  # truth (injected sinusoid)
    
    # Median subtraction on reconstruction
    med_subtracted = pupil_phase - np.median(pupil_phase[mask])
    
    # Difference image: truth - median-subtracted reconstruction
    difference_image = pupil_image_phase - med_subtracted
    
    # Error region within pupil
    check_error_region = difference_image[mask]
    
    # Stats
    median_error = np.median(check_error_region)
    rms_error = np.sqrt(np.nanmean(check_error_region**2))
    rms_error_nm = rms_error * seal_parameters['wavelength_meter'] * 1e9 / (2*np.pi)
    
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.suptitle(f'SPOT CHECK: dz={dz_mm:.1f} mm, v0={v0:.2f} cycles/ap, Trial {trial_idx}', fontsize=16)
    
    # row 1: injected phases
    ax = axes[0, 0]
    im = ax.imshow(phi_sine_rad.shaped, cmap='RdBu_r')
    ax.set_title('Injected Sinusoidal Phase [rad]\n(pupil_image.phase.shaped)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[0, 1]
    im = ax.imshow(phi_def.shaped, cmap='RdBu_r')
    ax.set_title('Defocus Phase [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[0, 2]
    combined = (phi_sine_rad + phi_def).shaped
    im = ax.imshow(combined, cmap='RdBu_r')
    ax.set_title('Combined Phase (sine + defocus) [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[0, 3]
    im = ax.imshow(telescope_pupil.shaped, cmap='gray')
    ax.set_title('Telescope Pupil Mask')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    #row 2: psfs
    ax = axes[1, 0]
    im = ax.imshow(np.log10(psf_focus_clean + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Focused PSF (clean, log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1, 1]
    im = ax.imshow(np.log10(psf_defoc_clean + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Defocused PSF (clean, log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1, 2]
    im = ax.imshow(np.log10(psf0 + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Focused PSF (noisy, log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[1, 3]
    im = ax.imshow(np.log10(psfd + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('Defocused PSF (noisy, log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    # row 3: reconstructions
    ax = axes[2, 0]
    im = ax.imshow((np.log10(psf_reconstruction) + 1e-10), vmin=-5, cmap='inferno')
    ax.set_title('FDPR reconstructed psf |psf_rec| (log10)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[2, 1]
    im = ax.imshow(pupil_phase, cmap='RdBu_r')
    ax.set_title('Reconstructed Pupil Phase [rad]\n(pupil_phase)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[2, 2]
    im = ax.imshow(med_subtracted, cmap='RdBu_r')
    ax.set_title('Reconstruction (median sub) [rad]\n(med_subtracted)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[2, 3]
    im = ax.imshow(pupil_image_phase, cmap='RdBu_r')
    ax.set_title('Truth [rad]\n(pupil_image.phase.shaped)')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    # ===== Row 4: Difference analysis =====
    ax = axes[3, 0]
    # Show truth masked by pupil
    im = ax.imshow(pupil_image_phase * telescope_pupil.shaped, cmap='RdBu_r')
    ax.set_title('Truth × Pupil Mask [rad]')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    
    ax = axes[3, 1]
    # Difference image: pupil_image.phase.shaped - med_subtracted
    vmax_diff = max(abs(np.nanmin(difference_image)), abs(np.nanmax(difference_image)), 0.01)
    im = ax.imshow(difference_image, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_title(f'DIFFERENCE IMAGE\n(truth - med_subtracted)\nRMS = {rms_error_nm:.1f} nm')
    plt.colorbar(im, ax=ax)
    ax.axis('off')

    ax = axes[3, 2]
    # Histogram of error within pupil
    valid_error = check_error_region[np.isfinite(check_error_region)]
    if len(valid_error) > 0:
        ax.hist(valid_error, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(median_error, color='orange', linestyle='-', linewidth=2, 
                   label=f'Median={median_error:.4f}')
    ax.set_xlabel('Error [rad]')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Histogram (check_error_region)\nMedian={median_error:.4f} rad')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # nice lil summary type shit 
    print(f"\n{'='*10}")
    print(f"SPOT CHECK SUMMARY: dz={dz_mm:.1f} mm, v0={v0:.2f} cycles/ap, Trial {trial_idx}")
    print(f"{'='*10}")
    print(f"RMS error: {rms_error:.4f} rad = {rms_error_nm:.1f} nm")
    print(f"  RMS error: {rms_error:.4f} rad = {rms_error_nm:.1f} nm")
    print(f"  P2V error: {np.max(valid_error) - np.min(valid_error):.4f} rad")
    print(f"{'='*10}\n")
    
    return rms_error_nm

def safe_log10(img, eps=1e-10):
    img = np.asarray(img, dtype=float)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0.0, None)
    m = img.max()
    if m <= 0:
        return np.full_like(img, np.nan)
    return np.log10(img / m + eps)

'''
Here we set the conditions of the Monte Carlo Loop
'''
dzs_mc = np.linspace(5, 250, 2)      # mm
dzs_mc = np.unique(np.sort(np.append(dzs_mc, 12.69)))
#dzs_mc = (12.69,-7) # in the case you want specific dz. however the vebrose plotting breaks because its a tuple
v0s_mc = np.linspace(0.5, 30, 2)     # cycles/aperture
# convergence testing
SAVE_ITERS = [0,10,20,50,100,150,200,250,300,400,500,700,900]  # iterations to plot
snapshot_data = {}


N_trials = 1
sigma_e = 11
seed = 12345
num_photons = 1e6
rng = np.random.default_rng(seed)
save_label = "MC_2x2_toggle_test_1000steps.npz"


'''
Here we set the system parameters, dictionaries, and simulation elements
'''
seal_parameters = {
    'image_dx': 2.0071,
    'efl': 500,
    'wavelength_meter': 650e-9,
    'pupil_size': 10.12e-3,
    'small_pupil_size_meter': 9.5e-3,
    'pupil_pixel_dimension': 256,
    'focal_length_meters': 500e-3,
    'q': 16,
    'Num_airycircles': 16, #interchange with 64, original used 16
    'grid_dim': 10,
    'm_waves': 0.00
}

seal_param_config = {
    'image_dx': 2.0071,
    'efl': seal_parameters['focal_length_meters'] * 1e3,
    'wavelength': 0.65,
    'pupil_size': seal_parameters['pupil_size'] * 1e3,
}

# Build simulation elements
conf = InstrumentConfiguration(seal_param_config)
pupil_grid = make_pupil_grid(256, seal_parameters['pupil_size'])
focal_grid = make_focal_grid(
    q=seal_parameters['q'],
    num_airy=seal_parameters['Num_airycircles'],
    pupil_diameter=seal_parameters['pupil_size'],
    focal_length=seal_parameters['focal_length_meters'],
    reference_wavelength=seal_parameters['wavelength_meter']
)
aperture = make_circular_aperture(seal_parameters['pupil_size'])
telescope_pupil = aperture(pupil_grid)
prop_p2f = FraunhoferPropagator(pupil_grid, focal_grid, seal_parameters['focal_length_meters'])

zernike_modes = make_zernike_basis(
    num_modes=256,
    D=seal_parameters['pupil_size'],
    grid=pupil_grid
)
defocus_template = zernike_modes[3]

m_waves = seal_parameters['m_waves']  # Peak amplitude of sinusoid in waves, for debugging can be set to 0
D_m = seal_parameters['pupil_size']
lam_m = seal_parameters['wavelength_meter']
f_m=seal_parameters['focal_length_meters']
F=f_m/D_m
pupil_mask = np.array(telescope_pupil.shaped > 0, dtype=bool)
wavelength_um = seal_param_config['wavelength']

'''
Time to Setup the Grid, the time estimations, and if we wanted a spot check
'''
Nd, Nv = len(dzs_mc), len(v0s_mc)
spot_check_dz = 100.0     # mm - defocus for spot check (will find nearest)
spot_check_v0 = 3.0       # cycles/ap - spatial frequency for spot check (will find nearest)

# Find nearest spot check indices
if do_spot_check:
    spot_check_i = int(np.argmin(np.abs(dzs_mc - spot_check_dz)))
    spot_check_j = int(np.argmin(np.abs(v0s_mc - spot_check_v0)))
    actual_dz = dzs_mc[spot_check_i]
    actual_v0 = v0s_mc[spot_check_j]
else:
    spot_check_i = -1
    spot_check_j = -1

# Time estimation
time_per_sample = 4
time_monte_carlo = Nd * Nv * time_per_sample * N_trials
print(f"\n{'='*50}")
print(f"Monte Carlo Configuration:")
print(f"  Grid size: {Nd} x {Nv} = {Nd*Nv} points")
print(f"  Trials per point: {N_trials}")
print(f"  Read noise: {sigma_e} e-/px")
print(f"  Estimated time: {time_monte_carlo / 3600:.2f} hours")
print(f"  Verbose: {verbose}")
if do_spot_check:
    print(f"  Spot check: True")
    print(f"    -> want a dz={spot_check_dz} mm, v0={spot_check_v0} cycles/ap")
    print(f"    -> closest indice to it is dz={actual_dz:.1f} mm (idx={spot_check_i}), v0={actual_v0:.2f} cycles/ap (idx={spot_check_j})")
else:
    print(f"  Spot check: False")
print(f"{'='*50}\n")


'''
Setup the storage arrays for....storage
'''
# Residual RMS (reconstruction - injected sinusoid) or other way around?
rms_residual_trials = np.full((Nd, Nv, N_trials), np.nan)
rms_residual_mean = np.full((Nd, Nv), np.nan)
rms_residual_std = np.full((Nd, Nv), np.nan)

# Total RMS (just the reconstruction magnitude)
rms_total_trials = np.full((Nd, Nv, N_trials), np.nan)
rms_total_mean = np.full((Nd, Nv), np.nan)
rms_total_std = np.full((Nd, Nv), np.nan)

convergence_rate = np.zeros((Nd, Nv))

t0 = time.time()

'''
Now we get to the actual loop, how exciting
'''

for i, dz in enumerate(dzs_mc):
    dz_mm = float(dz)

    if DEFOCUS_MODEL == "pupil_zernike":
        phi_def = calculate_defocus_phase(seal_parameters, dz_mm, telescope_pupil, defocus_template)
    else:
        phi_def = None
    
    for j, v0 in enumerate(v0s_mc):
        if v0 == 0:
            continue

        # Create sinusoidal aberration
        phi_sine_waves = make_sinusoidal_phase_waves(pupil_grid, D_m, float(v0), m_waves)
        phi_sine_rad = (2 * np.pi * phi_sine_waves)

        is_spot_check = do_spot_check and (i == spot_check_i) and (j == spot_check_j)
        n_converged = 0
        trial_results_residual = []
        trial_results_total = []
        #THIS IS THE TOGGLE

        if DEFOCUS_MODEL == "pupil_zernike":
            # defocus via pupil phase + Fraunhofer
            wf_focus = Wavefront(telescope_pupil * np.exp(1j * phi_sine_rad), lam_m)
            wf_focus.total_power = num_photons

            wf_defoc = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)), lam_m)
            wf_defoc.total_power = num_photons

            psf_focus_clean = psf_from_wavefront(wf_focus)
            psf_defoc_clean = psf_from_wavefront(wf_defoc)
        

        elif DEFOCUS_MODEL == "image_AS":
            # defocus via angular spectrum propagation of the focal-plane E-field 
            if not _HAS_AS:
                raise ImportError("import the functions idiot")

            # Focused pupil -> focused focal-plane COMPLEX FIELD via HCIPy Fraunhofer
            wf_focus = Wavefront(telescope_pupil * np.exp(1j * phi_sine_rad), lam_m)
            wf_focus.total_power = num_photons

            # HCIPy returns complex focal-plane E field on focal_grid
            wf_focus_focal = prop_p2f(wf_focus)
            E_focus = wf_focus_focal.electric_field.shaped
            imshow_field(np.exp(1j * np.angle(E_focus.ravel())), vmin=0) # do this for both input and output
            plt.show()
            # Focused PSF
            psf_focus_clean = np.asarray(wf_focus_focal.power.shaped, float)

            # Defocus distance must be physical propagation distance in microns for FDPR/AS
            dz_um = mm_to_um(dz_mm)

            # Angular spectrum propagate E-field by +dz
            wvl_um = wavelength_um
            dx_um  = seal_parameters['image_dx']
            H_fwd  = _angular_spectrum_transfer_function(E_focus.shape, wvl_um, dx_um, dz_um)
            E_def  = _angular_spectrum_prop(E_focus, H_fwd)

            psf_defoc_clean = np.abs(E_def)**2
            psf_defoc_clean = np.asarray(psf_defoc_clean, float)
            s0 = psf_focus_clean.sum()
            sd = psf_defoc_clean.sum()
            if sd > 0:
                psf_defoc_clean *= (s0 / sd)
        elif DEFOCUS_MODEL == "my_fdpr":
            # Use SimpleFDPR with pupil-plane defocus
            # Need to compute defocus phase if not already done
            if phi_def is None:
                phi_def = calculate_defocus_phase(seal_parameters, dz_mm, telescope_pupil, defocus_template)
            
            # Generate PSFs using HCIPy (same as pupil_zernike)
            wf_focus = Wavefront(telescope_pupil * np.exp(1j * phi_sine_rad), lam_m)
            wf_focus.total_power = num_photons
            
            wf_defoc = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)), lam_m)
            wf_defoc.total_power = num_photons
            
            psf_focus_clean = psf_from_wavefront(wf_focus)
            psf_defoc_clean = psf_from_wavefront(wf_defoc)

        print(f"PSF sum: {psf_focus_clean.sum():.2e}, max: {psf_focus_clean.max():.2e}")
        # NaN check
        if not np.all(np.isfinite(psf_focus_clean)) or not np.all(np.isfinite(psf_defoc_clean)):
            print(f"Non-finite clean PSF at dz={dz:.1f}, v0={v0:.1f} - skipping")
            continue

        for t in range(N_trials):
            if if_noise:
            # Add noise
                psf0 = add_read_noise(psf_focus_clean, sigma_e, rng)
                psfd = add_read_noise(psf_defoc_clean, sigma_e, rng)

                psf0 = np.clip(psf0, 0, None)
                psfd = np.clip(psfd, 0, None)
            else:
                # Temporarily bypass noise, debugging
                psf0 = psf_focus_clean.copy()  # no noise
                psfd = psf_defoc_clean.copy()  # no noise
            
            if not np.all(np.isfinite(psf0)) or not np.all(np.isfinite(psfd)):
                print(f"Nans in the noisy PSF at dz={dz:.1f}, v0={v0:.1f}, trial={t}\
                    probably want to fix that")
                continue

            try:
                # Run FDPR
                dz_um = mm_to_um(dz_mm)  # mm to µm
                if DEFOCUS_MODEL == "my_fdpr":
                    # Use SimpleFDPR with defocus phase directly
                    mp = SimpleFDPR(psf0)
                    phi_def_resized = resize(phi_def.shaped, psf0.shape, preserve_range=True)
                    mp.add_defocused_image(psfd, phi_def_resized)
                    
                    for it in range(250):
                        psf_reconstruction = mp.step()
                        if snapshot and it in SAVE_ITERS:
                            snapshot_data[it] = {
                                "psf_rec": psf_reconstruction,
                                "psf_rec_complex": mp.amp_focused * np.exp(1j * mp.phase_estimate),
                                "cost": [cf[-1] for cf in mp.cost_functions if len(cf) > 0]
                            }
                else:
                    mp = FocusDiversePhaseRetrieval(
                        [psf0, psfd], 
                        wavelength_um, 
                        [seal_parameters['image_dx']], 
                        [dz_um]
                    )

                    for it in range(250):
                        psf_reconstruction = mp.step()
                        if it in SAVE_ITERS:
                            snapshot_data[it] = {
                            "psf_rec": np.abs(psf_reconstruction)**2,
                            "psf_rec_complex": psf_reconstruction.copy(),
                            "cost": [cf[-1] for cf in mp.cost_functions if len(cf) > 0]
                        }
                if snapshot:
                    plot_psf_convergence(snapshot_data)
                    use_fft = (DEFOCUS_MODEL == "my_fdpr")
                    plot_pupil_convergence(snapshot_data, conf, telescope_pupil, use_fft=use_fft)
                    plot_cost_convergence(mp)

                # Clear snapshots so next (dz, v0) starts clean
                snapshot_data.clear()
                if PUPIL_READOUT == "mft_rev":
                    #  current readout
                    raw_pupil = np.angle(mft_rev(psf_reconstruction, conf))
                    if not np.any(np.isfinite(raw_pupil)):
                        continue
                    real_pupil = resize(raw_pupil, (256, 256), preserve_range=True) * telescope_pupil.shaped

                elif PUPIL_READOUT == "hcipy_backward":
                    # readout using HCIPy adjoint of Fraunhofer propagator 
                    # avoids the mft_rev geometry assumptions and matches HCIPy forward model.
                    from hcipy import Field, Wavefront

                    Erec = psf_reconstruction.shaped if hasattr(psf_reconstruction, "shaped") else np.asarray(psf_reconstruction)
                    Erec = np.asarray(Erec)

                    wf_focal = Wavefront(Field(Erec.ravel(), focal_grid), lam_m)
                    wf_pupil = prop_p2f.backward(wf_focal)

                    E_pupil = wf_pupil.electric_field.shaped
                    real_pupil = np.angle(E_pupil) * telescope_pupil.shaped
                    '''
                    Piston error is not wrong, just have to correct for it
                    Normalization factor to correct for :instance_data.norm_factor = 1 / (1j * focal_length * wavelength) 
                    '''
                elif PUPIL_READOUT == "fft":
                    # Direct FFT readout for myfdpr
                    if DEFOCUS_MODEL == "my_fdpr":
                        pupil_phase_full = mp.get_pupil_phase() 
                        real_pupil = resize(pupil_phase_full, (256, 256), preserve_range=True) * telescope_pupil.shaped
                    else:
                        # For other models, do FFT of the reconstruction, similar to  what i did
                        focal_field = np.sqrt(psf_reconstruction) * np.exp(1j * np.angle(psf_reconstruction))
                        pupil_field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(focal_field)))
                        real_pupil = resize(np.angle(pupil_field), (256, 256), preserve_range=True) * telescope_pupil.shaped

                masked_phase = real_pupil[pupil_mask]

                if not np.any(np.isfinite(masked_phase)):
                    continue

                # Get the injected sinusoidal phase as truth
                truth_phase = phi_sine_rad.shaped * telescope_pupil.shaped
                truth_masked = truth_phase[pupil_mask]

                # Median subtraction on reconstruction (following futzing notebook)
                reconstruction_median_subtracted = real_pupil - np.median(real_pupil[np.array(telescope_pupil.shaped, dtype=bool)])
                reconstruction_median_subtracted_masked = reconstruction_median_subtracted[pupil_mask]
                residual = truth_masked - reconstruction_median_subtracted_masked  # both (51468,)
                rms_residual_rad = np.sqrt(np.nanmean(residual**2))

                # Total RMS (magnitude of reconstruction)
                rms_total_rad = np.sqrt(np.nanmean(masked_phase**2))

                if not np.isfinite(rms_residual_rad) or not np.isfinite(rms_total_rad):
                    continue

                # Convert to nm
                rms_residual_nm = rms_residual_rad * lam_m * 1e9 / (2*np.pi)
                rms_total_nm = rms_total_rad * lam_m * 1e9 / (2*np.pi)

                if np.isfinite(rms_residual_nm) and np.isfinite(rms_total_nm):
                    rms_residual_trials[i, j, t] = rms_residual_nm
                    rms_total_trials[i, j, t] = rms_total_nm
                    trial_results_residual.append(rms_residual_nm)
                    trial_results_total.append(rms_total_nm)
                    n_converged += 1

                    # Verbose plotting: first and last trial at every grid point
                    if verbose and (t == 0 or t == N_trials - 1):
                        plot_verbose_trial(
                            dz_mm, v0, t, phi_sine_rad, phi_def,
                            psf_focus_clean, psf_defoc_clean, psf0, psfd,
                            psf_reconstruction, real_pupil, telescope_pupil, seal_parameters
                        )

                    # Spot check detailed plot (first trial only)
                    if is_spot_check and t == 0:
                        plot_spot_check_full(
                            dz_mm, v0, phi_sine_rad, phi_def,
                            psf_focus_clean, psf_defoc_clean,
                            psf0, psfd, psf_reconstruction, real_pupil,
                            telescope_pupil, seal_parameters, trial_idx=t
                        )
            except Exception as e:
                print(f"FDPR error at dz={dz:.1f}, v0={v0:.1f}, trial={t}: {e}")
                continue
        # Store statistics
        convergence_rate[i, j] = n_converged / N_trials

        if len(trial_results_residual) > 0:
            rms_residual_mean[i, j] = np.mean(trial_results_residual)
            rms_residual_std[i, j] = np.std(trial_results_residual)
        if len(trial_results_total) > 0:
            rms_total_mean[i, j] = np.mean(trial_results_total)
            rms_total_std[i, j] = np.std(trial_results_total)

    # Progress
    elapsed = time.time() - t0
    rate = (i + 1) / elapsed if elapsed > 0 else 0
    remaining = (Nd - i - 1) / rate if rate > 0 else 0
    print(f"Done dz index {i+1}/{Nd} ({dz:.1f} mm) - "
          f"elapsed: {elapsed/60:.1f} min, remaining: {remaining/60:.1f} min")

elapsed = time.time() - t0
print(f"\nMC finished in {elapsed/60:.2f} min")

'''
SUMMARY STATISTICS
'''
print(f"\n{'='*10}")
print("MONTE CARLO TRIALS SUMMARY")
print(f"{'='*10}")
print(f"Grid shape: {Nd} dz x {Nv} v0")
print(f"Trials per point: {N_trials}")

valid_residual = np.sum(np.isfinite(rms_residual_trials))
valid_total = np.sum(np.isfinite(rms_total_trials))
print(f"Valid residual trials: {valid_residual} ({100*valid_residual/rms_residual_trials.size:.1f}%)")
print(f"Valid total trials: {valid_total} ({100*valid_total/rms_total_trials.size:.1f}%)")

valid_res_means = rms_residual_mean[np.isfinite(rms_residual_mean)]
valid_tot_means = rms_total_mean[np.isfinite(rms_total_mean)]

if len(valid_res_means) > 0:
    print(f"\nResidual RMS statistics:")
    print(f"  Mean: {np.mean(valid_res_means):.2f} nm")
    print(f"  Std: {np.std(valid_res_means):.2f} nm")
    print(f"  Range: {np.min(valid_res_means):.2f} - {np.max(valid_res_means):.2f} nm")

if len(valid_tot_means) > 0:
    print(f"\nTotal RMS statistics:")
    print(f"  Mean: {np.mean(valid_tot_means):.2f} nm")
    print(f"  Std: {np.std(valid_tot_means):.2f} nm")
    print(f"  Range: {np.min(valid_tot_means):.2f} - {np.max(valid_tot_means):.2f} nm")

print(f"{'='*10}\n")


'''
SAVE RESULTS
'''
np.savez(save_label,
         dzs_mc=dzs_mc,
         v0s_mc=v0s_mc,
         rms_residual_trials=rms_residual_trials,
         rms_residual_mean=rms_residual_mean,
         rms_residual_std=rms_residual_std,
         rms_total_trials=rms_total_trials,
         rms_total_mean=rms_total_mean,
         rms_total_std=rms_total_std,
         convergence_rate=convergence_rate,
         N_trials=N_trials,
         sigma_e=sigma_e,
         seed=seed,
         m_waves=m_waves)
print(f"Saved: {save_label}")


'''
PLOTTING
'''
extent_mc = [v0s_mc.min(), v0s_mc.max(), dzs_mc.min(), dzs_mc.max()]

# Residual RMS heatmap
plt.figure(figsize=(8, 6))
finite_vals = rms_residual_mean[np.isfinite(rms_residual_mean)]
if len(finite_vals) > 0:
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
    plt.imshow(rms_residual_mean, origin='lower', aspect='auto', extent=extent_mc,
               cmap='magma_r', vmin=vmin, vmax=vmax)
else:
    plt.imshow(rms_residual_mean, origin='lower', aspect='auto', extent=extent_mc, cmap='magma_r')
plt.colorbar(label="Residual RMS [nm]")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("MC Residual RMS (Reconstruction - Truth)")
plt.tight_layout()
plt.show()

# Total RMS heatmap
plt.figure(figsize=(8, 6))
finite_vals = rms_total_mean[np.isfinite(rms_total_mean)]
if len(finite_vals) > 0:
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
    plt.imshow(rms_total_mean, origin='lower', aspect='auto', extent=extent_mc,
               cmap='magma_r', vmin=vmin, vmax=vmax)
else:
    plt.imshow(rms_total_mean, origin='lower', aspect='auto', extent=extent_mc, cmap='magma_r')
plt.colorbar(label="Total RMS [nm]")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("MC Total RMS (Reconstruction Magnitude)")
plt.tight_layout()
plt.show()

# Convergence rate
plt.figure(figsize=(8, 6))
plt.imshow(convergence_rate, origin='lower', aspect='auto', extent=extent_mc,
           cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label="Convergence rate")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("FDPR Convergence Rate")
plt.tight_layout()
plt.show()

# Residual std heatmap
plt.figure(figsize=(8, 6))
finite_vals = rms_residual_std[np.isfinite(rms_residual_std)]
if len(finite_vals) > 0:
    vmin, vmax = np.nanpercentile(finite_vals, [5, 95])
    plt.imshow(rms_residual_std, origin='lower', aspect='auto', extent=extent_mc,
               cmap='viridis', vmin=vmin, vmax=vmax)
else:
    plt.imshow(rms_residual_std, origin='lower', aspect='auto', extent=extent_mc, cmap='viridis')
plt.colorbar(label="Residual RMS Std [nm]")
plt.xlabel("v0 [cycles/ap]")
plt.ylabel("dz [mm]")
plt.title("MC Residual RMS Variability")
plt.tight_layout()
plt.show()

def spot_check_interactive(dz_target, v0_target, npz_file='MC_4x4_new.npz'):
    """
    Run spot check on any (dz, v0) point after simulation is done.
    Regenerates PSFs and runs FDPR fresh.
    """
    # Find nearest indices
    mc = np.load(npz_file)
    dzs_mc = mc['dzs_mc']
    v0s_mc = mc['v0s_mc']
    
    i = int(np.argmin(np.abs(dzs_mc - dz_target)))
    j = int(np.argmin(np.abs(v0s_mc - v0_target)))
    dz_mm = float(dzs_mc[i])
    v0 = float(v0s_mc[j])
    
    print(f"Running spot check at dz={dz_mm:.1f} mm, v0={v0:.2f} cycles/ap")
    
    # Regenerate everything for this point
    phi_def = calculate_defocus_phase(seal_parameters, dz_mm, telescope_pupil, defocus_template)
    phi_sine_waves = make_sinusoidal_phase_waves(pupil_grid, D_m, float(v0), m_waves)
    phi_sine_rad = 2 * np.pi * phi_sine_waves
    
    wf_focus = Wavefront(telescope_pupil * np.exp(1j * phi_sine_rad), lam_m)
    wf_focus.total_power = num_photons

    psf_focus_clean = psf_from_wavefront(wf_focus)
    
    wf_defoc = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)), lam_m)
    wf_defoc.total_power = num_photons
    psf_defoc_clean = psf_from_wavefront(wf_defoc)

    if if_noise:
    # Add noise
        psf0 = np.clip(add_read_noise(psf_focus_clean, sigma_e, rng), 0, None)
        psfd = np.clip(add_read_noise(psf_defoc_clean, sigma_e, rng), 0, None)
    else:
        psf0 = psf_focus_clean.copy()
        psfd =psf_defoc_clean.copy()
    
    
    # Run FDPR
    dz_um = mm_to_um(dz_mm)
    mp = FocusDiversePhaseRetrieval([psf0, psfd], wavelength_um, 
                                     [seal_parameters['image_dx']], [dz_um])
    for _ in range(100):
        psf_reconstruction = mp.step()
    
    # Reconstruct pupil
    raw_pupil = np.angle(mft_rev(psf_reconstruction, conf))
    real_pupil = resize(raw_pupil, (256, 256)) * telescope_pupil.shaped
    
    # Plot!
    plot_spot_check_full(dz_mm, v0, phi_sine_rad, phi_def,
                         psf_focus_clean, psf_defoc_clean,
                         psf0, psfd, psf_reconstruction, real_pupil,
                         telescope_pupil, seal_parameters, trial_idx=0)
    

