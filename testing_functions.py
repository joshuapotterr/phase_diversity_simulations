#TODO

'''
double chevk with jaren about how airy rings change instrument config
        ie we had one set of parameters with 12 airy rings, and use same parameters iwht 64 airy rings would that change anythign
    check with emiel about ocnversion of pupil->focal plane
        spceifically mft_rev/smarter way of conversion from focal to pupil after reconstruction
        

'''

#Testing Blocks for Phase Diversity
find_L=False
if find_L:
    F_num = seal_parameters['focal_length_meters'] / seal_parameters['pupil_size']
    W020 = 1.0  # one wave of defocus
    L_one_wave_m = 8 * W020 * lam_m * F_num**2  # result in meters
    L_one_wave_mm = L_one_wave_m * 1e3           # convert to mm for the function

    print(f"F-number: {F_num:.1f}")
    print(f"Distance for 1 wave defocus: {L_one_wave_mm:.2f} mm")

    # Generate PSF with exactly 1 wave of defocus, no aberration
    phi_def_test = calculate_defocus_phase(seal_parameters, L_one_wave_mm, telescope_pupil, defocus_template)
    print(f"defocus after conversion function:{phi_def_test}")
    wf_test = Wavefront(telescope_pupil * np.exp(1j * phi_def_test), lam_m)
    wf_test.total_power = num_photons
    psf_test = psf_from_wavefront(wf_test)

    #  P2V of defocus phase
    mask = np.array(telescope_pupil.shaped, dtype=bool)
    phi_masked = phi_def_test.shaped[mask]
    p2v = phi_masked.max() - phi_masked.min()
    print(f"Defocus P2V: {p2v:.2f} rad (expected: {2*np.pi:.2f} rad for 1 wave)")

    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(phi_def_test.shaped, cmap='RdBu_r')
    plt.colorbar()
    plt.title(f'Defocus phase at L={L_one_wave_mm:.2f} mm\n(should be ~2pi rad P2V)')

    plt.subplot(132)
    plt.imshow(psf_test, cmap='inferno')
    plt.colorbar()
    plt.title('PSF (linear scale)\nShould have DARK CENTER')

    plt.subplot(133)
    plt.imshow(np.log10(psf_test/psf_test.max()), vmin=-4, cmap='inferno')
    plt.colorbar()
    plt.title('PSF (log scale)\nShould be donut shaped')

    plt.tight_layout()
    plt.show()

    # center value
    center = psf_test.shape[0]//2
    print(f"Center pixel value: {psf_test[center, center]:.2e}")
    print(f"Max pixel value: {psf_test.max():.2e}")
    print(f"Ratio (should be ~0): {psf_test[center, center]/psf_test.max():.4f}")

#another test

plotting_donut=False
if plotting_donut:


    #mechanical defocus for one wave
    lam_m = seal_parameters['wavelength_meter']
    f_m   = seal_parameters['focal_length_meters']
    D_m   = seal_parameters['pupil_size']
    F_num = f_m / D_m

    W020 = 1.0  # interpret as OPD_P2V = 1*lambda
    L_one_wave_m  = 8 * W020 * lam_m * (F_num**2)   # meters
    L_one_wave_mm = 1e3 * L_one_wave_m              # mm

    print(f"F/# = {F_num:.3f}")
    print(f"Defocus distance for 1 wave P2V OPD: {L_one_wave_mm:.6f} mm")


    #psf generation
    # In-focus PSF (no aberration)
    wf0 = Wavefront(telescope_pupil, lam_m)
    wf0.total_power = num_photons
    psf0 = psf_from_wavefront(wf0)
    psf0_img = psf0.shaped if hasattr(psf0, "shaped") else np.array(psf0)

    # Defocus phase (rad) via your conversion
    phi_def_test = calculate_defocus_phase(
        seal_parameters,
        L_one_wave_mm,
        telescope_pupil,
        defocus_template
    )

    # Defocused PSF
    wf1 = Wavefront(telescope_pupil * np.exp(1j * phi_def_test), lam_m)
    wf1.total_power = num_photons
    psf1 = psf_from_wavefront(wf1)
    psf1_img = psf1.shaped if hasattr(psf1, "shaped") else np.array(psf1)

    # Normalize flux
    psf0_img = psf0_img / np.sum(psf0_img)
    psf1_img = psf1_img / np.sum(psf1_img)

    # Sanity: injected defocus P2V in radians
    mask = np.array(telescope_pupil.shaped, dtype=bool)
    phi_masked = phi_def_test.shaped[mask]
    p2v_phi = phi_masked.max() - phi_masked.min()
    print(f"Injected defocus P2V: {p2v_phi:.4f} rad  ({p2v_phi/(2*np.pi):.4f} waves)  expected ~1.0000 waves")


    #fdpr time
    psf_list = [psf0_img, psf1_img]

    # distance_list in microns (mm -> um)
    delta_um = L_one_wave_mm * 1e3
    distance_list = [delta_um]

    # dx_list matches number of defocused images
    dx_val = float(seal_parameters.get('image_dx', 2.0071))
    dx_list = [dx_val]
    efl_m = f_m

    print("psf shapes:", psf0_img.shape, psf1_img.shape)
    print("distance_list [um]:", distance_list)
    print("dx_list:", dx_list)
    print("efl [m]:", efl_m)


    #iterations
    mp = FocusDiversePhaseRetrieval(psf_list, efl_m, dx_list, distance_list)

    n_iter = 200
    for k in range(n_iter):
        psf00 = mp.step()

    # array for plotting
    psf00_img = psf00.shaped if hasattr(psf00, "shaped") else np.array(psf00)
    psf00_img = np.real(psf00_img)
    psf00_img = psf00_img / np.sum(psf00_img)


    # pupil plane phase estimate
    # InstrumentConfiguration expects: efl (mm), wavelength (microns), pupil_size (mm)
    conf_dict = {
        'image_dx': dx_val,
        'efl': float(efl_m * 1e3),
        'wavelength': float(lam_m * 1e6),
        'pupil_size': float(D_m * 1e3),
    }
    conf = InstrumentConfiguration(conf_dict)

    raw_pupil_phase = np.angle(mft_rev(psf00, conf))

    N = telescope_pupil.shaped.shape[0]
    pupil_phase = resize(raw_pupil_phase, (N, N), preserve_range=True) * telescope_pupil.shaped

    # pupil phase P2V
    p2v_retr = pupil_phase[mask].max() - pupil_phase[mask].min()
    print(f"Retrieved pupil phase P2V: {p2v_retr:.4f} rad  ({p2v_retr/(2*np.pi):.4f} waves)")



    print("psf00 diagnostics:")
    print("  min:", np.nanmin(psf00_img))
    print("  max:", np.nanmax(psf00_img))
    print("  any NaN:", np.isnan(psf00_img).any())
    print("  any inf:", np.isinf(psf00_img).any())

    def safe_log10_psf(psf, floor=1e-12):
        """
        Safely compute log10-normalized PSF for plotting.
        - clips negatives
        - removes NaNs/Infs
        - normalizes by max
        """
        psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
        psf = np.clip(psf, 0.0, None)

        maxval = psf.max()
        if maxval <= 0:
            return np.full_like(psf, np.nan)

        psf_norm = psf / maxval
        return np.log10(psf_norm + floor)


    #plots
    eps = 1e-12
    psf0_log  = safe_log10_psf(psf0_img)
    psf1_log  = safe_log10_psf(psf1_img)
    psf00_log = safe_log10_psf(psf00_img)

    # Center-pixel donut diagnostic (for the defocused frame)
    c = psf1_img.shape[0] // 2
    print(f"Defocused PSF center/max: {psf1_img[c,c]/psf1_img.max():.6e}")

    fig = plt.figure(figsize=(16, 10))

    # --- Row 1: injected phase + pupil + template preview ---
    ax = plt.subplot(2, 4, 1)
    im = ax.imshow(phi_def_test.shaped, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Injected defocus phase phi (rad)")

    ax = plt.subplot(2, 4, 2)
    im = ax.imshow(telescope_pupil.shaped, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pupil amplitude")

    ax = plt.subplot(2, 4, 3)
    im = ax.imshow(defocus_template.shaped, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Defocus template (arb)")

    ax = plt.subplot(2, 4, 4)
    # show masked phi histogram-ish via image: use centered version for dynamic range
    phi0 = phi_def_test.shaped.copy()
    phi0[~mask] = np.nan
    im = ax.imshow(phi0, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"phi inside pupil\nP2V={p2v_phi/(2*np.pi):.3f} waves")

    #Row 2: PSFs (linear + log) and retrieved pupil phase 
    ax = plt.subplot(2, 4, 5)
    im = ax.imshow(psf0_img, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("PSF (in-focus) linear")

    ax = plt.subplot(2, 4, 6)
    im = ax.imshow(psf1_img, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("PSF (defocused) linear")

    ax = plt.subplot(2, 4, 7)
    im = ax.imshow(psf1_log, origin='lower', vmin=-6, vmax=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("PSF (defocused) log10\n(vmin=-6, vmax=0)")

    ax = plt.subplot(2, 4, 8)
    pp = pupil_phase.copy()
    pp[~mask] = np.nan
    im = ax.imshow(pp, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Retrieved pupil phase (rad)\nP2V={p2v_retr/(2*np.pi):.3f} waves")

    plt.tight_layout()
    plt.show()

    # Optional extra plot: compare measured defocused PSF vs FDPR predicted (if psf00 is that)
    fig = plt.figure(figsize=(14,4))
    ax = plt.subplot(1,3,1)
    im = ax.imshow(psf1_log, origin='lower', vmin=-6, vmax=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Measured defocused PSF (log10)")

    ax = plt.subplot(1,3,2)
    im = ax.imshow(psf00_log, origin='lower', vmin=-6, vmax=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("FDPR output PSF00 (log10)")

    ax = plt.subplot(1,3,3)
    diff = psf1_log - psf00_log
    im = ax.imshow(diff, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("log10(Meas) - log10(FDPR)")

    plt.tight_layout()
    plt.show()


jaren_complete_check=True
if jaren_complete_check:
    def safe_log10_img(img, floor=1e-10):
        """Safe log10 for plotting: clip negatives, remove NaNs/Infs, normalize by max."""
        arr = np.asarray(img, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)
        m = arr.max()
        if m <= 0:
            return np.full_like(arr, np.nan)
        return np.log10(arr / m + floor)

    def plot_jaren_sanity(phi_def, psf_focus_clean, psf_defoc_clean,
                        psf0, psfd, psf_reconstruction, real_pupil,
                        telescope_pupil, seal_parameters, dz_mm, title_tag=""):
        """2x4 debug plot in your style."""
        mask = np.array(telescope_pupil.shaped, dtype=bool)

        # reconstruction median-sub (same as your pipeline)
        recon_medsub = real_pupil - np.median(real_pupil[mask])
        diff = (phi_def.shaped * pupil_mask) - recon_medsub
        print(f"this is the difference bewtween phi_def.shaped* pupil mask - recon_medsub{diff}")


        # If "truth" is zero aberration, then truth phase = 0 (inside pupil)
        truth_phase = np.zeros_like(real_pupil)

        diff = truth_phase - recon_medsub
        rms_rad = np.sqrt(np.nanmean((diff[mask])**2))
        rms_nm = rms_rad * seal_parameters['wavelength_meter'] * 1e9 / (2*np.pi)

        # Defocus P2V inside pupil (conversion check)
        phi_def_s = phi_def.shaped
        p2v_phi = phi_def_s[mask].max() - phi_def_s[mask].min()
        p2v_waves = p2v_phi / (2*np.pi)

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(f"JAREN SANITY CHECK {title_tag}\n"
                    f"dz={dz_mm:.4f} mm | defocus P2V={p2v_waves:.3f} waves | RMS(recon)={rms_nm:.2f} nm",
                    fontsize=14)

        ax = axes[0,0]
        im = ax.imshow(phi_def.shaped, cmap='RdBu_r')
        ax.set_title("Injected Defocus Phase [rad]")
        plt.colorbar(im, ax=ax); ax.axis('off')

        ax = axes[0,1]
        im = ax.imshow(safe_log10_img(psf_defoc_clean), vmin=-6, vmax=0, cmap='inferno')
        ax.set_title("Defocused PSF (clean, log10)")
        plt.colorbar(im, ax=ax); ax.axis('off')

        ax = axes[0,2]
        im = ax.imshow(safe_log10_img(psf0), vmin=-6, vmax=0, cmap='inferno')
        ax.set_title("Focused PSF (fed to FDPR, log10)")
        plt.colorbar(im, ax=ax); ax.axis('off')

        ax = axes[0,3]
        im = ax.imshow(safe_log10_img(psfd), vmin=-6, vmax=0, cmap='inferno')
        ax.set_title("Defocused PSF (fed to FDPR, log10)")
        plt.colorbar(im, ax=ax); ax.axis('off')

        ax = axes[1,0]
        # show FDPR reconstruction magnitude (if complex)
        rec = psf_reconstruction
        rec_abs = np.abs(rec) if np.iscomplexobj(rec) else np.asarray(rec)
        im = ax.imshow(safe_log10_img(rec_abs), vmin=-6, vmax=0, cmap='inferno')
        ax.set_title("FDPR psf_reconstruction |.| (log10)")
        plt.colorbar(im, ax=ax); ax.axis('off')

        ax = axes[1,1]
        im = ax.imshow(real_pupil, cmap='RdBu_r')
        ax.set_title("Reconstructed Pupil Phase [rad]")
        plt.colorbar(im, ax=ax); ax.axis('off')

        ax = axes[1,2]
        im = ax.imshow(recon_medsub, cmap='RdBu_r')
        ax.set_title("Recon (median sub) [rad]")
        plt.colorbar(im, ax=ax); ax.axis('off')

        ax = axes[1,3]
        im = ax.imshow(diff, cmap='RdBu_r')
        ax.set_title(f"Difference (0 - recon_medsub)\nRMS={rms_nm:.2f} nm")
        plt.colorbar(im, ax=ax); ax.axis('off')

        plt.tight_layout()
        plt.show()

        # extra scalar diagnostics
        c = psfd.shape[0]//2
        print(f"[Sanity] defocus P2V: {p2v_phi:.4f} rad = {p2v_waves:.4f} waves")
        print(f"[Sanity] defocused center/max: {psfd[c,c]/psfd.max():.3e}  (donut-ish if small)")
        print(f"[Sanity] FDPR recon pupil RMS (median-sub, vs 0 truth): {rms_nm:.2f} nm")


    # 1 wave of defocus
    D_m = seal_parameters['pupil_size']
    lam_m = seal_parameters['wavelength_meter']
    f_m  = seal_parameters['focal_length_meters']
    F = f_m / D_m

    W020 = 1.0
    L_one_wave_meter = 8 * W020 * lam_m * (F**2)
    L_one_wave_mm = L_one_wave_meter * 1e3
    print(f"Distance for 1 wave defocus: {L_one_wave_mm:.4f} mm")

    # wf construction, m_waves is 0
    phi_sine_rad = 0.0 * telescope_pupil  # no injected aberration
    phi_def = calculate_defocus_phase(seal_parameters, L_one_wave_mm, telescope_pupil, defocus_template)

    wf_focus = Wavefront(telescope_pupil * np.exp(1j * phi_sine_rad), lam_m)
    wf_focus.total_power = num_photons

    wf_defoc = Wavefront(telescope_pupil * np.exp(1j * (phi_sine_rad + phi_def)), lam_m)
    wf_defoc.total_power = num_photons

    psf_focus_clean = psf_from_wavefront(wf_focus)
    psf_defoc_clean = psf_from_wavefront(wf_defoc)

    # quick display of psfs
    plt.figure()
    plt.imshow(safe_log10_img(psf_defoc_clean), vmin=-6, vmax=0, cmap='inferno')
    plt.colorbar()
    plt.title("Defocused (clean) log10")
    plt.show()

    plt.figure()
    plt.imshow(safe_log10_img(psf_focus_clean), vmin=-6, vmax=0, cmap='inferno')
    plt.colorbar()
    plt.title("Focused (clean) log10")
    plt.show()
    if_noise=False
    # noise toggle
    if if_noise:
        psf0 = np.clip(add_read_noise(psf_focus_clean, sigma_e, rng), 0, None)
        psfd = np.clip(add_read_noise(psf_defoc_clean, sigma_e, rng), 0, None)
    else:
        psf0 = psf_focus_clean.copy()
        psfd = psf_defoc_clean.copy()

    # fdpr time
    dz_um = mm_to_um(L_one_wave_mm)

    mp = FocusDiversePhaseRetrieval(
        [psf0, psfd],
        wavelength_um,                   # you pass wavelength in microns here
        [seal_parameters['image_dx']],
        [dz_um]
    )

    N_iter = 150
    psf_reconstruction = None
    for _ in range(N_iter):
        psf_reconstruction = mp.step()

    # diagnositcs b4 plotting
    rec_arr = np.asarray(psf_reconstruction)
    print("psf_reconstruction diagnostics:")
    print("  complex:", np.iscomplexobj(rec_arr))
    print("  min:", np.nanmin(np.real(rec_arr)))
    print("  max:", np.nanmax(np.real(rec_arr)))
    print("  any NaN:", np.isnan(rec_arr).any(), " any inf:", np.isinf(rec_arr).any())

    # pupil recosntruction
    raw_pupil = np.angle(mft_rev(psf_reconstruction, conf))
    real_pupil = resize(raw_pupil, (256, 256), preserve_range=True) * telescope_pupil.shaped
    

    #verbose plotting call
    plot_jaren_sanity(
        phi_def=phi_def,
        psf_focus_clean=psf_focus_clean,
        psf_defoc_clean=psf_defoc_clean,
        psf0=psf0, psfd=psfd,
        psf_reconstruction=psf_reconstruction,
        real_pupil=real_pupil,
        telescope_pupil=telescope_pupil,
        seal_parameters=seal_parameters,
        dz_mm=L_one_wave_mm,
        title_tag="(m_waves=0, 1-wave defocus)"
    )

if_angular_prop=True
if if_angular_prop:
    from image_sharpening import _angular_spectrum_transfer_function, _angular_spectrum_prop
    # from image_sharpening import FocusDiversePhaseRetrieval, mft_rev, InstrumentConfiguration

    def safe_log10_img(img, floor=1e-10):
        arr = np.asarray(img, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, None)
        m = arr.max()
        if m <= 0:
            return np.full_like(arr, np.nan)
        return np.log10(arr / m + floor)

    # physical def distance being chosen
    D_m   = seal_parameters['pupil_size']
    lam_m = seal_parameters['wavelength_meter']
    f_m   = seal_parameters['focal_length_meters']
    Fnum  = f_m / D_m

    W020 = 1.0
    L_one_wave_m  = 8 * W020 * lam_m * (Fnum**2)
    dz_mm = 1e3 * L_one_wave_m
    dz_um = 1e3 * dz_mm

    print(f"Using dz = {dz_mm:.4f} mm  ({dz_um:.2f} um)")

    #pupil phase with def=0
    phi_aberr = 0.0 * telescope_pupil  # (optional) replace with phi_sine_rad for aberration tests

    wf_focus = Wavefront(telescope_pupil * np.exp(1j * phi_aberr), lam_m)
    wf_focus.total_power = num_photons

    # Complex focal-plane field (in-focus) using your HCIPy propagator
    E_focus = prop_p2f(wf_focus).electric_field.shaped   # complex ndarray
    psf_focus_clean = np.abs(E_focus)**2
    psf_focus_clean = np.asarray(psf_focus_clean, float)

 
    #asp the focal field by a chosen dz, results in def focal plane field

    wvl_um = lam_m * 1e6
    dx_um = seal_parameters['image_dx']  # IMPORTANT: must match sampling of the focal-plane array

    H_fwd = _angular_spectrum_transfer_function(E_focus.shape, wvl_um, dx_um, dz_um)
    E_defoc = _angular_spectrum_prop(E_focus, H_fwd)

    psf_defoc_clean = np.abs(E_defoc)**2
    psf_defoc_clean = np.asarray(psf_defoc_clean, float)

    # Normalize (optional but helps consistency)
    psf_focus_clean /= psf_focus_clean.sum()
    psf_defoc_clean /= psf_defoc_clean.sum()

    #input plots
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(safe_log10_img(psf_focus_clean), vmin=-6, vmax=0, cmap='inferno')
    plt.title("Focused PSF (AS test input)")
    plt.colorbar(); plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(safe_log10_img(psf_defoc_clean), vmin=-6, vmax=0, cmap='inferno')
    plt.title("Defocused PSF (AS propagated)")
    plt.colorbar(); plt.axis('off')
    plt.tight_layout()
    plt.show()

    # fdpr call
    if if_noise:
        psf0 = np.clip(add_read_noise(psf_focus_clean, sigma_e, rng), 0, None)
        psfd = np.clip(add_read_noise(psf_defoc_clean, sigma_e, rng), 0, None)
    else:
        psf0 = psf_focus_clean.copy()
        psfd = psf_defoc_clean.copy()

    mp = FocusDiversePhaseRetrieval(
        [psf0, psfd],
        wavelength_um,                      # microns
        [seal_parameters['image_dx']],       # microns/pixel
        [dz_um]                              # microns
    )

    psf_reconstruction = None
    for _ in range(150):
        psf_reconstruction = mp.step()
    mask = telescope_pupil.shaped.astype(bool)

    # prop_p2f.backward route
    Erec = psf_reconstruction.shaped if hasattr(psf_reconstruction, "shaped") else np.asarray(psf_reconstruction)
    Erec = np.asarray(Erec)  # keep complex dtype

    # prop_p2f.backward needs a wavefront
    E_focal_fdpr = Field(Erec.ravel(), focal_grid)
    wf_focal_fdpr = Wavefront(E_focal_fdpr, lam_m)

    # Adjoint back to pupil
    wf_pupil_est = prop_p2f.backward(wf_focal_fdpr)
    E_pupil_est = wf_pupil_est.electric_field.shaped
    phase_hcipy = np.angle(E_pupil_est) * telescope_pupil.shaped

    # mft_rev rout
    raw_pupil_mft = np.angle(mft_rev(psf_reconstruction, conf))
    phase_mft = resize(raw_pupil_mft, telescope_pupil.shaped.shape, preserve_range=True) * telescope_pupil.shaped

    # comparison
    diff = (phase_mft - phase_hcipy) * telescope_pupil.shaped
    rms_diff = np.sqrt(np.mean(diff[mask]**2))

    plt.figure(figsize=(15,4))

    plt.subplot(1,3,1)
    plt.imshow(phase_hcipy, cmap='RdBu_r')
    plt.colorbar()
    plt.title("Pupil phase via HCIPy backward()")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(phase_mft, cmap='RdBu_r')
    plt.colorbar()
    plt.title("Pupil phase via mft_rev()")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(diff, cmap='RdBu_r')
    plt.colorbar()
    plt.title(f"mft_rev − HCIPy (RMS={rms_diff:.3f} rad)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("RMS difference [rad] =", rms_diff)
    # focal to pupil
    raw_pupil = np.angle(mft_rev(psf_reconstruction, conf))
    real_pupil = resize(raw_pupil, (256, 256), preserve_range=True) * telescope_pupil.shaped

    # plot fdpr
    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    plt.imshow(safe_log10_img(psfd), vmin=-6, vmax=0, cmap='inferno')
    plt.title("Defocused PSF fed to FDPR")
    plt.colorbar(); plt.axis('off')

    plt.subplot(1,3,2)
    rec_abs = np.abs(psf_reconstruction) if np.iscomplexobj(psf_reconstruction) else np.asarray(psf_reconstruction)
    plt.imshow(safe_log10_img(rec_abs), vmin=-6, vmax=0, cmap='inferno')
    plt.title("FDPR psf_reconstruction |.|")
    plt.colorbar(); plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(real_pupil, cmap='RdBu_r')
    plt.title("Reconstructed pupil phase [rad]")
    plt.colorbar(); plt.axis('off')

    plt.tight_layout()
    plt.show()


#WORKING TEST
working_test = True
if working_test:
    lam_m = seal_parameters['wavelength_meter']
    f_m   = seal_parameters['focal_length_meters']
    D_m   = seal_parameters['pupil_size']
    F_num = f_m / D_m

    W020 = 1.0  # interpret as OPD_P2V = 1*lambda
    L_one_wave_m  = 8 * W020 * lam_m * (F_num**2)   # meters
    L_one_wave_mm = 1e3 * L_one_wave_m              # mm

    print(f"F/# = {F_num:.3f}")
    print(f"Defocus distance for 1 wave P2V OPD: {L_one_wave_mm:.6f} mm")


    #psf generation
    # In-focus PSF (no aberration)
    wf0 = Wavefront(telescope_pupil, lam_m)
    wf0.total_power = num_photons
    psf0 = psf_from_wavefront(wf0)
    psf0_img = psf0.shaped if hasattr(psf0, "shaped") else np.array(psf0)

    # Defocus phase (rad) via conversion
    phi_def_test = calculate_defocus_phase(
        seal_parameters,
        L_one_wave_mm,
        telescope_pupil,
        defocus_template
    )

    # Defocused PSF
    wf1 = Wavefront(telescope_pupil * np.exp(1j * phi_def_test), lam_m)
    wf1.total_power = num_photons
    psf1 = psf_from_wavefront(wf1)
    psf1_img = psf1.shaped if hasattr(psf1, "shaped") else np.array(psf1)

    # Normalize flux
    psf0_img = psf0_img / np.sum(psf0_img)
    psf1_img = psf1_img / np.sum(psf1_img)

    # Sanity: injected defocus P2V in radians
    mask = np.array(telescope_pupil.shaped, dtype=bool)
    phi_masked = phi_def_test.shaped[mask]
    p2v_phi = phi_masked.max() - phi_masked.min()
    print(f"Injected defocus P2V: {p2v_phi:.4f} rad  ({p2v_phi/(2*np.pi):.4f} waves)  expected ~1.0000 waves")


    #fdpr time
    psf_list = [psf0_img, psf1_img]

    # distance_list in microns (mm -> um)
    delta_um = L_one_wave_mm * 1e3
    distance_list = [delta_um]

    # dx_list matches number of defocused images
    dx_val = float(seal_parameters.get('image_dx', 2.0071))
    dx_list = [dx_val]
    efl_m = f_m

    print("psf shapes:", psf0_img.shape, psf1_img.shape)
    print("distance_list [um]:", distance_list)
    print("dx_list:", dx_list)
    print("efl [m]:", efl_m)


    #iterations
    mp = FocusDiversePhaseRetrieval(psf_list, lam_m*1e-3, dx_list, distance_list)

    n_iter = 200
    for k in range(n_iter):
        psf00 = mp.step()

    # array for plotting
    psf00_img = psf00.shaped if hasattr(psf00, "shaped") else np.array(psf00)
    psf00_img = np.real(psf00_img)
    psf00_img = psf00_img / np.sum(psf00_img)


    # pupil plane phase estimate
    # InstrumentConfiguration expects: efl (mm), wavelength (microns), pupil_size (mm)
    conf_dict = {
        'image_dx': dx_val,
        'efl': float(efl_m * 1e3),
        'wavelength': float(lam_m * 1e6),
        'pupil_size': float(D_m * 1e3),
    }
    conf = InstrumentConfiguration(conf_dict)

    raw_pupil_phase = np.angle(mft_rev(psf00, conf))

    N = telescope_pupil.shaped.shape[0]
    pupil_phase = resize(raw_pupil_phase, (N, N), preserve_range=True) * telescope_pupil.shaped

    # pupil phase P2V
    p2v_retr = pupil_phase[mask].max() - pupil_phase[mask].min()
    print(f"Retrieved pupil phase P2V: {p2v_retr:.4f} rad  ({p2v_retr/(2*np.pi):.4f} waves)")



    print("psf00 diagnostics:")
    print("  min:", np.nanmin(psf00_img))
    print("  max:", np.nanmax(psf00_img))
    print("  any NaN:", np.isnan(psf00_img).any())
    print("  any inf:", np.isinf(psf00_img).any())

    def safe_log10_psf(psf, floor=1e-12):
        """
        Safely compute log10-normalized PSF for plotting.
        - clips negatives
        - removes NaNs/Infs
        - normalizes by max
        """
        psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
        psf = np.clip(psf, 0.0, None)

        maxval = psf.max()
        if maxval <= 0:
            return np.full_like(psf, np.nan)

        psf_norm = psf / maxval
        return np.log10(psf_norm + floor)


    #plots
    eps = 1e-12
    psf0_log  = safe_log10_psf(psf0_img)
    psf1_log  = safe_log10_psf(psf1_img)
    psf00_log = safe_log10_psf(psf00_img)

    # Center-pixel donut diagnostic (for the defocused frame)
    c = psf1_img.shape[0] // 2
    print(f"Defocused PSF center/max: {psf1_img[c,c]/psf1_img.max():.6e}")

    fig = plt.figure(figsize=(16, 10))

    # --- Row 1: injected phase + pupil + template preview ---
    ax = plt.subplot(2, 4, 1)
    im = ax.imshow(phi_def_test.shaped, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Injected defocus phase φ (rad)")

    ax = plt.subplot(2, 4, 2)
    im = ax.imshow(telescope_pupil.shaped, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pupil amplitude")

    ax = plt.subplot(2, 4, 3)
    im = ax.imshow(defocus_template.shaped, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Defocus template (arb)")

    ax = plt.subplot(2, 4, 4)
    # show masked φ histogram-ish via image: use centered version for dynamic range
    phi0 = phi_def_test.shaped.copy()
    phi0[~mask] = np.nan
    im = ax.imshow(phi0, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"φ inside pupil\nP2V={p2v_phi/(2*np.pi):.3f} waves")

    # --- Row 2: PSFs (linear + log) and retrieved pupil phase ---
    ax = plt.subplot(2, 4, 5)
    im = ax.imshow(psf0_img, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("PSF (in-focus) linear")

    ax = plt.subplot(2, 4, 6)
    im = ax.imshow(psf1_img, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("PSF (defocused) linear")

    ax = plt.subplot(2, 4, 7)
    im = ax.imshow(psf1_log, origin='lower', vmin=-6, vmax=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("PSF (defocused) log10\n(vmin=-6, vmax=0)")

    ax = plt.subplot(2, 4, 8)
    pp = pupil_phase.copy()
    pp[~mask] = np.nan
    im = ax.imshow(pp, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Retrieved pupil phase (rad)\nP2V={p2v_retr/(2*np.pi):.3f} waves")

    plt.tight_layout()
    plt.show()

    # Optional extra plot: compare measured defocused PSF vs FDPR predicted (if psf00 is that)
    fig = plt.figure(figsize=(14,4))
    ax = plt.subplot(1,3,1)
    im = ax.imshow(psf1_log, origin='lower', vmin=-6, vmax=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Measured defocused PSF (log10)")

    ax = plt.subplot(1,3,2)
    im = ax.imshow(psf00_log, origin='lower', vmin=-6, vmax=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("FDPR output PSF00 (log10)")

    ax = plt.subplot(1,3,3)
    diff = psf1_log - psf00_log
    im = ax.imshow(diff, origin='lower')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("log10(Meas) - log10(FDPR)")

    plt.tight_layout()
    plt.show()


class SimpleFDPR:
    """
    Simple Focus-Diverse Phase Retrieval using Gerchberg-Saxton iteration.
    Uses FFT instead of Angular Spectrum for simplicity.
    Supports multiple defocused images.
    """
    
def __init__(self, psf_focused, defocus_list=None):
        """
        Parameters
        ----------
        psf_focused : numpy.ndarray
            Focused PSF (intensity image)
        defocus_list : list of tuples, optional
            List of (psf_defocused, defocus_phase) tuples
            Can also add later with add_defocused_image()
        """
        # Store amplitude constraint for focused image
        self.amp_focused = np.sqrt(psf_focused)
        
        # Store defocused data as list of (amplitude, phase) tuples
        self.defocused_data = []
        
        if defocus_list is not None:
            for psf_defoc, defoc_phase in defocus_list:
                self.add_defocused_image(psf_defoc, defoc_phase)
        
        # Initialize with random phase guess
        self.phase_estimate = np.random.rand(*psf_focused.shape) * 2 * np.pi
        
        self.iter = 0
        self.cost_history = []
    
def add_defocused_image(self, psf_defocused, defocus_phase):
        """
        Add a defocused image to the retrieval.
        
        Parameters
        ----------
        psf_defocused : numpy.ndarray
            Defocused PSF (intensity image)
        defocus_phase : numpy.ndarray
            Known defocus phase applied to get this PSF (in radians)
        """
        amp_defocused = np.sqrt(psf_defocused)
        self.defocused_data.append({
            'amplitude': amp_defocused,
            'defocus_phase': defocus_phase,
            'cost': []
        })
        print(f"Added defocused image. Total: {len(self.defocused_data)} defocused images.")
    
def _focal_to_pupil(self, focal_field):
        """Inverse FFT: focal plane -> pupil plane"""
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(focal_field)))
    
def _pupil_to_focal(self, pupil_field):
        """Forward FFT: pupil plane -> focal plane"""
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_field)))
    
def _compute_mse(self, estimate, target):
        """Compute mean squared error between estimate and target"""
        return np.mean((estimate - target)**2)
    
def step(self):
        """
        One iteration of Gerchberg-Saxton with focus diversity.
        Cycles through all defocused images.
        
        Returns
        -------
        focal_field : numpy.ndarray
            Current estimate of focused PSF field (complex)
        """
        if len(self.defocused_data) == 0:
            raise ValueError("No defocused images added. Use add_defocused_image() first.")
        
        # Start with current estimate in focal plane
        focal_field = self.amp_focused * np.exp(1j * self.phase_estimate)
        
        # Cycle through each defocused image
        for data in self.defocused_data:
            amp_defocused = data['amplitude']
            defocus_phase = data['defocus_phase']
            
            # Go to pupil plane
            pupil_field = self._focal_to_pupil(focal_field)
            pupil_phase = np.angle(pupil_field)
            pupil_amp = np.abs(pupil_field)
            
            # Add defocus and propagate to defocused focal plane
            pupil_defocused = pupil_amp * np.exp(1j * (pupil_phase + defocus_phase))
            focal_defocused = self._pupil_to_focal(pupil_defocused)
            
            # Compute cost before constraint
            cost = self._compute_mse(np.abs(focal_defocused), amp_defocused)
            data['cost'].append(cost)
            
            # Apply defocused amplitude constraint, keep phase
            focal_defocused_constrained = amp_defocused * np.exp(1j * np.angle(focal_defocused))
            
            # Go back to pupil plane
            pupil_field_2 = self._focal_to_pupil(focal_defocused_constrained)
            
            # Remove defocus to get back to focused pupil
            pupil_focused = np.abs(pupil_field_2) * np.exp(1j * (np.angle(pupil_field_2) - defocus_phase))
            
            # Propagate to focused focal plane
            focal_field = self._pupil_to_focal(pupil_focused)
            
            # Apply focused amplitude constraint
            focal_field = self.amp_focused * np.exp(1j * np.angle(focal_field))
        
        # Update phase estimate
        self.phase_estimate = np.angle(focal_field)
        self.iter += 1
        
        return focal_field
    
def run(self, n_iterations=100, verbose=True):
        """
        Run multiple iterations.
        
        Parameters
        ----------
        n_iterations : int
            Number of iterations to run
        verbose : bool
            Print progress
            
        Returns
        -------
        focal_field : numpy.ndarray
            Final estimate of focused PSF field (complex)
        """
        for i in range(n_iterations):
            focal_field = self.step()
            if verbose and (i+1) % 20 == 0:
                total_cost = sum(data['cost'][-1] for data in self.defocused_data)
                print(f"Iteration {i+1}/{n_iterations}, Total cost: {total_cost:.2e}")
        
        return focal_field
    
def get_pupil_phase(self):
        """Get current estimate of pupil phase"""
        focal_field = self.amp_focused * np.exp(1j * self.phase_estimate)
        pupil_field = self._focal_to_pupil(focal_field)
        return np.angle(pupil_field)
    
def get_pupil_field(self):
        """Get current estimate of full pupil field (complex)"""
        focal_field = self.amp_focused * np.exp(1j * self.phase_estimate)
        return self._focal_to_pupil(focal_field)
    
def plot_cost(self):
        """Plot cost function history for all defocused images"""
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
Usage:
# Method 1: Add defocused images at initialization
fdpr = SimpleFDPR(psf_focus_clean, defocus_list=[
    (psf_defoc_clean, phi_def.shaped),
])

# Method 2: Add defocused images later
fdpr = SimpleFDPR(psf_focus_clean)
fdpr.add_defocused_image(psf_defoc_1, phi_def_1.shaped)
fdpr.add_defocused_image(psf_defoc_2, phi_def_2.shaped)  # add more if you want

# Run iterations
psf_estimate = fdpr.run(n_iterations=100, verbose=True)

# Or step manually
for _ in range(100):
    psf_estimate = fdpr.step()

# Get reconstructed pupil phase
pupil_phase = fdpr.get_pupil_phase()

# Plot convergence
fdpr.plot_cost()
'''