"""
Enrich april28_test.ipynb markdown cells with explanatory prose in the
author's voice (matched against sample701.tex tone: long technical sentences,
semicolons and em-dashes, direct "we", explicit operating-point references,
honest hedges).  Preserves all code cells.
"""
import json
from pathlib import Path

NB = Path("/Users/joshuapotter/Documents/SEAL/FDPRNotebooks/april28_test.ipynb")

# index -> new markdown source (string).  Indices that don't appear are left
# untouched.  Each replacement preserves the existing header and integrates
# the prior text where possible.
NEW = {
3: """## Simulation Parameters

We fix a single diagnostic operating point --- $A = 0.10$ waves of sinusoidal phase at
$\\nu_0 = 10$ cyc/aperture, retrieved with a three-image diversity stack at
$\\Delta z = 40$ mm --- and vary one knob at a time downstream.  The point sits well
inside the regime in which the sideband expansion of Section~\\ref{sec:otf_theory}
of the paper is valid ($A \\ll 1$ rad, $\\nu_0 < \\nu_{\\rm cutoff}$); the goal is
not to map a parameter space but to characterize what the FDPR pipeline returns at
a regime where it should be working, so the residuals we see are attributable to
specific pipeline components rather than to a breakdown of the small-signal model.""",

5: """## Build Truth Aberration and Defocus Diversity

Construct the pupil-plane truth $\\phi(x, y) = A \\sin(2 \\pi \\nu_0 x)$ in radians
(so the peak-to-valley wavefront error is $2A$ waves), plus the diversity defocus
$\\phi_{\\Delta z}$ from Eq.~\\ref{eq:defocus_phase} of the paper.  Both fields
share the pupil grid and are multiplied by the SEAL aperture mask before
propagation; this keeps the reconstruction problem matched to the
forward model and avoids the off-aperture phase the FDPR algorithm has no
information about.""",

7: """## Generate PSFs (Focused + Diversity) with Noise

Build the three focal-plane intensities $\\{I_0, I_{+\\Delta z}, I_{-\\Delta z}\\}$
using the same Fraunhofer forward model the retrieval uses internally; this keeps the
data-vs-model comparison self-consistent and isolates noise as the only source of
discrepancy at this stage.  Inject Poisson photon noise at $N_\\gamma = 10^{6}$
photons/frame and Gaussian read noise at $\\sigma_r = 11\\,e^{-}$ rms, matching the
SEAL detector defaults; the resulting per-pixel variance is the quadrature sum of
Eq.~\\ref{eq:pixel_variance}.  We deliberately stay below the per-pixel saturation
regime so the noise budget is read-limited at the OTF-pixel level
(Section~\\ref{sec:noise_budget}); pushing into the photon-limited corner is left to
the photon-count sweep further down this notebook.""",

9: """## Run FDPR

Run the Misell-style amplitude-projection FDPR loop (Section~\\ref{sec:convergence})
on the three-image stack for $N_{\\rm iter} = 150$ steps with a random phase
initialization.  The 3-image configuration $(I_0, I_{+\\Delta z}, I_{-\\Delta z})$
is the minimum multi-image geometry that carries enough defocus signal for
Dean-Bowers selection while remaining symmetric; we use it as the reference
throughout the paper.  The iteration runs to a fixed budget rather than a
tolerance criterion (the latter is identified as future work in
Section~\\ref{sec:discussion}), so $N_{\\rm iter}$ has to be set large enough that the
recon is at the small-signal fixed point at the operating point chosen above ---
the iteration sweep below verifies this empirically.""",

11: """## Extract Recon Phase + Truth / Recon / Residual RMS

Take the angle of the MFT-reversed focused complex amplitude as the recon phase,
piston-subtract both truth and recon by the aperture-median (which coincides with
the aperture mean for the sinusoid and coma modes used here; bases with nonzero
aperture mean would need this convention revisited), and compute the residual RMS
over illuminated pupil pixels as in Eq.~\\ref{eq:rms_residual}.  Three numbers come
out of this cell: the truth RMS (a property of the injected aberration), the recon
RMS (a property of what the algorithm returns), and the residual RMS (the
quantity that actually matters for diversity-selection scoring).  The relative
ordering of these three is the first diagnostic --- residual $<$ truth means the
retrieval beats the null, residual $>$ truth means the retrieval has degraded past
the point of improvement and the small-signal model has broken down for this
operating point.""",

13: """## Alpha Decomposition: Amplitude Bias vs Orthogonal Noise

Decompose the recon as $\\phi_{\\rm rec} = \\alpha\\,\\phi_{\\rm truth} + \\delta$,
where $\\alpha = \\langle \\phi_{\\rm rec}, \\phi_{\\rm truth}\\rangle / \\|\\phi_{\\rm truth}\\|^2$
is the scalar projection onto the truth direction and $\\delta$ is the orthogonal
remainder by construction.  The point of separating $\\alpha$ from $\\delta$ is that
the two failure modes that show up in §3.5 of the paper (small-signal saturation
and noise-driven scatter) live in different places in this decomposition:
amplitude-bias / linearization saturation lives entirely in $\\alpha$, while
noise, modal cross-talk, MFT sampling, and `skimage` resize artifacts live entirely
in $\\delta$.  A reconstruction with $\\alpha = 0.86$ and $\\delta = 0$ would still
report a large residual RMS even though the algorithm is recovering the truth
direction perfectly --- residual $>$ recon whenever $\\alpha < 0.5$ is the
geometric consequence, not a bug.""",

15: """## Project Delta onto Tip / Tilt / Defocus

The orthogonal $\\delta$ contains everything that isn't a scaled copy of the truth;
the question is whether that "everything" is incoherent noise or whether it carries
coherent low-order modes that the retrieval is leaking into.  We project $\\delta$
onto $\\{x, y, x^2 + y^2\\}$ (tip, tilt, defocus) on the masked pupil and report the
fraction of $\\delta$'s power each mode explains.  A non-trivial tip/tilt
component would indicate a pixel-centering inconsistency between the forward
model and the recon; a non-trivial defocus component would indicate that the
diversity defocus is being absorbed into the unknown phase, which is the
classic sign of a sign-degeneracy not being broken cleanly by the chosen
$\\Delta z$.""",

17: """## Iteration Sweep: Is Alpha Still Climbing?

Run FDPR for $N_{\\rm iter} = 300$ rather than 150 and log $\\alpha$, the residual
RMS, and the recon RMS every 10 steps.  Three patterns can show up:

- **$\\alpha$ saturates below 1.0**: the small-signal model itself is the floor;
  more iterations cannot help, and the gap from $\\alpha_\\infty$ to 1 must be
  attributed to pipeline geometry (MFT sampling, resize interpolation, edge
  effects) rather than to the retrieval.
- **$\\alpha$ still climbing at 150**: 150 iterations is undersampling the
  fixed point; the production sweeps need to be rerun at a larger budget
  (try 500 or 1000) or with a tolerance criterion.
- **$\\alpha$ oscillates**: the cost landscape has multiple basins and the
  amplitude-projection loop is hopping between them; an MLE formulation
  (Section~\\ref{sec:discussion} of the paper) is the standard fix.""",

19: """## Noiseless Comparison: Pipeline Floor vs Noise Contribution

Repeat the same retrieval on noiseless PSFs.  The orthogonal $\\delta$ from the
clean run is the *pipeline floor* --- MFT sampling, resize interpolation, and any
even/odd grid asymmetries propagate through the noiseless forward model and end up
in $\\delta$ regardless of how many photons we collect.  Adding noise inflates
$\\delta$ in quadrature, so $\\delta_{\\rm noisy}^2 - \\delta_{\\rm clean}^2$
is the noise contribution.  This is the cleanest way to ask whether the residuals
in §3.5 of the paper are dominated by the pipeline or by the detector budget; if
$\\delta_{\\rm clean}$ is already comparable to the production residual floor,
expanding the photon budget will not help.""",

21: """## 4-Panel Decomposition Figure

Plot, side by side, the truth phase, the recon phase, the parallel component
$\\alpha \\phi_{\\rm truth}$, and the orthogonal remainder $\\delta$, all on the
same color scale.

- If panel 4 (the orthogonal $\\delta$) looks like high-frequency noise plus a
  thin ring around the aperture edge, the model is confirmed: residual energy
  is incoherent + edge-effect.
- If panel 4 shows coherent structure --- a tilted plane, a defocus bowl, or a
  sinusoid at a different $\\nu_0$ than truth --- there is a real
  cross-coupling worth chasing in a follow-up, because that energy is in
  principle removable by adjusting the basis or the diversity.""",

23: """## Frequency Cutoff Diagnostic: Where Does Recovery Break Down?

Sweep $\\nu_0$ from 2 to 40 cyc/aperture at fixed $A$ and $\\Delta z$ and track
$\\alpha$, the residual RMS, and the recon RMS at each frequency.  This is the
diagnostic that motivates the $\\nu_0 \\geq 5$ cyc/aperture lower bound on the
regime in which Section~\\ref{sec:discussion} of the paper reports the OTF / FDPR
correlation; below 5 cyc/aperture the sinusoid resembles a low-order Zernike that
the focused PSF resolves directly, and above the cutoff the retrieval just stops
working.

The Nyquist rule of thumb sets the absolute upper limit at 128 cyc/aperture for a
256-pixel pupil, but the *practical* FDPR cutoff is much lower --- expected near
15 cyc/aperture given the MFT focal-plane sampling and the `skimage` resize step
that downsamples the focal-plane onto the pupil grid.  The point of the sweep is
to measure that cutoff rather than estimate it.""",

26: """## Before / After Cutoff: Visual Comparison

Plot truth vs recon maps at five frequencies straddling the ~15 cyc/aperture
cutoff measured above ($\\nu_0 \\in \\{6, 10, 15, 20, 30\\}$).  The eye picks up the
failure mode that the scalar $\\alpha$ averages over: below the cutoff the recon
is a clean copy of the truth, at the cutoff it picks up an amplitude bias but the
spatial pattern is right, and above the cutoff the recon develops a
visibly-different frequency or collapses to noise.  This is the visual analog of
the per-frequency $\\alpha$ that the next plot shows quantitatively.""",

28: """## Map α(N_γ): Photon-Count Sweep

Hold $A = 0.10$ waves, $\\nu_0 = 10$ cyc/aperture, and $\\Delta z = 40$ mm fixed
and sweep $N_\\gamma$ over three decades.  The shot-noise variance scales as
$1/N_\\gamma$, so $\\alpha$ should rise monotonically with $N_\\gamma$ and
asymptote to a noise-floor value $\\alpha_\\infty$.

The point of the sweep is to distinguish two qualitatively different regimes:

- If $\\alpha_\\infty \\to 1$ as $N_\\gamma \\to \\infty$, then the residual at any
  finite $N_\\gamma$ is *all noise* and can be reduced by throwing photons at it.
- If $\\alpha_\\infty < 1$ at infinite photons --- we measure
  $\\alpha_\\infty \\approx 0.86$ for this operating point --- the residual
  splits into a noise-floor piece and a *pipeline floor* piece, and the gap from
  0.86 to 1.0 is geometric (MFT + resize + edge sampling) rather than budgetary.

In the second case, no amount of integration removes that 14\\% of the truth that
the pipeline is systematically losing.  Identifying which factor in the
geometric floor is dominant is the role of the three single-knob tests below.""",

31: """## Localize the Phase Offset (φ₀)

The recon sinusoid is phase-shifted relative to the truth (i.e.\\ the recovered
sinusoid peaks at a different $x$ than the injected one), which contributes
deterministically to the orthogonal $\\delta$ even at infinite photons.  Three
single-knob tests isolate the source.  Each varies one component of the
recon pipeline while holding the rest fixed; whichever one drives $\\phi_0$ to
zero is the offender.

1. **Bypass `skimage.resize`** --- interpolate from the MFT output to the pupil
   grid via HCIPy directly.  If $\\phi_0 \\to 0$, the resize step is shifting
   the recon by a fractional pixel.
2. **Vary `q`** (focal-plane oversampling factor) in
   `make_focal_grid` / `InstrumentConfiguration` (16 $\\to$ 32 $\\to$ 64).
   If $\\phi_0 \\propto q^{-1}$, the focal-plane sampling itself is the source.
3. **Pad-symmetrize the PSF** by averaging with its 180-deg rotation, forcing
   even-pixel centering.  If $\\phi_0 \\to 0$, the FFT centering convention
   (the standard fftshift / ifftshift asymmetry between even and odd grids) is
   the source.""",

33: """### Test 1: Bypass `skimage.resize` --- use HCIPy interpolation

Resample from the MFT focal-plane grid onto the pupil grid via
`scipy.interpolate.RegularGridInterpolator` instead of `skimage.transform.resize`.
The recon RMS and the measured $\\phi_0$ are the diagnostic outputs; a clean
$\\phi_0 \\to 0$ at fixed recon RMS indicates that resize was the offender.""",

35: """### Test 2: Vary `q` (focal-plane oversampling factor)

Rebuild the focal grid at three values of the oversampling factor $q \\in \\{16, 32, 64\\}$
and rerun the retrieval.  A $1/q$-scaling of $\\phi_0$ at fixed truth/diversity
operates as the signature: the focal-plane sampling cell width *is* the
quantization that produces the offset, and increasing $q$ makes the offset
proportionally smaller.""",

37: """### Test 3: Pad-symmetrize PSF (force even-pixel centering)

Replace each input PSF with $\\tfrac{1}{2}\\bigl(\\text{PSF} + \\text{rot180}(\\text{PSF})\\bigr)$.
The rotation removes any odd-symmetric component of the PSF without changing the
encoded aberration, so $\\phi_0 \\to 0$ under this transformation indicates that
the FFT centering convention is producing the offset.  The cost is a small
power loss at the bright pixel; the recon RMS will rise correspondingly if
this is the offender being patched rather than the root cause being fixed.""",

39: """## Confirm δ Has No Random Component

Run the *clean* recipe (noiseless PSFs) ten times with different seeds for the
FDPR phase initialization.  The forward model is deterministic in this
configuration, so any seed-dependent variation in $\\delta_{\\rm rms}$ is
iteration tie-breaking rather than physics.

- $\\delta_{\\rm rms}$ constant to $<$ 0.5 nm across seeds $\\Rightarrow$ $\\delta$
  is fully geometric (MFT + resize + edge), and the photon-count sweep above
  measures a *real* pipeline floor.
- $\\delta_{\\rm rms}$ varies by 1--2 nm across seeds $\\Rightarrow$ the amplitude-
  projection iteration is hopping between near-degenerate basins even without
  measurement noise, and the residual at any operating point carries a
  seed-uncertainty component that has to be marginalized over (or eliminated
  by moving to the Poisson-MLE formulation of \\citet{Paxman1992}, as
  discussed in Section~\\ref{sec:discussion} of the paper).""",
}


def main():
    nb = json.loads(NB.read_text())
    n_replaced = 0
    for idx, new_src in NEW.items():
        cell = nb["cells"][idx]
        assert cell["cell_type"] == "markdown", f"cell {idx} is {cell['cell_type']}"
        # Store source as a list of lines, with trailing newlines except the last.
        lines = new_src.splitlines(keepends=True)
        cell["source"] = lines
        n_replaced += 1
    NB.write_text(json.dumps(nb, indent=1))
    print(f"Updated {n_replaced} markdown cells in {NB}")


if __name__ == "__main__":
    main()
