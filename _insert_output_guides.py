"""
Insert interpretation-guideline markdown cells immediately after each
code cell that produces a notable output.  Inserts are done in reverse
index order so the earlier indices remain stable as we go.
"""
import json
from pathlib import Path

NB = Path("/Users/joshuapotter/Documents/SEAL/FDPRNotebooks/april28_test.ipynb")

# Mapping: code-cell-index  ->  markdown text to insert right after it.
# Indices are validated against the actual cell types before insertion.
GUIDES = {
10: """### Reading the cost-function plot

Both `+Δz` and `-Δz` curves should drop monotonically and plateau.

- **Smooth monotonic decrease, asymptote within ~50 iter:** the iteration is at its fixed
  point; you are in the regime the rest of this notebook assumes.
- **Curve still descending steeply at 150 iter:** the budget is too small; either raise
  `N_ITERS` or move to a tolerance-based stop (identified as future work in the paper).
- **Oscillations or non-monotonic descent:** the amplitude-projection loop is bouncing
  between near-degenerate basins. The seed test at the end of the notebook will tell you
  whether this is reproducible across initializations.
- **`-Δz` curve much higher than `+Δz`:** asymmetric encoding in the diversity pair.
  Re-check the sign convention on `phi_df`; the Dean-Bowers theory assumes
  $|\\Delta z_+| = |\\Delta z_-|$.""",

12: """### Reading `truth_rms / recon_rms / resid_rms`

Three numbers, in nm of wavefront error.  The relative ordering tells the failure mode:

- **`resid_rms < recon_rms < truth_rms`:** retrieval beats the null, in a healthy
  amplitude-bias regime ($\\alpha > 0.5$).  Production sweeps should produce this.
- **`recon_rms < truth_rms` and `resid_rms > recon_rms`** (the case here, 22 vs 27 nm
  with truth 46 nm): geometric consequence of $\\alpha < 0.5$.  Not a bug; the next cell
  separates the bias piece from the noise piece to prove it.
- **`recon_rms ≈ 0` and `resid_rms ≈ truth_rms`:** retrieval has collapsed to a near-zero
  phase ($\\alpha \\to 0$).  Either the noise budget is too thin, or the operating point
  is past the cutoff documented by the frequency sweep further down.
- **`resid_rms > truth_rms`:** retrieval is *worse* than the null.  This happens at
  $A \\sim \\pi$ rad (Section~\\ref{sec:mc_sweeps} of the paper) when the small-signal
  expansion no longer describes the focal-plane intensity.""",

14: """### Reading the alpha decomposition

The point of this cell is the quadrature check at the bottom: the measured residual
RMS should equal $\\sqrt{((1-\\alpha)\\phi_{\\rm truth})^2 + \\delta^2}$ to within
floating-point precision.  Once that check passes, $\\alpha$ alone tells you which
failure mode you are in.

| $\\alpha$ value | Interpretation |
|----|----|
| $\\approx 1.0$ | Truth fully recovered; residual is pure noise + pipeline floor. |
| $\\approx 0.85$ | **Pipeline ceiling** -- $\\alpha_{\\rm clean}$ for this operating point. Even infinite photons cannot exceed this; the gap from 0.85 to 1.0 is MFT/resize/centering. |
| $0.5 - 0.85$ | Noise-induced amplitude bias.  More photons (or a Poisson-MLE algorithm) will raise $\\alpha$; pipeline-floor still applies. |
| $\\approx 0.5$ | The geometric crossover: $\\text{resid\\_rms} = \\text{recon\\_rms}$.  No deeper meaning, but useful to anchor the plots. |
| $0 - 0.5$ | Severe noise-induced bias.  Recon is a faithful-but-attenuated copy of the truth. |
| $\\approx 0$ | Past the cutoff.  Recon is uncorrelated with truth. |
| Negative | Recon is *anti-correlated* with truth.  Almost always means a sign or basin-flip in the diversity, not signal. |

The split between $\\alpha\\,\\phi_{\\rm truth}$ and $\\delta$ is the diagnostic that lets you
ask "is more exposure going to help?" -- the answer is *only* if $\\alpha$ is what's killing
you and you are still on the rising part of the photon-count curve.""",

16: """### Reading the tip / tilt / defocus split

The orthogonal $\\delta$ is split into low-order coherent modes (tip, tilt, defocus) and
a "high-frequency residual" lump.  What you want to see is **the low-order pieces being
small** relative to the high-frequency lump.

- **tt rms / total $\\delta$ rms $\\lesssim 10\\%$ (the case here, 0.43/9.09):** clean.
  No pixel-centering inconsistency between forward model and recon.  Move on.
- **tt rms $\\gtrsim$ 5 nm and growing with frequency:** the forward model and the recon
  disagree on where the pupil center is.  Inspect `pupil_grid` and the MFT-reverse
  step.  This often shows up alongside a non-zero $\\phi_0$ (see end of notebook).
- **Defocus rms $\\gtrsim$ 5 nm:** the diversity defocus is leaking into the unknown
  phase, i.e.\\ the sign degeneracy is not being broken cleanly.  Either $|\\Delta z|$ is
  too small for the current $\\nu_0$ (consult Section~\\ref{sec:otf_heatmap}), or the
  $+\\Delta z / -\\Delta z$ pair is asymmetric.
- **High-frequency residual dominant:** expected.  This is incoherent noise plus
  pixel-scale edge effects, and is what the $\\delta$ piece is supposed to absorb.""",

18: """### Reading the iteration sweep

Three diagnostic patterns to look for in the $\\alpha$-vs-iteration plot:

- **$\\alpha$ plateaus within 100-150 iter:** the 150-iter budget the paper uses is fine
  for this operating point.  The fixed point is reached.
- **$\\alpha$ still climbing at iter 300:** the budget is undersampled.  Either raise
  `N_ITERS` for production sweeps, or move to a tolerance-based stop.  The 150-iter
  budget will produce numerically-different residuals than a 300- or 500-iter budget.
- **$\\alpha$ oscillates between two values:** the cost landscape has two near-degenerate
  basins and the amplitude-projection loop is hopping between them.  The seed test at the
  end of the notebook tells you whether the oscillation is reproducible (basin choice
  determined by initialization) or genuine non-convergence.  In either case the
  Poisson-MLE formulation (Section~\\ref{sec:discussion} of the paper) is the standard fix.

The recon RMS and residual RMS plots are the same information in physical units: as
$\\alpha$ rises toward its asymptote, the recon RMS approaches the truth RMS and the
residual approaches the orthogonal-only floor.""",

20: """### Reading the noiseless vs noisy delta breakdown

This is the most consequential cell in the notebook for interpreting the
production residuals.  Three numbers come out:

- **`alpha (clean)` vs `alpha (noisy)`:** the gap is the noise-induced amplitude
  bias.  A 0.86 -> 0.43 collapse (as here) means the noise is robbing about half of the
  amplitude even though it is invisible in the orthogonal direction.  This is a known
  failure mode of amplitude-projection algorithms and is one of the reasons the paper
  identifies a Poisson-MLE rewrite as future work.
- **`delta_rms (clean)`:** the **pipeline floor**.  MFT sampling, `skimage` resize, and
  any odd/even centering asymmetries propagate through the noiseless forward model and
  are stuck in $\\delta$ regardless of photon count.  Production residuals can never go
  below this number at this operating point.
- **`delta_rms (noise)` via quadrature:** if this is $\\approx 0$ (as here, 0.0 nm), the
  detector noise is invisible in the orthogonal direction at the current $N_\\gamma$ and
  $\\sigma_r$.  Adding more photons will fix $\\alpha$ but will not reduce $\\delta$ further.
  If the quadrature is large compared to `delta_clean`, you are in the photon-noise-limited
  regime and lowering $\\sigma_r$ or raising $N_\\gamma$ both help.

**Practical rule:** if `delta_clean` $\\gtrsim$ `truth_rms / 10`, the pipeline floor is the
dominant residual and the bottleneck is the recon-side propagation, not the data.""",

22: """### Reading the four-panel decomposition figure

This is a sanity check on the alpha decomposition rather than a quantitative tool.
The four maps are on a common color scale so they can be compared directly.

- **Panel 1 (truth):** the injected sinusoid.  Visible structure depends on $\\nu_0$ and
  $A$, but should look like a clean grating.
- **Panel 2 (recon):** the algorithm output.  At healthy operating points it should be
  a visibly attenuated version of panel 1 with no extra features.
- **Panel 3 ($\\alpha\\cdot$ truth):** what the recon would look like if it were a *pure*
  amplitude-biased copy of the truth.  This is the "model prediction" for the recon.
- **Panel 4 ($\\delta$, orthogonal):** the part of the recon that is *not* a scaled copy
  of the truth.  This is the diagnostic panel.

  - **Looks like incoherent noise with a thin ring at the aperture edge:** model
    confirmed.  Residual energy is pipeline floor + noise, no coherent leakage.
  - **Has a visible tilted plane or a defocus bowl:** modal cross-talk into low-order
    Zernikes.  Inspect the tip/tilt/defocus split numbers above; the projection should
    have caught it.
  - **Has a visible sinusoid at a different $\\nu_0$ than truth:** aliasing.  Either the
    truth is near or past the practical cutoff, or the MFT grid is undersampling the
    defocused PSF.""",

24: """### Reading the alpha-vs-frequency table

This is the diagnostic that pins down the practical FDPR cutoff at this aberration
amplitude, photon budget, and read noise.  Three regimes are visible:

| $\\alpha$ range | Regime | What it means |
|----|----|----|
| $\\alpha > 0.85$ | Pipeline-ceiling regime | Algorithm at the geometric limit; residual is mostly the pipeline floor.  Operating points to use for cross-validation. |
| $0.4 < \\alpha < 0.85$ | Noise-bias regime | Recon recovers the truth direction with severe amplitude bias.  More photons will move you up this band. |
| $|\\alpha| < 0.2$ | Past-cutoff regime | Recon decorrelates from truth.  Recon RMS $\\to$ noise floor (~5 nm), residual RMS $\\to$ truth RMS. |
| $\\alpha < 0$ | Anti-correlated | Recon is the *negative* of truth.  Algorithm picked the wrong basin; almost always a sign issue with the diversity. |

The transition is **sharp** -- $\\alpha$ drops from 0.88 at $\\nu_0=8$ to 0.43 at
$\\nu_0=10$, and from 0.57 at $\\nu_0=15$ to $-0.12$ at $\\nu_0=16$.  These two thresholds
($\\nu_0 \\approx 8$ and $\\nu_0 \\approx 15$) bound the band in which a noisy single-cell
retrieval has any chance of producing the right answer at this $\\Delta z$.  Below 8
c/aperture and above 15 c/aperture, the per-cell retrieval is decoupled from the OTF
sideband prediction for the reasons discussed in Section~\\ref{sec:discussion} of the paper.""",

27: """### Reading the before/after-cutoff figure

Visual analog of the table above.  Look at each column:

- **Below the cutoff (e.g.\\ $\\nu_0 = 6$):** recon panel should be a clean shrunk copy
  of the truth panel.  No phase shift, no broken-up structure.
- **In the bias regime (e.g.\\ $\\nu_0 = 10$):** recon still has the right grating
  orientation and period, but the amplitude is visibly lower.  $\\alpha$ tells you the
  exact ratio.
- **At the cutoff ($\\nu_0 = 15$):** recon is still recognizable but starting to develop
  phase-shifted patches.  $\\alpha$ is around 0.5 and the residual map has noticeable
  coherent structure.
- **Past the cutoff ($\\nu_0 = 20$, 30):** recon panel looks like noise.  No phase
  relationship with truth; $\\alpha \\to 0$.

If a panel **above** the cutoff shows visible structure that *isn't* truth (e.g.\\ a
sinusoid at half the expected period), that is aliasing -- the recon is locking onto a
beat frequency between the truth and the focal-plane sampling, not retrieving the input.""",

29: """### Reading the photon-count sweep

Two qualitatively different behaviors can show up:

- **$\\alpha$ rises and saturates well below 1.0** (the case here, $\\alpha_\\infty \\approx 0.85$): the
  pipeline ceiling dominates over noise.  Increasing $N_\\gamma$ beyond the saturation
  point yields no improvement; the only way to push past the ceiling is to change
  the recon-side propagation.
- **$\\alpha$ rises toward 1.0 without obvious saturation:** the pipeline ceiling is
  at or above 1.0 for this operating point (i.e.\\ the geometric residual is negligible),
  and the only limit is noise.  More photons will keep helping.

The $\\delta$ curve is the complementary diagnostic.  Two behaviors:

- **$\\delta_{\\rm rms}$ floors at a finite value (~10 nm here):** that floor is the
  pipeline floor.  More photons do not reduce it.
- **$\\delta_{\\rm rms}$ keeps falling as $N_\\gamma$ grows:** noise is still dominant
  in the orthogonal direction.  Photon-noise-limited regime.

**Reading the SNR column:** SNR is the peak-pixel signal-to-noise of the focused PSF
under the quadrature variance budget.  Operating points with SNR $\\lesssim$ 10 are
shot-noise-dominated and almost always produce $\\alpha < 0.2$ regardless of operating
point.  SNR $\\gtrsim$ 50 is where the bias regime becomes navigable.""",

30: """### Reading the alpha-vs-photons figure

Two panels, both log-x:

- **Left ($\\alpha$ vs $N_\\gamma$):**  Look for the dashed *clean-ceiling* line.  If the
  curve asymptotes *to* it, the pipeline is saturated and adding photons buys nothing.
  If the curve is still climbing toward it, you have room.  The dashed line at $\\alpha = 0.5$
  is just the geometric crossover (where `resid_rms = recon_rms`).
- **Right ($\\delta_{\\rm rms}$ vs $N_\\gamma$, log-log):** the *clean floor* line is the
  pipeline floor.  A curve that floors at it is pipeline-limited; a curve that stays well
  above it (with negative slope) is noise-limited.  A $1/\\sqrt{N_\\gamma}$ slope is the
  photon-shot-noise signature.""",

32: """### Reading the baseline $\\phi_0$

$\\phi_0$ is the phase shift (degrees) between the recon sinusoid and the truth sinusoid
at the same $\\nu_0$, projected against the truth's sin/cos basis.

- **$|\\phi_0| < 5°$:** subdominant; can be left alone for now.
- **$5° \\leq |\\phi_0| < 20°$:** correctable systematic.  Means the recon is producing the
  right grating at the wrong phase.  Worth fixing because it deterministically inflates
  $\\delta$.
- **$|\\phi_0| > 30°$:** severe coupling.  Either a pixel-centering bug or a basin-flip
  in the iteration; the three knob tests below isolate which.
- **$|\\phi_0| \\to 90°$ or 180°:** sign flip in the recon, not a phase shift.  Almost
  always means the diversity polarity is wrong somewhere.""",

34: """### Reading Test 1 (bypass `skimage.resize`)

This compares `phi_0` and `alpha` with `skimage.transform.resize` replaced by
`scipy.interpolate.RegularGridInterpolator` (cubic).  The forward model and the FDPR
iteration are unchanged; only the *final downsampling step* from the MFT focal grid back
to the pupil grid differs.

- **`phi_0` drops to near zero, `alpha` unchanged:** resize was bin-snapping the recon
  by a fractional pixel.  Use scipy interpolation in production.
- **`phi_0` unchanged, `alpha` unchanged** (the case here, both at 6.6°): the resize step
  is *not* the source of the offset.  The offset is upstream in the MFT or in the FFT
  centering.
- **`alpha` changes by more than a few percent:** the resize step was distorting the
  amplitude itself, not just the phase.  Cubic interp gives a different power normalization
  than linear; check that one of the two is closing the energy budget.""",

36: """### Reading Test 2 (vary `q`)

If focal-plane sampling is the source of the phase offset, $\\phi_0$ should scale as
$1/q$ (doubling $q$ should halve the offset).  This test sweeps $q \\in \\{16, 32, 64\\}$.

- **$\\phi_0$ halves each time you double $q$:** focal-plane sampling is the cause.
  Raising $q$ in production buys you a proportionally smaller offset; the trade-off is
  PSF array size (memory + speed).
- **$\\phi_0$ unchanged with $q$:** sampling is *not* the cause.
- **$\\phi_0$ *grows* with $q$** (the case here, 1.3° -> 8.4° -> 12.0°): the offset is
  coupled to the PSF array shape itself, not to the sampling pitch.  This is the signature
  of an even/odd-pixel centering convention that flips between $q$ values: when the PSF
  shape is $(2q \\cdot N_{\\rm airy})^2$, going from $q=16$ to $q=32$ doubles the array
  dimension, and the FFT centering convention treats odd vs even differently.  The
  resulting offset is *not* a 1/q quantization but a parity effect.

This is the test that proved the phase offset has a non-obvious origin in this notebook
-- the standard explanation (focal-plane sampling) was ruled out.""",

38: """### Reading Test 3 (pad-symmetrize PSF)

This averages each input PSF with its 180° rotation, forcing even-pixel centering by
construction.  The forward model is unchanged; only the data being passed into FDPR
have had any odd-symmetric component removed.

- **$\\phi_0$ drops to near zero, `alpha` roughly preserved:** FFT centering was the
  cause.  Pad-symmetrize in production, or rewrite the propagation to use a centered
  even grid throughout.
- **$\\phi_0$ drops but `alpha` also drops by a few percent:** symmetrization is removing
  real information.  The phase offset *is* coming from FFT centering, but the cure costs
  amplitude.  Worth doing only if $\\phi_0$ is dominating the residual.
- **$\\phi_0$ blows up and `alpha` collapses or inverts** (the case here, 117.5° and
  $\\alpha = -0.054$): the PSF's odd-symmetric component is *carrying the sign of the
  aberration*, and removing it destroys the recon.  This is unique to defocus diversity:
  the $+\\Delta z$ and $-\\Delta z$ PSFs are mirror images of each other in the
  small-signal limit, and the *difference* between them is what encodes the unknown
  phase.  Pad-symmetrize kills that difference.

Net result of Tests 1-3: **the phase offset survives all three single-knob tests.**
Its origin is some coupling between FFT centering, PSF asymmetry, and MFT-reverse
grid choice that is not isolable by any one of these knobs alone.  Resolving it requires
rebuilding the recon-side propagation on a fully-centered grid from the pupil out,
which is more work than this notebook scopes.""",

40: """### Reading the multi-seed reproducibility test

Ten different random-phase initializations on *noiseless* PSFs.  Each statistic tells
you something different about the algorithm:

- **`alpha` std $< 0.01$** (the case here, $\\sigma_\\alpha = 0.008$): the amplitude bias
  is essentially deterministic.  Whatever ceiling is in place doesn't depend on the
  initialization.
- **`alpha` std $> 0.02$:** the iteration is genuinely seed-dependent.  Production
  sweeps must average over at least 5 seeds, ideally 10.
- **`delta_rms` std $< 0.5$ nm:** $\\delta$ is fully geometric.  No multi-trial averaging
  needed for that component.
- **`delta_rms` std $\\sim 1$-2 nm** (the case here, $\\sigma_\\delta = 1.2$ nm): there is
  a small but non-negligible seed-uncertainty floor.  The amplitude-projection loop has
  multiple near-degenerate fixed points and the initialization picks among them.  Five
  trials/cell is enough to characterize this; one trial/cell will be slightly biased.
- **`delta_rms` std $> 5$ nm:** the cost landscape has genuinely-different basins.
  Either run more trials, or move to a deterministic (MLE) initialization.
- **`phi_0` std $> 10°$:** the phase offset is itself a seed-dependent artifact rather
  than a deterministic systematic.  Reduces the credibility of any single-trial $\\phi_0$
  measurement; you need to average across seeds.

**Verdict for this notebook**: the orthogonal $\\delta$ has a small (~1 nm) random
component that varies trial-to-trial; the rest of $\\delta$ is deterministic pipeline
floor.""",
}


def main():
    nb = json.loads(NB.read_text())
    cells = nb["cells"]

    # Validate every target is a code cell and the *next* cell is not already
    # one of our markdown guides (so re-running is idempotent).
    for idx in GUIDES:
        if cells[idx]["cell_type"] != "code":
            raise SystemExit(f"cell {idx} is {cells[idx]['cell_type']}, expected code")

    # Insert in reverse-index order so earlier indices stay stable.
    for idx in sorted(GUIDES.keys(), reverse=True):
        new_md = {
            "cell_type": "markdown",
            "metadata": {},
            "source": GUIDES[idx].splitlines(keepends=True),
        }
        # Idempotency: if the next cell already contains "### Reading", skip.
        if idx + 1 < len(cells) and cells[idx + 1]["cell_type"] == "markdown":
            src = "".join(cells[idx + 1].get("source", []))
            if src.startswith("### Reading"):
                cells[idx + 1] = new_md  # overwrite the existing guide
                continue
        cells.insert(idx + 1, new_md)

    NB.write_text(json.dumps(nb, indent=1))
    print(f"Inserted/updated {len(GUIDES)} guide cells; notebook now has {len(cells)} cells.")


if __name__ == "__main__":
    main()
