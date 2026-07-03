# Surface renewal: theory and governing equations

This document collects the governing equations for the five sensible-heat (H)
estimators in `surface_renewal`, plus the free-convection fallback and the shared
Monin–Obukhov similarity theory (MOST) relations they rest on. Every equation
below has been checked against the implementation in `src/surface_renewal/`; the
module each lives in is named at the head of its section. Where the code carries a
"verify against the paper" caveat (the Castellví weighting factor), that caveat is
reproduced here explicitly.

Throughout, `ρ` is air density, `c_p` the specific heat of air at constant
pressure, `κ = 0.41` the von Kármán constant, `g = 9.81 m s⁻²`, and `T` a
high-frequency temperature series sampled at `hz` Hz.

## Contents

1. [Shared building blocks](#1-shared-building-blocks)
2. [Snyder (1996) cubic-ramp method](#2-snyder-1996-cubic-ramp-method--methodssnyderpy)
3. [Chen (1997) method](#3-chen-1997-method--methodschen97py)
4. [Flux–variance similarity (FVS)](#4-fluxvariance-similarity-fvs--methodsfvspy)
5. [Free-convection fallback](#5-free-convection-fallback--methodsfvspy)
6. [Castellví (2004) calibration-free method](#6-castellví-2004-calibration-free-method--methodscastellvipy)
7. [Wavelet ramp detection (Collineau & Brunet 1993)](#7-wavelet-ramp-detection-collineau--brunet-1993--methodswaveletpy)
8. [References](#8-references)

---

## 1. Shared building blocks

### 1.1 Structure functions — `structure.py`

For lag `Δt` (in samples `k`, so `Δt = k / hz`), the order-`p` temperature
structure function is

$$
S_p(\Delta t) = \big\langle\, (T(t+\Delta t) - T(t))^p \,\big\rangle,
$$

with the sign convention that **even** orders use the absolute increment
`⟨|ΔT|^p⟩` and **odd** orders keep the signed increment `⟨ΔT^p⟩`. The SR methods
use `S₂`, `S₃`, and `S₅`.

The optimal lag `Δt*` (equivalently `τ*`) is selected by

$$
\Delta t^* = \arg\max_{\Delta t}\ \frac{|S_3(\Delta t)|}{\Delta t},
$$

(`pick_optimal_lag`), which emphasises sharp ramps. A negative `S₃(Δt*)` indicates
warming ramps (gradual rise, sudden drop → upward heat flux, `H > 0`).

### 1.2 Temperature structure parameter C_T² — `structure.py`

In the inertial subrange the second-order structure function follows Kolmogorov
scaling,

$$
D_T(r) = C_T^2\, r^{2/3},
$$

where the **spatial** separation `r` is obtained from the time lag via Taylor's
frozen-turbulence hypothesis, `r = U · Δt`, with `U` the block-mean horizontal
wind speed. The 2/3 slope is held **fixed** (it is theoretical), so only the
intercept `log(C_T²)` is fitted:

$$
\log C_T^2 = \big\langle\, \log S_2 - \tfrac{2}{3}\log r \,\big\rangle .
$$

**Taylor hypothesis caveat.** `r = U·Δt` is only meaningful when the mean wind
advects a frozen turbulence field past the sensor. In near-calm conditions this
breaks down, so `estimate_CT2` returns `(nan, nan)` when `U ≤ 0.1 m s⁻¹`. The
default lag window is `0.05–0.5 s`, which assumes the inertial subrange of 10–20 Hz
data; very rough sites (short inertial subrange) may need a shorter window. The fit
also returns an `r²` against the fixed-slope prediction as a quality flag.

### 1.3 MOST universal functions — `most.py`

**Obukhov length:**

$$
L = -\frac{\rho\, c_p\, T_K\, u_*^3}{\kappa\, g\, H}.
$$

`L < 0` under unstable (daytime, upward-flux) conditions and `L > 0` under stable
conditions. Returns NaN when `H = 0` or `u* ≤ 0`. The stability parameter is
`ζ = z / L` (with `z = z_m`, the height above the zero-plane displacement).

**Temperature-gradient function φ_h (Businger–Dyer):**

$$
\phi_h(\zeta) =
\begin{cases}
(1 - 16\,\zeta)^{-1/2}, & \zeta < 0 \quad\text{(unstable)}\\[4pt]
1 + 5\,\zeta, & \zeta \ge 0 \quad\text{(stable)}
\end{cases}
$$

`φ_h(0) = 1`.

**Flux–variance function σ_T/|T\*| (Tillman 1972):** with `c₁ = 0.95`, `c₂ = 0.05`,

$$
\frac{\sigma_T}{|T_*|} =
\begin{cases}
c_1\,(c_2 - \zeta)^{-1/3}, & \zeta < -c_2 \quad\text{(unstable)}\\[4pt]
c_1\, c_2^{-1/3}, & \zeta \ge -c_2 \quad\text{(near-neutral / stable floor)}
\end{cases}
$$

The constant floor on the near-neutral/stable side avoids the singularity as
`ζ → c₂⁻` and reflects that flux–variance similarity for temperature is weak there.

**Structure-parameter function f_{C_T²} (Wyngaard et al. 1971):** with `c₁ = 4.9`,
`c₂ = 7.0`,

$$
\frac{C_T^2\, z^{2/3}}{T_*^2} = f_{C_T^2}(\zeta) =
\begin{cases}
c_1\,(1 - c_2\,\zeta)^{-2/3}, & \zeta < 0\\[4pt]
c_1\,\big(1 + c_2\,\zeta^{2/3}\big), & \zeta \ge 0
\end{cases}
$$

`f_{C_T²}(0) = c₁ = 4.9`.

All four MOST functions are scalar-in/scalar-out and return NaN on non-finite
input so they propagate NaNs predictably when mapped over arrays.

---

## 2. Snyder (1996) cubic-ramp method — `methods/snyder.py`

The classical SR model (Van Atta 1977; Snyder et al. 1996) treats the temperature
record as a succession of ramp structures of amplitude `A` and mean period `τ`,
recovered from the structure-function moments at `Δt*`.

**Ramp recovery** (`recover_ramp`): form the depressed cubic for the amplitude,

$$
A^3 + p\,A + q = 0,
\qquad
p = 10\,S_2 - \frac{S_5}{S_3},
\qquad
q = 10\,S_3,
$$

with `S₂, S₃, S₅` evaluated at `Δt*`. The cubic is solved in closed form
(Cardano / trigonometric branch, `_cardano_real_roots`), and **A is the maximum
real root**. The mean ramp period follows from the model relation

$$
\tau = -\frac{A^3\, \Delta t^*}{S_3(\Delta t^*)}.
$$

Only a **warming** ramp (`S₃(Δt*) < 0`) yields a positive `τ` under the
maximum-root convention; cooling ramps give a degenerate (NaN) recovery, so Snyder
is effectively a magnitude estimate for unstable/daytime conditions.

**Sensible heat** (`estimate_H_snyder`):

$$
H = \rho\, c_p\, \frac{A}{\tau}.
$$

This is **uncalibrated**; the historical empirical weighting factor `α` is left to
a downstream block-scale calibration against an EC reference (see
`Calibration.from_reference`). The same `(A, τ, Δt*)` recovery is reused by the
Castellví method (§6) so both see identical ramp characteristics.

---

## 3. Chen (1997) method — `methods/chen97.py`

Chen et al. (1997) scale the SR flux with the friction velocity `u*` and the
third-order structure function at the optimal lag.

**Friction velocity** from the rotated covariances (double or planar-fit rotation,
signals de-meaned per block):

$$
u_* = \left[\, \overline{u'w'}^2 + \overline{v'w'}^2 \,\right]^{1/4}.
$$

**Optimal lag** `τ*` is picked with the same `argmax |S₃(τ)| / τ` rule as Snyder,
scanning `0.2–8 s`.

**Sensible heat** (uncalibrated, dimensional scaling; `β` is a dimensionless
coefficient absorbed later by the `alpha` calibration):

$$
H = \rho\, c_p\, \beta\, u_*\,
     \frac{\operatorname{sign}(S_3)\,|S_3(\tau^*)|^{1/3}}
          {\tau^{\,2/3}}.
$$

In the implementation the denominator carries a small `+1e-12` guard against
division by zero, and the sign of `H` follows the sign of `S₃(τ*)` (so, unlike
Snyder, Chen retains directional information across stabilities). The overall scale
is refined by a block-scale `alpha` calibration against EC.

---

## 4. Flux–variance similarity (FVS) — `methods/fvs.py`

FVS infers `H` from the temperature standard deviation `σ_T` through MOST, without
any ramp model. Under unstable conditions (Tillman 1972; Katul et al. 1995),

$$
\frac{\sigma_T}{|T_*|} = c_1\,(c_2 - z/L)^{-1/3},
\qquad
T_* = -\frac{H}{\rho\, c_p\, u_*}.
$$

Because `L` itself depends on `H`, the relation is solved by fixed-point iteration
(`estimate_H_fvs`):

1. **Initial guess:** `|T*| ≈ σ_T / 2`, giving `H = sign · ρ c_p u* |T*|`.
2. **Iterate:** recompute `L = L(u*, T_K, H)`, then `ζ = z_m / L`, then invert the
   similarity function to update `|T*| = σ_T / (σ_T/|T*|)(ζ)`, and re-form
   `H = sign · ρ c_p u* |T*|`.
3. **Stop** when `|H_new − H| < tol` (default `1e-3`), up to `max_iter = 20`.

The `σ_T/|T*|` function and `L` are imported from `most.py` (§1.3) so FVS and the
other methods share one definition of the MOST relations.

**Sign.** Flux–variance similarity yields only `|H|`; the direction must be
supplied externally. The pipeline passes `sign_hint = sign(S3(tau*))` (a negative
`S₃` → warming ramps → upward flux). The returned `T* = −H / (ρ c_p u*)` therefore
carries the opposite sign to `H`.

**Guards.** A NaN, non-converged result is returned when `σ_T`, `u*`, or `T_K` is
non-finite, or when `u* < 0.01 m s⁻¹` (too weak for a reliable similarity
inversion). FVS is an **unstable-regime** method: the near-neutral/stable floor in
`σ_T/|T*|` makes it unreliable there.

---

## 5. Free-convection fallback — `methods/fvs.py`

In the strongly unstable, low-wind limit `−z/L ≫ 1`, `u*` drops out of the
surface-layer scaling and `H` can be recovered from `σ_T` alone
(`estimate_H_free_convection`):

$$
H_{fc} = \rho\, c_p\, c_{fc}\, \sigma_T^{3/2}\,\sqrt{\frac{g\, z_m}{T_K}}.
$$

**Derivation of `c_fc`.** Starting from the Tillman relation in its free-convection
limit (`−ζ ≫ c₂`, so the `c₂` offset is negligible),

$$
\frac{\sigma_T}{|T_*|} = c_1\,(-\zeta)^{-1/3},
\qquad
-\zeta = \frac{\kappa\, g\, z\, H}{\rho\, c_p\, T_K\, u_*^3},
\qquad
|T_*| = \frac{H}{\rho\, c_p\, u_*},
$$

substituting makes `u*` cancel exactly, leaving the closed form above with

$$
c_{fc} = c_1^{-3/2}\, \kappa^{1/2}.
$$

With the module's Tillman constant `c₁ = 0.95` and `κ = 0.41` this gives
`c_fc ≈ 0.69`. The code default is the higher, widely-cited practical value
`c_fc = 0.9`; it is exposed as a parameter so it can be replaced by the
site-specific `c₁^{−3/2}√κ` value or an empirical calibration.

**Dimensional check.** `√(g z_m / T_K)` has units `m s⁻¹ K^{−1/2}` and `σ_T^{3/2}`
has units `K^{3/2}`, so their product is `K m s⁻¹`; multiplying by `ρ c_p`
(`J m⁻³ K⁻¹`) gives `W m⁻²`. ✓

**Scope.** This returns the **magnitude** `|H| > 0` only; it is valid solely for
the unstable, upward-flux case and must **not** be applied at night. In the
pipeline it is enabled with `free_convection_fallback=True` and substitutes for a
u\*-based primary method on a block only when that block is both low-wind
(`u* < fc_ustar_max`, default `0.1`) and sufficiently unstable
(`ζ = z_m/L < fc_zeta_max`, default `−0.5`), and only when the primary `H` was
itself positive. The choice is recorded per block in `flux_method_used`.

---

## 6. Castellví (2004) calibration-free method — `methods/castellvi.py`

Castellví (2004) removed the need to calibrate the SR weighting factor `α` against
EC by deriving it analytically, combining SR analysis with MOST through the
dissipation of temperature variance. Classical SR writes

$$
H = \rho\, c_p\, \alpha\, \frac{A}{\tau},
$$

and for measurements in the inertial sublayer (above the roughness sublayer) the
production–dissipation balance of temperature variance yields the closed form

$$
\alpha = \sqrt{\frac{\pi\,\kappa\,z_m}{4\,\tau\,u_*\,\phi_h(\zeta)}}
$$

(`alpha_castellvi`). The radicand is dimensionless (`κ z_m` in metres, `τ u*` in
metres, `π/4` and `φ_h` dimensionless), so `α` is dimensionless — consistent with
the SR convention under which Snyder (§2) forms `H = ρ c_p (A/τ)` with no explicit
height factor.

**Iteration** (`estimate_H_castellvi`). The ramp `(A, τ)` is recovered with the
shared Van Atta cubic (§2, `recover_ramp`), then the stability loop starts from
neutral (`ζ = 0`, `φ_h = 1`):

$$
\alpha = \sqrt{\frac{\pi\,\kappa\,z_m}{4\,\tau\,u_*\,\phi_h(\zeta)}},
\qquad
H = \rho\, c_p\, \alpha\, \frac{A}{\tau},
\qquad
L = -\frac{\rho\, c_p\, T_K\, u_*^3}{\kappa\, g\, H},
\qquad
\zeta = \frac{z_m}{L},
$$

iterated until `|H_new − H| < tol` (default `1e-3`, up to `max_iter = 20`). The
ramp sign is carried by `A`, so `H` is signed naturally (`sign(H) = sign(A)`, since
`α, τ > 0`). Because `recover_ramp` selects the maximum real root, only warming
ramps (`S₃(τ*) < 0`, daytime unstable) give a valid `τ`; cooling ramps return a
NaN, non-converged result — exactly the regime for which Castellví's
inertial-sublayer derivation is intended. NaN results are also returned when `u*`,
`T_K`, or `z_m` is non-finite, or `u* < 0.01`, or `z_m ≤ 0`.

> **Verification caveat (carried from the code).** The inertial-sublayer form
> implemented here — the `√(π κ z_m / (4 τ u* φ_h))` weighting factor — was checked
> for dimensional consistency and against the ingredients of Castellví's
> derivation as reproduced in the secondary literature (the dissipation-method
> combination of SR with MOST involving `u*`, `φ_h`, `κ`, `z_m`, `τ`; e.g.
> Castellví & Snyder 2009, Mengistu & Savage 2010). The primary WRR (2004) PDF was
> not machine-readable during implementation, so the exact prefactor and the
> placement/exponent of `φ_h` should be confirmed against the published paper; **if
> the paper differs, prefer the paper and update the module.**

**Independent C_T² cross-check** (`alpha_from_CT2`, a QA aid, *not* part of the
flux calculation). Using the MOST scaling of the structure parameter (§1.3),

$$
T_*^2 = \frac{C_T^2\, z_m^{2/3}}{f_{C_T^2}(\zeta)},
$$

gives a second, independent route to the temperature scale. Comparing this `T*`
with the `T* = −H/(ρ c_p u*)` implied by the primary Castellví estimate flags
problems in the ramp recovery, the inertial-subrange `C_T²` fit, or the stability
estimate.

---

## 7. Wavelet ramp detection (Collineau & Brunet 1993) — `methods/wavelet.py`

SR theory models the record as coherent ramps separated by abrupt renewals.
Collineau & Brunet (1993) isolate these with the continuous wavelet transform
(CWT): at the scale matching the ramp duration, the wavelet coefficients change
sign across each renewal, so zero-crossings delimit events and the scale of maximum
wavelet variance measures the dominant ramp duration. The module implements this
with the **Mexican-hat (Ricker)** wavelet and plain NumPy convolution (no SciPy CWT
or PyWavelets dependency).

**Ricker wavelet** (unit-energy normalisation, so wavelet variance is comparable
across scales):

$$
\psi(t) = \frac{2}{\sqrt{3a}\,\pi^{1/4}}
          \left(1 - \left(\tfrac{t}{a}\right)^2\right)
          \exp\!\left(-\frac{t^2}{2a^2}\right).
$$

**Scale → period.** The Ricker wavelet is the second derivative of a Gaussian; its
power spectrum `|ψ̂(aω)|² ∝ (aω)⁴ e^{−(aω)²}` peaks over scale where `(aω)² = 2`,
i.e. `aω = √2`. A sinusoid of period `P` (`ω₀ = 2π/P`) therefore gives maximum
wavelet variance at `a_peak = √2 / ω₀`, so

$$
P = \frac{2\pi}{\sqrt{2}}\, a_\text{peak} = \pi\sqrt{2}\, a_\text{peak}
  \approx 4.443\, a_\text{peak}
$$

(`RICKER_PERIOD_FACTOR = 2π/√2`). This is the power-spectrum-peak match; it is close
to, but not identical with, Torrence & Compo's "equivalent Fourier period"
`2πa/√2.5` for the DOG-2 wavelet — either is defensible, and the code uses and
documents the analytically clean power-peak factor.

**Algorithm.**

1. Build `n_scales` (default 40) log-spaced scales over `scales_s` (default
   `1–120 s`, capped so the widest wavelet fits), transform the de-meaned series,
   and form the wavelet variance `W(a) = ⟨|c(a,t)|²⟩_t`.
2. `a_peak = argmax_a W(a)` gives the dominant ramp duration; the scale-based
   period estimate is `a_peak · RICKER_PERIOD_FACTOR`.
3. **Peak-significance test** (Collineau & Brunet 1993): a genuine dominant scale
   stands out above the roughly flat background; white noise does not. If
   `max(W)/median(W) < MIN_PEAK_PROMINENCE` (`= 3.0`), report the peak scale but
   **no** events and **no** flux.
4. At `a_peak` the coefficient series completes one full oscillation (a positive and
   a negative lobe) per ramp, so its zero-crossings come in pairs. Each *pair* of
   consecutive crossings bounds one full ramp; an event is kept when the
   coefficient extremum within the window exceeds one standard deviation of the
   coefficients. Each event's amplitude is the peak-to-peak range of the lightly
   denoised (moving-average) temperature over the event window.
5. `τ` is the mean inter-ramp period (`record length / n_ramps`) when `n_ramps ≥ 3`,
   otherwise the scale-based estimate.
6. `A` is the signed mean amplitude; the sign comes from the skewness of the
   temperature increments `dT` (a warming ramp — gradual rise, sudden drop — has
   negatively skewed increments and positive `A`).

**Sensible heat** (uncalibrated, same SR scaling as Snyder):

$$
H = \rho\, c_p\, \frac{A}{\tau}.
$$

---

## 8. References

- **Van Atta, C. W. (1977).** Effect of coherent structures on structure functions
  of temperature in the atmospheric boundary layer. *Archives of Mechanics*, 29,
  161–171. — Basis of the cubic ramp-recovery relations (§2).
- **Snyder, R. L., Spano, D., & Paw U, K. T. (1996).** Surface renewal analysis for
  sensible and latent heat flux density. *Boundary-Layer Meteorology*, 77(3–4),
  249–266. — Snyder cubic-ramp SR method (§2).
- **Chen, W., Novak, M. D., Black, T. A., & Lee, X. (1997).** Coherent eddies and
  temperature structure functions for three contrasting surfaces. Part I: Ramp
  model with finite microfront time. *Boundary-Layer Meteorology*, 84(1), 99–124.
  — Chen `u*`/`S₃(τ*)` scaling (§3).
- **Tillman, J. E. (1972).** The indirect determination of stability, heat and
  momentum fluxes in the atmospheric boundary layer from simple scalar variables
  during dry unstable conditions. *Journal of Applied Meteorology*, 11(5),
  783–792. — Flux–variance similarity function and free-convection limit (§§1.3, 4, 5).
- **Katul, G. G., Goltz, S. M., Hsieh, C.-I., Cheng, Y., Mowry, F., & Sigmon, J.
  (1995).** Estimation of surface heat and momentum fluxes using the flux-variance
  method above uniform and non-uniform terrain. *Boundary-Layer Meteorology*,
  74(3), 237–260. — FVS above heterogeneous terrain (§4).
- **Castellví, F. (2004).** Combining surface renewal analysis and similarity
  theory: A new approach for estimating sensible heat flux. *Water Resources
  Research*, 40, W05201. doi:10.1029/2003WR002677. — Analytic (calibration-free)
  SR weighting factor (§6).
- **Castellví, F., & Snyder, R. L. (2009).** Combining the dissipation method and
  surface renewal analysis to estimate scalar fluxes from the low frequency of
  scalar high-frequency measurements. *Journal of Hydrology*, 373(3–4), 142–151.
  doi:10.1016/j.jhydrol.2009.04.020. — Dissipation-method combination underpinning
  the Castellví `α` (§6).
- **Wyngaard, J. C., Izumi, Y., & Collins, S. A. (1971).** Behavior of the
  refractive-index-structure parameter near the ground. *Journal of the Optical
  Society of America*, 61(12), 1646–1650. — MOST scaling of `C_T²` (§§1.2, 1.3, 6).
- **Collineau, S., & Brunet, Y. (1993).** Detection of turbulent coherent motions
  in a forest canopy. Part II: Time-scales and conditional averages.
  *Boundary-Layer Meteorology*, 66(1–2), 49–73. — Wavelet ramp detection (§7).

### Supporting references

- **Businger, J. A., Wyngaard, J. C., Izumi, Y., & Bradley, E. F. (1971).**
  Flux-profile relationships in the atmospheric surface layer. *Journal of the
  Atmospheric Sciences*, 28(2), 181–189. — `φ_h` (Businger–Dyer form) and `L` (§1.3).
- **Dyer, A. J. (1974).** A review of flux-profile relationships. *Boundary-Layer
  Meteorology*, 7(3), 363–372. — `φ_h` (§1.3).
- **Torrence, C., & Compo, G. P. (1998).** A practical guide to wavelet analysis.
  *Bulletin of the American Meteorological Society*, 79(1), 61–78. — Wavelet
  equivalent-Fourier-period comparison (§7).
- **Mengistu, M. G., & Savage, M. J. (2010).** Surface renewal method for
  estimating sensible heat flux. *Water SA*, 36(1), 9–18. — Secondary reference used
  to cross-check the Castellví weighting factor (§6).
