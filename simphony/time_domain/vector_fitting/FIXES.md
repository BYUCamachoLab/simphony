# Vector-fitting fixes (z_domain.py, s_domain.py) — 2026-11-02

These came up while recreating the z-vs-s comparison of Jiang & Ye, Opt. Laser Technol. 175 (2024) 110718
(`time_domain_paper/z_vs_s/jiang_lattice_comparison.ipynb`). They were first made on branch `instantiated_circuit`
(`../simphony`) and then ported to this `multirate` worktree, which already had #1 (z_domain) and #3. Each change is
small and local. Nothing is committed; review with `git diff` here.

Status on this branch: **#1 (z_domain) and #3 were already present**; everything else was applied here and verified
with si_nitride's venv (identical numbers to the other branch: z K = 44 MAE −53.02 dB, s −48.29 dB in THz and Hz).

| # | File | Problem | Change |
|---|------|---------|--------|
| 1 | s_domain (z_domain already had it) | O(N²) memory in `_lstsq_matrices` | row scaling instead of a dense `diag` |
| 2 | z_domain | stability flip `1/p` moves the pole's frequency | `1/conj(p)` |
| 3 | z_domain | float64 depended on import order | already present on multirate; nothing changed |
| 4 | z_domain | returned MSE ignored `sign_convention` | pass it through |
| 5 | s_domain | odd `model_order` crashed | add one real starting pole |
| 6 | s_domain | only square S matrices supported | handle q outputs × m inputs, like z_domain |
| 7 | performance.py | old cached fits would be reused | `CACHE_VERSION` "v1" → "v2" |
| 8 | s_domain | `vector_fitting` failed with frequency in Hz | normalize frequency internally, rescale results |
| 9 | s_parameters.py | (new) s-domain models in circuits | `fitter="s_bilinear"` |
| 10 | both + new common.py | s and z interfaces differed | unified s interface; shared helpers; full-QR least squares |
| 11 | both + common.py | (new) no continuous-time solver; no state-space discretization | `state_space_response_continuous`; `continuous_to_discrete_state_space`; `bilinear_transform` moved to z_domain |
| 12 | new passivity.py, s_parameters.py | (new) z fits are not checked for passivity anywhere on the unit circle | `passivity_assessment_discrete`; `enforce_passivity_discrete` (cutting planes); `vector_fitting_parameters["passivity"]` |

## 1. Memory: dense diagonal in `_lstsq_matrices` (both files)

**What.** For every S element the SK system was built as `-jnp.diag(H) @ phi1`. `jnp.diag(H)` is a dense N×N complex
matrix (N = number of frequency points): 256 MB at N = 4000, 400 MB at 5000, and 40 GB at 50 000. This is what ran the
machine out of memory while the notebook was executing.

**Fix.** `diag(H) @ phi1` equals `H[:, None] * phi1`, so the dense matrix is never formed. On multirate,
`z_domain._lstsq_matrices` already did this (with a Gram–Schmidt reduction against phi0); only `s_domain` changed.

**Verified.**
- M and B agree with the old formula to 5e-12 (QR round-off).
- Peak RSS (K = 100, 2×2 S): z at N = 4000 dropped from 1.47 GB to 0.46 GB; N = 20 000 now takes 0.82 GB (z) and
  1.16 GB (s). Memory is now linear in N.
- This is also faster: the old product was O(N²K) per element.

`s_domain._full_lstsq_matrices` still uses `jnp.diag`. It is not called anywhere (only mentioned in a comment), so I
left it alone.

## 2. z_domain: unstable poles reflected with `1/p` instead of `1/conj(p)`

**What.** `vector_fitting_discrete` forces stability with `p → 1/p` for |p| > 1. For p = r·e^{jθ}, 1/p = (1/r)·e^{−jθ}:
the radius is inverted, but the angle, i.e. the pole's frequency, is negated too. For real-coefficient (passband)
models the poles come in conjugate pairs, so the set is unchanged and the bug is invisible. Baseband photonic models
are complex and **not** conjugate-paired, so the flipped pole jumps from +f to −f.
`1/conj(p) = (1/r)·e^{jθ}` is the reflection across the unit circle that keeps the frequency. It is the z-domain
counterpart of the s-domain fix `Re(p) → −Re(p)`, which already keeps Im(p).

**Fix.** `jnp.where(mask, 1 / jnp.conj(current_poles), current_poles)`.

**Verified.** Single resonance at +0.2·f_s whose pole lies outside the unit circle (6 poles, 10 iterations):
- with `1/p`, the flipped poles collected at −0.2·f_s;
- with `1/conj(p)`, a pole sits at +0.2·f_s.

On causal data (the lattice and ring examples, including noisy and over-fitted cases) no pole ever left the unit
circle, so results there are unchanged. The bug therefore did **not** cause the colleague's "z is not more accurate"
result. It matters when data are noisy, anti-causal (for example a sign-convention mix-up), or delay-dominated.

## 3. z_domain: double precision depended on import order

**What.** `s_domain.py` runs `jax.config.update("jax_enable_x64", True)` at import time, but `z_domain.py` did not.
A script that imports only `z_domain` therefore fitted in complex64. A fresh interpreter showed
`jax_enable_x64 == False` after importing only z_domain. Errors near −54 dB are close to what single precision can
resolve in these ill-conditioned least-squares problems, so a z-vs-s comparison could depend on which module happened
to be imported first.

**Fix (other branch only; multirate already had it).** The same one-line `jax.config.update("jax_enable_x64", True)` at the top of `z_domain.py`, matching
`s_domain.py`. (Changing global JAX config inside a library is debatable, but it is the existing convention here, and
both modules now behave the same way.)

## 4. z_domain: `_mean_squared_error` ignored `sign_convention`

**What.** It always evaluated the model with `sign_convention=PHYSICIST`, whatever convention the fit used. The error
returned by `vector_fitting_discrete(..., sign_convention=ENGINEER)` was therefore computed against the mirrored model
and was meaningless.

**Fix.** Pass `sign_convention` through.

**Verified.** Fitting S (physicist) and conj(S) (engineer) now report the same MSE: 2.785e-08 for both.

## 5. s_domain: odd model order crashed

**What.** `_initial_poles` builds `model_order // 2` conjugate pairs, so an odd order produced one pole too few. The
eigenvalue step then had mismatched shapes (for example K = 43, the paper's value).

**Fix.** For odd orders, append one real pole −0.1·2π·max|f|, damped like the complex pairs.

**Verified.** `vector_fitting(43, ...)` returns 43 poles.

## 6. s_domain: only square transfer matrices

**What.** `_lstsq_matrices` and `_fit_to_poles` assumed `num_ports × num_ports`, while z_domain already handles q
outputs × m inputs. A 4×1 S failed.

**Fix.** Use `shape[1]` (outputs) and `shape[2]` (inputs), with the same row-major (output, input) ordering as before,
so square inputs give identical results.

**Verified.** A 4×1 fit returns residues of shape (44, 4, 1).

## 7. Cache version bump (performance.py)

`vector_fitting_discrete` and `optimize_order_vector_fitting_discrete` are wrapped in `persistent_cache`, keyed on
the inputs, not the code. Without a bump, fits cached before fixes 2 and 4 would keep being returned from the many
`.simphony_cache` folders across the projects. `CACHE_VERSION` "v1" → "v2" makes them recompute. No cache files were
deleted; the old ones are simply ignored and can be removed by hand.

## 8. s_domain: `vector_fitting` depended on the frequency unit

**What.** The s-domain basis is $1/(j2\pi f - p)$ next to a column of ones. With f in Hz (~10¹³) those columns are
~10⁻¹⁴, and the SK least-squares problems become numerically singular. On the paper-matched lattice (K = 44,
10 iterations) the same data gave an MAE of −48.3 dB with frequency in THz or GHz, but **+1.1 dB with frequency in
Hz**, a complete failure with no warning. Callers had to know to pre-scale frequency (the notebooks pass THz or
f/f_scale). The z domain does not have this problem because it only uses f/f_s.

**Fix.** `vector_fitting` divides frequency by c = max|f| before fitting, then returns poles·c and residues·c, using
$r/(j2\pi f/c - p) = cr/(j2\pi f - cp)$; the feedthrough and the returned MSE are unchanged. Results come back in the
caller's units, so code that already passes THz is unaffected. Only `vector_fitting` changed; the private helpers
still work in whatever units they are given.

**Verified.** The MAE is −48.3 dB with frequency in THz, GHz and Hz.

## 9. New: `fitter="s_bilinear"` for sample- and block-mode elements (2026-11-03)

This is an addition, not a bug fix. It lets the same circuit be run with s-domain models for the z-vs-s comparison.

**What.** `vector_fitting_parameters` takes a new key, `fitter`:
- `"z"` is the default, so existing code is unchanged.
- `"s_bilinear"`, in `_calculate_state_space_coefficients_from_sax_model` (`libraries/ideal/s_parameters.py`):
  1. Fits the delay-compensated data with `s_domain.vector_fitting_continuous` (centre `f_center`, `PHYSICIST`).
     The model order must be given; there is no order search for s here.
  2. Maps the fit to discrete time at the region's sampling frequency with `bilinear_transform` (now in `z_domain`, §11). This is
     the plain transform, s = (2/T)(z − 1)/(z + 1), with no prewarping.
  3. Builds the state space with `state_space_discrete`.

Originally s_domain was engineer-convention only, so the fit was made to conj(S) and conjugated back. Since §10 the
fit is made directly in the physicist convention (s = −jω), which the bilinear transform maps onto simphony's
z = e^{−jΩ} with no conjugation.

Fused S-parameter groups require their members to agree on `fitter`.

**Properties.**
- The model is exact at the fit centre: s = 0 maps to z = 1.
- It is stable and keeps a passive continuous fit passive. The imaginary axis maps onto the unit circle.
- The frequency axis is warped: f_digital = arctan(π f T)/(π T).

**Tested** in `tests/test_s_bilinear_fitter.py`:
- the discrete model equals the continuous fit evaluated at the warped frequency, to 1e-10;
- it is exact at the carrier;
- all poles lie inside the unit circle;
- the error grows away from the carrier while the z fit's does not.

The 43 block-mode and multirate tests still pass.

## 10. Unified s- and z-domain interfaces (2026-11-10)

`s_domain.py` now mirrors `z_domain.py` section for section, minus the sampling frequency. Helpers that do not
depend on the domain moved to a new `common.py`.

**s_domain API** (old name → new):

| Old | New |
|---|---|
| `vector_fitting(order, H, baseband_f, max_iterations, alpha, weight_threshold)` | `vector_fitting_continuous(order, H, f, center_frequency, sign_convention=PHYSICIST, max_iterations=10, alpha=0.01, weight_threshold=0.0)` |
| `vector_fitting_optimize_order(...)` | `optimize_order_vector_fitting_continuous(min, max, H, f, center_frequency, sign_convention=PHYSICIST, ...)` |
| `pole_residue_response(baseband_f, p, r, d)` | `pole_residue_response_continuous(f, center_frequency, p, r, d, sign_convention=PHYSICIST)` |
| `bilinear_transform(p, r, d, sampling_period)` | `bilinear_transform(p, r, d, sampling_frequency)` |
| none | `state_space_continuous`, `state_space_frequency_response_continuous` |

- **Centre frequency and convention.** s = sc·j2π(f − f_c), matching z = exp(sc·j2π(f − f_c)/f_s). z = e^{sT}
  then holds in either convention, Re p < 0 means stable in both, and `bilinear_transform` needs no convention
  argument. Poles and residues are in rad/s, relative to `center_frequency`.
- **Default convention is now `PHYSICIST`** (was engineer-only). To reproduce an old engineer fit, pass
  `sign_convention=ENGINEER`.
- **CVF starting poles.** `model_order` poles spread linearly over the band, with a common damping of
  `alpha · 2π max|f − f_c|`. `alpha` is now used. This replaces the conjugate pairs, so odd orders need no special
  case. On a two-ring all-pass test (800 GHz band, off-centre) the new start matched or beat the old one at every
  order (20, 30, 40) and iteration count (2–20).
- **`jax.lax.while_loop`** replaces `python_based_while_loop`, and the fit is `@persistent_cache`d like the z fit.
- **Removed:** `_full_lstsq_matrices`, `_enforce_conjugacy`, the commented-out `_lstsq_matrices` variants, the old
  tuple-returning `optimize_order`, and the `main`/`main2` demos in both files.

**common.py** holds `_lstsq_matrices`, `_fit_to_poles(H, phi0)`, `_weight_error(phi1, w)`,
`_relocate_poles(p, w)` (the eigenvalue step), `optimize_order` (z's cached, int-returning version, re-exported by
both modules) and `_state_space` (the replicated-pole ABCD realization, identical in both domains). Each domain
keeps its own `_phi_matrices`, `_initial_poles` and stability flip: `1/conj(p)` for z, `−conj(p)` for s.

**Least squares: full QR per pair, in both domains.** z's `_lstsq_matrices` factored phi0 once and orthogonalized
each pair block with one classical Gram–Schmidt pass (`A2 − Q1 Q1ᴴ A2`). Once a fit nears machine precision, that
pass loses orthogonality and the relocation drifts. Sharing it made s fits on the two-ring test fall from −255 dB to
−73 dB over 20 iterations, at N = 30. A second projection pass (CGS2) only partly helped. A Householder QR of
`[phi0, −H·phi1]` per pair (the old s version, still with row scaling instead of a dense diag) is stable. z fits did
not show the drift (they plateau near −125 dB on that test with every variant). On a seeded 2×3 z problem the
full-QR z fit equals the old one to 5e-15 in poles and 2e-14 in residues; only the order `eig` returns them in
changed. `CACHE_VERSION` was therefore **not** bumped. Cached z fits are the same models.

**z_domain:** `state_space_frequency_response_discrete` gains `sign_convention=PHYSICIST`, resolving its TODO. The
default keeps the old behaviour. It is also vectorized (batched `solve` instead of a per-frequency `inv`).

**Tested:**
- `tests/test_s_domain_vector_fitting.py`:
  - exact 2×3, order-7 model recovered in both conventions, with frequency in Hz around 193.4 THz (poles to 1e-8);
  - the physicist fit of H equals the conjugate of the engineer fit of conj(H);
  - moving `center_frequency` by Δf shifts the poles by −sc·j2πΔf;
  - the state-space response matches the pole-residue response;
  - the order search stays within its bounds.
- `tests/test_z_domain_vector_fitting.py`: the z state-space frequency response matches the pole-residue response
  in both conventions.
- z regression: fits on a fixed problem are unchanged, as above.
- 136 tests pass. Not run: sipann and plugins, whose optional dependencies `SiPANN` and `parsimonious` are not
  installed.

## 11. New: continuous-time simulation and state-space discretization (2026-11-10)

Additions for driving fitted models with waveforms without the simulator class
(`time_domain_paper/code/reservoir_application/gm_vector_fitting_z_vs_s.ipynb`).

- **`s_domain.state_space_response_continuous(A, B, C, D, u, sampling_frequency, x0=None, interpolation="linear")`.**
  The continuous counterpart of `z_domain.state_space_response_discrete`. It solves dx/dt = Ax + Bu exactly between
  samples for piecewise-linear (`"linear"`, as `scipy.signal.lsim`) or piecewise-constant (`"zero"`) input. Only
  the input is sampled; the model is not approximated.
- **`z_domain.continuous_to_discrete_state_space(A, B, C, D, sampling_frequency, method="bilinear")`.** The
  `scipy.signal.cont2discrete` conversions (`"bilinear"` Tustin, `"zoh"`, `"foh"`) for complex baseband models.
  Its B is dense, so the `_optimized` discrete path does not apply to the result.
- **`bilinear_transform` moved from `s_domain` to `z_domain`**, unchanged: conversions to a sampling period live
  with the discrete-time code. Its one caller, `libraries/ideal/s_parameters.py`, was updated.
- **`common._hold_matrices(A, B, T, hold)`**: the shared ZOH/FOH matrix exponential behind both functions.

Tests: the conversions match `cont2discrete` for every method. The bilinear state space has the same frequency
response as `bilinear_transform` + `state_space_discrete`. The continuous response matches `lsim` for both
interpolations and the analytic step response of a single pole. 21 tests pass in the three vector-fitting test
files.

## Numerical audit (no change needed)

Each step was checked on the paper-matched lattice (K = 44) against 50-digit `mpmath` references
(`time_domain_paper/z_vs_s`, audit run 2026-11-02):

| Step | z domain | s domain |
|---|---|---|
| weight least squares (`jnp.linalg.lstsq` on stacked R22) | rel. error 1e-13 to 8e-11 | 5e-14 to 1e-9 |
| pole relocation (`jnp.linalg.eig`) | 4e-15 to 2e-13 | 2e-13 at iteration 6 (1e-3 abs. at iteration 1, see below) |
| residue fit (`_fit_to_poles`) | identical to 50-digit (−53.0 dB) | identical (−48.3 dB) |
| refit of an exact 44-pole model | MAE −240 dB, pole error 7e-13 | −246 dB, 3e-12 |

* **Solvers and scaling.** JAX's and NumPy's LAPACK solvers agree. Column equilibration did not improve the
  least-squares or residue solves. The remaining fit errors are set by the models, not by numerical precision.
* **Ill-conditioned first step with near-DC starting poles.** With the paper's CVF starting poles
  ($-0.01|\omega| + j\omega$), the two poles nearest DC sit only ~0.002 rad/ps off the jω axis. The first iteration
  then has cond(phi0) ≈ 1e9, and its eigenvalues are good only to ~1e-3 (relative 2e-5). This clears by iteration 6
  and is a property of that starting-pole choice, not of the code.
* **Relaxed SK.** The paper's eq. (9) frees the denominator constant (relaxed SK, Gustavsen 2006); simphony fixes it
  to 1. A relaxed variant (normalization row Σ_f σ(f) = N, weighted by ‖H‖/N) gained 1–2.5 dB in the first s-domain
  iterations but ended 0.3–0.5 dB worse in both domains. Not worth changing.

## 12. New: passivity assessment and enforcement for z-domain fits (2026-11-16)

A z model H(z) = D + Σ Rₙ/(z − pₙ) is the simulator's whole band, Ω ∈ [−π, π), including the part outside the
fitted spectral range, where nothing constrains the fit. A scattering model is passive when it is stable and
σ_max(H(e^{jΩ})) ≤ 1 on the whole circle. The models are complex baseband, so the full circle is searched.

**`passivity.passivity_assessment_discrete(poles, residues, feedthrough, sign_convention)`.** Samples σ_max on a
uniform grid (default max(4096, 64 r) points) plus points clustered at each pole's angle (±8 widths of
2(1 − |p|)), and refines every local maximum above 0.999 with a bounded scalar search. Returns a
`PassivityReport` (max σ and where, violation peaks, the grid). This is a grid check; an exact algebraic check is
planned.

**`passivity.enforce_passivity_discrete(poles, residues, feedthrough, fit_omega, ...)`.** Poles fixed; residues
and D change. It solves the convex problem

  min mean_fit |ΔH|² + w · mean_circle |ΔH|²  subject to  σ_max(H + ΔH) ≤ 1 − margin on the whole circle

(defaults w = 1e-3, margin = 1e-4). The constraint is convex (a spectral norm of something affine in the
residues), and for unit u, v, Re(uᴴXv) ≤ σ_max(X). So Re(uᴴ H_new(Ω) v) ≤ 1 − margin, with (u, v) the singular
vectors of the current model at a violation, is an **exact** necessary condition, valid for every later
iterate. The method is Kelley's cutting planes: assess, add cuts at the violation peaks and in the violation bands,
re-solve, keep all cuts. Classic residue perturbation re-linearizes σ around each iterate and discards the old
constraints; on the SiEPIC half ring that cycled around σ ≈ 1.04 for 30+ iterations, because the linearization
underestimates σ (σ is convex) and each correction raised new peaks next to the constrained frequencies.

Every S-parameter entry shares the poles and the objective weights the entries equally, so each cut is a
pole-basis vector times a rank-one u vᴴ. The quadratic program is solved through its dual, a non-negative least
squares problem whose size is the number of cuts, with M_lk = ½ Re[(ψ_lᴴ G⁻¹ ψ_k)(u_lᴴ u_k)(v_kᴴ v_l)]. If at
most 1 + 1e-3 remains after the iterations, the model is finally scaled by 1/σ_max (exact).

**`SParameterElement`.** `vector_fitting_parameters["passivity"]`: `None` (default, unchanged), `"check"`
(warn if not passive) or `"enforce"`. It applies after either fitter (`"z"`, `"s_bilinear"`), on the
delay-compensated model (a pure delay has unit magnitude). Fused groups must agree on it. Results are not
persistently cached.

**Measured** (dt = 10 fs, band 1.5–1.6 µm, Ω ∈ [−0.38, 0.41]):

| model | data σ_max in band | fit σ_max on circle | iterations | cuts | in-band change | time |
|---|---|---|---|---|---|---|
| SiEPIC y-branch (3×3, order 12) | 0.9957 | 0.9958 (passive) | 0 | 0 | 0 | 0.03 s |
| SiEPIC directional coupler (4×4, order 16) | 1.0023 | 1.0028 | 1 | 73 | 6.3e-4 | 0.7 s |
| SiEPIC half ring (4×4, order 16) | 1.0062 | 1.0476 (out of band, Ω = −0.53) | 14 | 643 | 2.0e-3 | 4.2 s |
| Green Machine, Lumerical (8×8, order 90) | 1.0071 | 1.0073 | 5 | 361 | 1.2e-3 | 5.9 s |

All end below 1 on a 200 000-point re-check. Note that the data themselves are not passive (σ up to 1.007), so
some in-band change is unavoidable.

**Tested** in `tests/test_passivity_discrete.py`: the single-pole peak |a|/(1 − |p|) is found at the right Ω in
both conventions; the assessment matches `pole_residue_response_discrete`; a passive model is returned unchanged;
an out-of-band resonance with gain 1.5 is removed with < 1 % in-band change, in both conventions; the real
directional-coupler fit becomes passive on a 200 000-point grid with < 0.2 % change; the `SParameterElement`
options warn, enforce and reject unknown values. 162 tests pass.

**Follow-up (2026-11-17): dual solve factorization.** The dual matrix M has rank at most 2 (r+1) q m, the number
of real coefficients. Once the cuts outnumber that, M is singular, and rounding can push its null-space
eigenvalues below the 1e-12 ridge. `np.linalg.cholesky(M)` then raised "Matrix is not positive definite". This
happened on the FMCW example's 5×1 front end at dt = 2.5 ps (order 16): 321 cuts against 170 coefficients, with an
original σ_max of 19. The solve now lives in `_solve_dual` and factors M with `eigh`, flooring the eigenvalues at
the same ridge. The algorithm is otherwise unchanged. On that front end, enforcement now converges to σ_max = 1.0000
(in-band σ_max 0.914 → 0.924) in 1.9 s. A regression test, `test_dual_solve_with_rank_deficient_cut_matrix`,
covers it, and 168 tests pass.

**Follow-up (2026-11-24): passivity-aware order selection and enforcement options.**
- **The problem.** With `model_order=None`, the golden-section search chose the order by fitted error before
  enforcement ran. It favoured high orders that fit well in band but have large out-of-band gain, and enforcement
  then changed them the most. For example, a SiEPIC y-branch block at dt = 2.5 ps: order 8 has −58 dB error fitted
  and −27 dB after enforcement, while order 2 is passive at −29 dB.
- **The fix, in `optimize_order_vector_fitting_discrete`.** A new argument, `passivity_enforcement` (`None`, or a
  dict of options for `enforce_passivity_discrete`), scores each candidate order by its mean-squared error after
  enforcement. It still returns the fit before enforcement; the caller enforces it.
- **New element keys.**
  - `"passivity_aware_order"` (default `False`, opt-in) turns this on when the order is searched and
    `passivity="enforce"`. The study found it changed the order in only 2 of 10 cases, and made one worse.
  - `"passivity_options"` (default `None`) passes enforcement options (`out_of_band_weight`, `margin`, ...) both to
    the search and to the final enforcement.
  - Fused groups must agree on both.
- **Unchanged.** A fixed order, or `passivity` other than `"enforce"`, behaves exactly as before.
- **Tests.** Two new tests, and 170 pass. A study of the effect on the selected order, the error and the time is in
  `time_domain_paper/passivity_order/passivity_order.ipynb`.

**Follow-up (2026-11-24): a-priori order bounds (`order_estimation.py`).**
- **The problem.** The golden-section search is reliable only where MSE(n) + λ·n has one valley. Responses with many
  fringes or resonances have a plateau-then-cliff error curve, and fixed defaults (2..50) or an arbitrary range
  leave false valleys below the cliff. Example: a 1 mm discriminator over 1515–1585 nm got order 24 at −7 dB.
- **The estimate, `estimate_order_bounds`.**
  - It counts the normalised singular values of Loewner matrices built from the data on the unit circle: above
    `min_order_tolerance` (default 1e-3) for the minimum, and above `max_order_tolerance` (default 1e-6) for the
    maximum, capped by the data size.
  - Multiport data use random scalar projections uᴴHv, so the count is the number of distinct poles (vector
    fitting's order), not the McMillan degree.
  - Delay × bandwidth (Moon & Cangellaris) sizes the sampling and is reported, but it isn't a bound: it overestimates
    resonant responses badly (250 cycles for a ring that needs order 17).
  - It takes under 0.5 s per element.
- **Element behaviour.**
  - `min_model_order` / `max_model_order` now default to `None`, meaning estimated.
  - Explicit bounds are used as given, and a minimum below the estimate warns.
  - Fused groups estimate any bound a member leaves unset.
- **Wide-band results.** The search inside the estimated range reached −37 dB (1 mm discriminator, at about its
  vector-fitting limit), −56 dB (0.25 mm discriminator) and −72 dB (ring), against −7, −32 and −40 dB before.
- **Tests.** `tests/test_order_estimation.py` (4 tests); one passivity test updated to bounds at or above the
  estimate. 174 pass.

## Noticed but not changed

Resolved in §10: the conjugate-pair s starting poles, the ignored `alpha`, and the hard-coded convention in
`state_space_frequency_response_discrete`. The unused-QR note was stale: z's `_lstsq_matrices` used Q1 for its
projection. Since §10 both domains use one full QR per pair instead.
