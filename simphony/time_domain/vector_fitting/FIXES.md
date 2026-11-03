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

## Noticed but not changed

- **`s_domain._initial_poles` builds conjugate pairs.** That is classic real vector fitting, not CVF. At baseband,
  half the starting poles mirror the band. In the lattice comparison this makes the shipped s fitter converge much
  more slowly (MAE −26 dB at iteration 2 vs −47.5 dB for a CVF-style non-conjugate start), though it reaches the same
  final error. Changing the default starting poles would change existing results, so I left it as a decision for you.
- **`z_domain.state_space_frequency_response_discrete` hard-codes z = e^{−jωT}** (physicist), as its own TODO notes.
- **`s_domain._initial_poles` ignores its `alpha` argument** (the damping is fixed at −0.1).
- **Unused `Q1, R11 = jnp.linalg.qr(phi0)`** in both `_lstsq_matrices`; it costs one extra QR per iteration.
