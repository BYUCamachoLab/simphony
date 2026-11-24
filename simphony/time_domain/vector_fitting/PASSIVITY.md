# Passivity of z-domain fits: derivation and algorithm (code map)

Uncommitted work on `time-domain-paper` (2026-11-16). The full derivation, with no code references, is in
`time_domain_paper/passivity-enforcement-derivation.md`. This file summarises it and points at the code. Paths are
relative to the repository root.

| File | What |
|---|---|
| `simphony/time_domain/vector_fitting/passivity.py` | new module: assessment and enforcement |
| `simphony/libraries/ideal/s_parameters.py` | `vector_fitting_parameters["passivity"]` option |
| `tests/test_passivity_discrete.py` | 8 tests |
| `FIXES.md` §12, `CHANGELOG.md` | notes |

## 1. Problem

A z fit H(z) = D + Σₙ Rₙ/(z − pₙ), z = e^{s_c jΩ}, Ω = 2π(f − f_c)/f_s, is evaluated on the whole unit circle,
Ω ∈ [−π, π), and only part of the circle is fitted. Passive means stable (already ensured by the pole flip in
`z_domain.vector_fitting_discrete`) and σ_max(H(Ω)) ≤ 1 for every Ω. The model is complex baseband, so the whole
circle is searched.

With the poles fixed, H is linear in c = (D, R₁, …, R_r), stacked as an (r+1, q, m) array:
H(Ω) = Σₙ φₙ(Ω) cₙ with φ(Ω) = [1, 1/(z − p₁), …, 1/(z − p_r)].
- `_coefficients` stacks c (`passivity.py:55`).
- `_basis` is φ (`passivity.py:62`).
- `_response` and `_sigma_max` evaluate H and σ_max (`passivity.py:68`, `:72`).

## 2. Assessment (`passivity_assessment_discrete`, `passivity.py:134`)

- **Grid** (`assessment_grid`, `passivity.py:80`). max(4096, 64 r) uniform points, plus 41 points per pole over
  ±8 widths, where a pole's width is 2(1 − |p|) at Ω = s_c·arg p. That's where |a/(z − p)| peaks, at
  |a|/(1 − |p|).
- **Refinement** (`_refine_peaks`, `passivity.py:98`). Circular local maxima above 1 − 1e-3 are refined with
  `minimize_scalar(method="bounded")` between their grid neighbours. The threshold sits below 1 so that a peak
  just over 1 between grid points is still found (`passivity.py:124`).
- **Report** (`_assess`, `passivity.py:120`, returns `PassivityReport`, `:35`). The max σ and where it occurs,
  the refined peaks above 1, and the grid.

It is a grid check only; see §6.

## 3. Optimization problem

Minimize J(d) = mean over fit |ΔH|²_F + w · mean over circle |ΔH|²_F, subject to σ_max(H⁰ + ΔH) ≤ 1 − μ on
the whole circle. Here d = c − c⁰ is the correction.

- Every entry shares the basis, so J(d) = Σ_ij d_ijᴴ G d_ij, with
  G = Φ_fitᴴΦ_fit/N_fit + w·Φ_circleᴴΦ_circle/N_circle + ridge (`passivity.py:252–258`).
- Defaults: w = `out_of_band_weight` = 1e-3, μ = `margin` = 1e-4.
- **Convex:** σ_max of an affine function of c is convex. **Always strictly feasible:** the zero model passes.
  So strong duality holds.

## 4. Cuts, not re-linearization (the key point)

- **The cut.** σ_max(X) = max over unit u, v of Re(uᴴXv). So for any unit u, v,
  Re(uᴴ H(Ω; c) v) ≤ 1 − μ is an exact linear necessary condition. Taking (u, v) as the singular pairs of the
  current iterate at a violation (σ_l > 1) gives cuts the iterate violates.
- **Where cuts come from.** `_cuts` (`passivity.py:158`) builds them. `_violating_omegas` (`passivity.py:174`)
  supplies the refined peaks, ≤ `max_band_points` = 50 grid points inside the violation bands, and the max point.
- **Kept forever.** Cuts accumulate in `psi, u, v, b` (`passivity.py:260–275`); nothing is ever removed. This is
  Kelley's cutting-plane method.
- **Why it matters.** Classic residue perturbation uses the same hyperplane (its first-order σ expansion equals
  the cut) but rebuilds the constraints each iteration. That cycled on the SiEPIC half ring at σ ≈ 1.04.
- **Right-hand side.** b_k = 1 − μ − Re(u_kᴴ H⁰(Ω_k) v_k) (`passivity.py:271`). It is written for the total
  change from the original c⁰, so the objective measures the total change.

## 5. Dual solve (`enforce_passivity_discrete`, `passivity.py:182`)

- **Cut as a vector.** Each cut is Re(a_kᴴ d) ≤ b_k with a_k,nij = ψ_k,n u_k,i conj(v_k,j) and ψ_k = conj(φ(Ω_k)):
  a basis vector times a rank-one u vᴴ.
- **Stationarity:** d = −½ G⁻¹ Σ λ_k a_k, i.e. d_n = −½ Σ_k λ_k (G⁻¹ψ_k)_n u_k v_kᴴ (`passivity.py:286`).
- **Dual:** min over λ ≥ 0 of ½λᵀMλ + bᵀλ, with
  M_lk = ½ Re[(ψ_lᴴG⁻¹ψ_k)(u_lᴴu_k)(v_kᴴv_l)] (`passivity.py:280–282`). The size is the number of cuts, not
  ports × poles.
- **Solved as NNLS:** M = LLᵀ, then `nnls(Lᵀ, −L⁻¹b)` (`passivity.py:283–285`).
- **Loop:** re-assess after each solve and stop when passive (`passivity.py:288–290`).
- **Final rescale.** Cutting planes approach the feasible set from outside. If 1 < σ_max ≤ 1 + `max_rescale`
  (1e-3) remains, scale c by (1 − 1e-12)/σ_max, which is exact (`passivity.py:292–295`). It wasn't needed in the
  examples below.
- **Returns** `(residues, feedthrough, info)` with `iterations`, `num_cuts`, `rescale`, before and after σ_max,
  `passive`, and `relative_change` (`passivity.py:297–306`). A passive input is returned unchanged.

## 6. SParameterElement integration (`s_parameters.py`)

- **The option.** `"passivity"`: `None` | `"check"` | `"enforce"`, default `None` (`s_parameters.py:69`). The
  docstring is just below it.
- **Where it runs.** In `_calculate_state_space_coefficients_from_sax_model`, after either fitter and before
  `state_space_discrete` (`s_parameters.py:843`). The fit frequencies Ω come from the existing `Omega` variable;
  the model is the delay-compensated one, and a pure delay doesn't change passivity.
- **What each value does** (`_apply_passivity`, `s_parameters.py:854`). `"check"` warns if the model isn't
  passive. `"enforce"` corrects it, and warns only if correction fails. Unknown values raise.
- **Fused groups** must agree on `passivity` (`s_parameters.py:578`).
- **Not persistently cached.** `persistent_cache` keys on arguments only, so it would return stale results after
  code changes. The cost is 1–6 s per element.

## 7. Results (dt = 10 fs, 1.5–1.6 µm)

| Model | Data σ_max | Fit σ_max (circle) | Iterations | Cuts | In-band change | Time |
|---|---|---|---|---|---|---|
| SiEPIC y-branch (order 12) | 0.9957 | 0.9958 | 0 | 0 | 0 | 0.03 s |
| SiEPIC directional coupler (order 16) | 1.0023 | 1.0028 | 1 | 73 | 6.3e-4 | 0.7 s |
| SiEPIC half ring (order 16) | 1.0062 | 1.0476 at Ω = −0.53 (out of band) | 14 | 643 | 2.0e-3 | 4.2 s |
| Green Machine (8×8, order 90) | 1.0071 | 1.0073 | 5 | 361 | 1.2e-3 | 5.9 s |

All are below 1 on a 200 000-point re-check. The data themselves are non-passive, so some in-band change is
unavoidable.

## 8. Tests (`tests/test_passivity_discrete.py`)

- `:41` the single-pole peak |a|/(1 − |p|) at Ω = s_c·arg p, both conventions.
- `:52` the grid σ matches `pole_residue_response_discrete`.
- `:63` a passive model is returned unchanged.
- `:72` an out-of-band resonance with gain 1.5 is removed with < 1 % in-band change, both conventions.
- `:94` the real SiEPIC directional coupler fit becomes passive on a 200k grid with < 0.2 % change.
- `:121` element options: `"check"` warns, `"enforce"` is silent, unknown values raise.

The full suite passes: 162 tests (sipann and plugins skipped, missing optional dependencies).

## 9. Worth scrutinising in review

- **Grid check only.** "Passive" means passive on the grid plus refined peaks. Very sharp features between
  clustered points could still be missed. Items 2 and 3 of the plan add exact checks: a Hamiltonian via the
  bilinear map, or a symplectic matrix in z.
- **Cut growth.** Cuts accumulate without pruning, and M is dense K × K. That's fine at a few hundred to two
  thousand cuts, and inactive cuts (λ = 0) could be dropped later.
- **NNLS conditioning.** NNLS on the Cholesky factor with a 1e-12 ridge on M was accurate in testing: primal
  infeasibility of the subproblem was about 1e-12. A dedicated QP solver would be more robust for very many
  near-duplicate cuts.
- **Objective weighting.** The objective weights all S-parameter entries equally (Frobenius norm), which is what
  makes the rank-one factorization possible. Relative or per-entry weighting would break it.
- **Changes allowed.** Only the residues and D change. Moving poles (pole perturbation) is not attempted.
- **Cut threshold.** `_cuts` is called with threshold 1.0, so only singular values above 1 get cuts, not those
  within the margin.
