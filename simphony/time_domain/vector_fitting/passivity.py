"""Passivity assessment and enforcement for discrete-time (z-domain) pole-residue models.

A z-domain fit H(z) = D + sum_n R_n / (z - p_n), with z = exp(sign_convention * j Omega) and
Omega = 2 pi (f - f_c) / f_s, describes the simulator's whole band: every Omega in [-pi, pi). A scattering model
is passive when it is stable (all |p_n| < 1, which vector fitting already ensures) and

    sigma_max(H(Omega)) <= 1   for every Omega on the unit circle,

including outside the fitted spectral range, where the fit is otherwise unconstrained. Models here are complex
baseband (no conjugate symmetry), so the whole circle is searched.

`passivity_assessment_discrete` samples sigma_max on a dense grid over the circle, densest at the pole angles,
and refines its local maxima. `enforce_passivity_discrete` is residue perturbation (Gustavsen & Semlyen) carried
out directly in z: with the poles fixed the model is linear in the residues and D, the singular values are
linearised at the violation peaks, and the smallest change to the response on the fitted band (plus a small
weight on the whole circle) is found subject to sigma <= 1 - margin there. Assess, constrain, solve, repeat.

Every S-parameter entry shares the poles and the objective weights all entries alike, so the constraint system
factorises: each constraint is a rank-one (u v^H) direction times a pole-basis vector, and the dual quadratic
program is only as large as the number of constraints.

The check is a dense grid, so "passive" means passive on that grid. An exact algebraic check (Hamiltonian or
symplectic eigenvalues) is a planned addition.
"""

from typing import NamedTuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar, nnls

from simphony.conventions import PHYSICIST


class PassivityReport(NamedTuple):
    """Result of `passivity_assessment_discrete`.

    `max_singular_value` is the largest sigma_max found on the circle (after refinement) and `max_omega` where.
    `violation_omegas` / `violation_singular_values` are the refined local maxima with sigma_max > 1, sorted by
    Omega. `omega` / `singular_values` are the sampling grid and sigma_max on it.
    """

    max_singular_value: float
    max_omega: float
    violation_omegas: np.ndarray
    violation_singular_values: np.ndarray
    omega: np.ndarray
    singular_values: np.ndarray

    @property
    def is_passive(self):
        return self.max_singular_value <= 1.0


def _coefficients(residues, feedthrough):
    """Stack D and the residues into one coefficient array of shape (r + 1, q, m)."""
    residues = np.asarray(residues, dtype=complex)
    feedthrough = np.asarray(feedthrough, dtype=complex)
    return np.concatenate([feedthrough[None], residues], axis=0)


def _basis(omega, poles, sign_convention):
    """phi0(Omega) = [1, 1 / (z - p_1), ..., 1 / (z - p_r)], shape (len(omega), r + 1)."""
    z = np.exp(sign_convention * 1j * np.asarray(omega, dtype=float))
    return np.concatenate([np.ones((z.shape[0], 1)), 1 / (z[:, None] - poles[None, :])], axis=1)


def _response(omega, poles, coefficients, sign_convention):
    return np.einsum("kn,nqm->kqm", _basis(omega, poles, sign_convention), coefficients)


def _sigma_max(omega, poles, coefficients, sign_convention):
    return np.linalg.svd(_response(omega, poles, coefficients, sign_convention), compute_uv=False)[:, 0]


def _wrap(omega):
    return (np.asarray(omega) + np.pi) % (2 * np.pi) - np.pi


def assessment_grid(poles, sign_convention=PHYSICIST, num_samples=None, points_per_pole=41):
    """Sampling grid on the unit circle: uniform, plus points clustered at each pole's angle.

    A pole at radius rho gives a feature about 2(1 - rho) rad wide at Omega = sign_convention * angle(p), so each
    pole adds `points_per_pole` points spread over +-8 of those widths.
    """
    poles = np.asarray(poles, dtype=complex)
    if num_samples is None:
        num_samples = max(4096, 64 * poles.size)
    grids = [np.linspace(-np.pi, np.pi, num_samples, endpoint=False)]
    if poles.size:
        center = sign_convention * np.angle(poles)
        width = np.maximum(1 - np.abs(poles), 1e-12)
        t = np.linspace(-8, 8, points_per_pole)
        grids.append((center[:, None] + width[:, None] * t[None, :]).ravel())
    return np.unique(_wrap(np.concatenate(grids)))


def _refine_peaks(omega, sigma, poles, coefficients, sign_convention, threshold):
    """Refine the local maxima of sigma_max above `threshold` (circular grid). Returns (omegas, sigmas)."""
    left, right = np.roll(sigma, 1), np.roll(sigma, -1)
    candidates = np.flatnonzero((sigma >= left) & (sigma >= right) & (sigma > threshold))
    n = omega.size
    peaks, values = [], []
    for i in candidates:
        lo = omega[i - 1] if i > 0 else omega[-1] - 2 * np.pi
        hi = omega[i + 1] if i < n - 1 else omega[0] + 2 * np.pi
        result = minimize_scalar(
            lambda w: -_sigma_max(np.array([w]), poles, coefficients, sign_convention)[0],
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-12 * max(1.0, abs(hi - lo))},
        )
        w, s = (result.x, -result.fun) if -result.fun > sigma[i] else (omega[i], sigma[i])
        peaks.append(_wrap(w))
        values.append(s)
    order = np.argsort(peaks)
    return np.asarray(peaks)[order], np.asarray(values)[order]


def _assess(poles, coefficients, sign_convention, num_samples):
    omega = assessment_grid(poles, sign_convention, num_samples)
    sigma = _sigma_max(omega, poles, coefficients, sign_convention)
    # Refine every peak near the limit: one just above 1 can sit between grid points that are below it.
    peaks, values = _refine_peaks(omega, sigma, poles, coefficients, sign_convention, threshold=1 - 1e-3)
    i = int(np.argmax(sigma))
    max_omega, max_sigma = omega[i], sigma[i]
    if values.size and values.max() > max_sigma:
        j = int(np.argmax(values))
        max_omega, max_sigma = peaks[j], values[j]
    violating = values > 1.0
    return PassivityReport(float(max_sigma), float(max_omega), peaks[violating], values[violating], omega, sigma)


def passivity_assessment_discrete(
    poles, residues, feedthrough, sign_convention=PHYSICIST, num_samples=None
):
    """Check sigma_max(H) <= 1 on the whole unit circle for a z-domain pole-residue model.

    Parameters
    ----------
    poles, residues, feedthrough:
        Model as returned by `z_domain.vector_fitting_discrete`: shapes `(r,)`, `(r, q, m)` and `(q, m)`.
    sign_convention:
        The convention the model was fitted in (z = exp(sign_convention * j Omega)).
    num_samples:
        Uniform samples on the circle; points clustered at the pole angles are added. Default
        `max(4096, 64 r)`.

    Returns
    -------
    PassivityReport
        Frequencies are normalized angular frequencies Omega = 2 pi (f - f_c) / f_s in [-pi, pi).
    """
    poles = np.asarray(poles, dtype=complex)
    return _assess(poles, _coefficients(residues, feedthrough), sign_convention, num_samples)


def _cuts(omegas, poles, coefficients, sign_convention, threshold):
    """Cutting planes at `omegas` for every singular value of the current model above `threshold`.

    For unit vectors u, v and any matrix X, Re(u^H X v) <= sigma_max(X). So with (u, v) a singular pair of the
    current model at Omega, Re(u^H H_new(Omega) v) <= 1 is a necessary condition for passivity of any corrected
    model: an exact linear constraint, valid forever, not a linearization. Returns (psi, u, v) arrays with
    psi = conj(phi0(Omega)), so that u^H dH(Omega) v = sum_n conj(psi_n) u^H dc_n v.
    """
    if len(omegas) == 0:
        return np.zeros((0, poles.size + 1), complex), None, None
    phi = _basis(omegas, poles, sign_convention)
    U, S, Vh = np.linalg.svd(np.einsum("kn,nqm->kqm", phi, coefficients))
    k, l = np.nonzero(S > threshold)
    return np.conj(phi[k]), U[k, :, l], np.conj(Vh[k, l, :])


def _violating_omegas(report, max_band_points):
    """Refined violation peaks plus (subsampled) grid points inside the violation bands."""
    band = report.omega[report.singular_values > 1.0]
    if band.size > max_band_points:
        band = band[np.linspace(0, band.size - 1, max_band_points).astype(int)]
    return np.concatenate([report.violation_omegas, band, [report.max_omega]])


def _solve_dual(M, b):
    """lambda >= 0 minimizing 1/2 lambda^T M lambda + b^T lambda, as NNLS on a factor of M.

    M = L L^T with L = V sqrt(w), so 1/2 |L^T lambda + L^-1 b|^2 = 1/2 lambda^T M lambda + b^T lambda + const.
    M is only positive semi-definite: its rank is at most the number of real coefficients, 2 (r+1) q m, and with
    more cuts than that (or near-duplicate cuts) its smallest eigenvalues can come out slightly negative, where a
    Cholesky factorization fails. The eigenvalues are floored at a 1e-12 relative ridge instead.
    """
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-12 * max(np.trace(M) / M.shape[0], 1e-300))
    lam, _ = nnls(np.sqrt(w)[:, None] * V.T, -(V.T @ b) / np.sqrt(w), maxiter=50 * M.shape[0])
    return lam


def enforce_passivity_discrete(
    poles,
    residues,
    feedthrough,
    fit_omega,
    sign_convention=PHYSICIST,
    margin=1e-4,
    out_of_band_weight=1e-3,
    max_iterations=50,
    num_samples=None,
    max_band_points=50,
    max_rescale=1e-3,
):
    """Make a z-domain pole-residue model passive by perturbing its residues and D (poles fixed).

    Solves the convex problem

        minimize   mean_fit |dH|^2 + out_of_band_weight * mean_circle |dH|^2
        subject to sigma_max(H(Omega) + dH(Omega)) <= 1 - margin   for every Omega on the unit circle,

    where dH is linear in the changes of the residues and D. The constraint is convex (a spectral norm of an
    affine function), so it is imposed with cutting planes (Kelley's method): each iteration assesses the current
    model, and at its violation peaks and in its violation bands adds cuts Re(u^H H_new v) <= 1 - margin with the
    singular vectors there. Cuts are exact necessary conditions and are kept, so the iterates approach the
    optimum from outside and stop as soon as the assessment finds sigma_max <= 1. (Re-linearizing instead, as in
    classic residue perturbation, discards the cuts and can cycle.) Each quadratic program is solved through its
    dual, whose size is the number of cuts.

    Parameters
    ----------
    poles, residues, feedthrough:
        The fitted model, shapes `(r,)`, `(r, q, m)`, `(q, m)`.
    fit_omega:
        Normalized frequencies Omega = 2 pi (f - f_c) / f_s of the fitted data; the response is held close to
        the original there.
    sign_convention:
        The convention the model was fitted in.
    margin:
        The cuts target 1 - margin, which leaves slack so the iterations end on the passive side.
    out_of_band_weight:
        Relative weight of changes anywhere on the circle. Small, because out-of-band freedom is what removes
        gain there; non-zero, which keeps the problem well posed.
    max_iterations:
        Assess-and-cut iterations.
    num_samples:
        Assessment grid size (see `passivity_assessment_discrete`).
    max_band_points:
        Grid points per iteration taken from inside the violation bands (in addition to the refined peaks).
    max_rescale:
        Cutting planes approach the limit from outside, so the last iterations can leave sigma_max slightly above
        1. If at most `1 + max_rescale` remains, the whole model is finally scaled by 1 / sigma_max, which is
        exact. Set 0 to disable.

    Returns
    -------
    residues, feedthrough, info
        The corrected model and a dict with `iterations`, `max_singular_value_before`, `max_singular_value`
        (after), `passive`, `num_cuts`, `rescale` (the final scale factor, 1 if unused), and `relative_change`
        (rms change of the response on the fitted band over the rms response there). If the model is already
        passive it is returned unchanged.
    """
    poles = np.asarray(poles, dtype=complex)
    c0 = _coefficients(residues, feedthrough)
    fit_omega = np.asarray(fit_omega, dtype=float)

    report = _assess(poles, c0, sign_convention, num_samples)
    info = {"max_singular_value_before": report.max_singular_value, "iterations": 0, "num_cuts": 0, "rescale": 1.0}
    c = c0
    if not report.is_passive:
        # Objective metric G, the same for every S-parameter entry, with a tiny ridge for conditioning.
        phi_fit = _basis(fit_omega, poles, sign_convention)
        circle = np.linspace(-np.pi, np.pi, max(2048, 16 * poles.size), endpoint=False)
        phi_circle = _basis(circle, poles, sign_convention)
        G = phi_fit.conj().T @ phi_fit / fit_omega.size
        G += out_of_band_weight * (phi_circle.conj().T @ phi_circle) / circle.size
        G += 1e-14 * np.real(np.trace(G)) / G.shape[0] * np.eye(G.shape[0])
        G_factor = cho_factor(G)

        psi = np.zeros((0, poles.size + 1), complex)
        u = np.zeros((0, c0.shape[1]), complex)
        v = np.zeros((0, c0.shape[2]), complex)
        b = np.zeros(0)
        for iteration in range(1, max_iterations + 1):
            info["iterations"] = iteration
            omegas = _violating_omegas(report, max_band_points)
            new_psi, new_u, new_v = _cuts(omegas, poles, c, sign_convention, 1.0)
            if new_psi.shape[0] == 0:
                break
            # Re(u^H d(Omega) v) <= 1 - margin - Re(u^H H0(Omega) v), for the total change d = c_new - c0
            h0 = np.real(np.einsum("kq,kn,nqm,km->k", new_u.conj(), np.conj(new_psi), c0, new_v))
            psi = np.concatenate([psi, new_psi])
            u = np.concatenate([u, new_u])
            v = np.concatenate([v, new_v])
            b = np.concatenate([b, 1 - margin - h0])

            # Dual QP: min_{lambda >= 0} 1/2 lambda^T M lambda + b^T lambda, with
            # M_lk = 1/2 Re[(psi_l^H G^-1 psi_k)(u_l^H u_k)(v_k^H v_l)] and
            # d = -1/2 sum_k lambda_k (G^-1 psi_k) u_k v_k^H.
            Ginv_psi = cho_solve(G_factor, psi.T)                      # (r+1, K)
            M = 0.5 * np.real((psi.conj() @ Ginv_psi) * (u.conj() @ u.T) * (v.conj() @ v.T).T)
            lam = _solve_dual(M, b)
            c = c0 - 0.5 * np.einsum("nk,k,kq,km->nqm", Ginv_psi, lam, u, v.conj())
            info["num_cuts"] = psi.shape[0]
            report = _assess(poles, c, sign_convention, num_samples)
            if report.is_passive:
                break

        if not report.is_passive and report.max_singular_value <= 1 + max_rescale:
            info["rescale"] = (1 - 1e-12) / report.max_singular_value
            c = c * info["rescale"]
            report = _assess(poles, c, sign_convention, num_samples)

        response_fit = np.einsum("kn,nqm->kqm", phi_fit, c0)
        change = np.einsum("kn,nqm->kqm", phi_fit, c - c0)
        info["relative_change"] = float(
            np.sqrt(np.mean(np.abs(change) ** 2)) / max(np.sqrt(np.mean(np.abs(response_fit) ** 2)), 1e-300)
        )
    else:
        info["relative_change"] = 0.0
    info["max_singular_value"] = report.max_singular_value
    info["passive"] = report.is_passive
    return c[1:], c[0], info
