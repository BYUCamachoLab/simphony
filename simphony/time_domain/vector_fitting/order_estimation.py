"""A-priori model-order bounds for vector fitting, from the frequency data alone (no fitting).

The golden-section order search (`common.optimize_order`) is reliable only in a range where its objective,
MSE(n) + lambda * n, has one valley. Responses with many fringes or resonances in the fitted band show a plateau-then-
cliff error curve: below roughly two poles per phase cycle the fit cannot follow the response at all, and the
search's objective has false valleys there. `estimate_order_bounds` places the search range past that cliff.

**Loewner rank (the bounds).** For data g(z_k) on points of the unit circle, split into two interleaved sets
(mu_i, lambda_j), the Loewner matrix L_ij = (g(mu_i) - g(lambda_j)) / (mu_i - lambda_j) has rank equal to the degree
of a rational g (Mayo & Antoulas 2007), and for real data its singular values decay as the remaining detail gets
finer. Vector fitting's order is the number of distinct poles shared by all S-parameter entries, not the McMillan
degree of the matrix function (which counts each pole once per rank of its residue, up to min(outputs, inputs)
times). So the Loewner matrix is built for scalar projections g = u^H H v with random unit vectors u, v, which
generically have exactly the distinct poles of H, and the largest count over a few projections is used. Counting the normalised singular values above a tolerance estimates the order needed for that
accuracy: on SiEPIC-based blocks over 1515-1585 nm the count above 1e-3 matched the order needed for a -60 dB
maximum error to about 20 % (time_domain_paper/wideband). The minimum is the count above `min_tolerance` and the
maximum is the count above `max_tolerance` (about the data's own accuracy floor, beyond which poles fit noise or
interpolation ripple).

**Delay x bandwidth (sampling and diagnostics).** The number of phase cycles across the band, B * tau, where tau is
the spread of the dominant paths in the impulse response, is the time-bandwidth rule of Moon & Cangellaris. It sizes
the Loewner sampling (enough points per cycle) and is reported, but it is not used as a bound: for resonant
responses the impulse response decays exponentially, which costs only about two poles per resonance, and the rule
would demand far more.
"""
from typing import NamedTuple

import numpy as np

from simphony.conventions import PHYSICIST


class OrderEstimate(NamedTuple):
    min_order: int
    max_order: int
    singular_values: np.ndarray       # normalised (largest = 1)
    phase_cycles: float               # delay x bandwidth of the dominant paths
    points_per_set: int               # frequency points in each Loewner partition


def delay_bandwidth_cycles(transfer_function, frequency, threshold_dB=-20.0):
    """Phase cycles across the band of the dominant paths: B * (spread of the impulse-response region within
    `threshold_dB` of its peak), maximised over the S-parameter entries. Uses a Hann-windowed inverse FFT of the
    (uniformly sampled) data. The spread is the shortest circular arc holding every sample above the threshold, so it
    does not depend on the sign convention (where the impulse response lands on the FFT's circular time axis)."""
    H = np.asarray(transfer_function)
    f = np.asarray(frequency)
    n = H.shape[0]
    bandwidth = f[-1] - f[0]
    h = np.abs(np.fft.ifft(H * np.hanning(n)[:, None, None], axis=0))
    cycles_per_sample = bandwidth / (n * (bandwidth / (n - 1)))     # samples are 1 / (n df) apart in time
    cycles = 0.0
    for i in range(H.shape[1]):
        for j in range(H.shape[2]):
            hij = h[:, i, j]
            if hij.max() == 0:
                continue
            idx = np.nonzero(hij >= hij.max() * 10 ** (threshold_dB / 20))[0]
            gaps = np.diff(np.concatenate([idx, [idx[0] + n]]))
            span = n - gaps.max()                                     # shortest circular arc covering idx
            cycles = max(cycles, span * cycles_per_sample)
    return float(cycles)


def loewner_singular_values(transfer_function, frequency, center_frequency, sampling_frequency,
                            sign_convention=PHYSICIST, points_per_set=None, projections=3, seed=0):
    """Normalised singular values of Loewner matrices of the data on z = exp(s_c j Omega),
    Omega = 2 pi (f - f_c) / f_s, with the points split into two interleaved sets of `points_per_set` each
    (default: all). For a multiport `transfer_function` the matrix is built for `projections` random scalar
    projections u^H H v (see the module docstring) and the elementwise maximum of their sorted, normalised singular
    values is returned, so counts above a tolerance estimate the number of distinct poles."""
    H = np.asarray(transfer_function)
    if H.shape[1] * H.shape[2] > 1:
        rng = np.random.default_rng(seed)
        curves = []
        for _ in range(projections):
            u = rng.normal(size=H.shape[1]) + 1j * rng.normal(size=H.shape[1])
            v = rng.normal(size=H.shape[2]) + 1j * rng.normal(size=H.shape[2])
            g = np.einsum("i,kij,j->k", np.conj(u / np.linalg.norm(u)), H, v / np.linalg.norm(v))[:, None, None]
            curves.append(loewner_singular_values(g, frequency, center_frequency, sampling_frequency,
                                                  sign_convention, points_per_set))
        return np.max(np.stack(curves), axis=0)
    f = np.asarray(frequency)
    z = np.exp(1j * sign_convention * 2 * np.pi * (f - center_frequency) / sampling_frequency)
    mu, lam, Hmu, Hlam = z[0::2], z[1::2], H[0::2], H[1::2]
    if points_per_set is not None and points_per_set < len(lam):
        keep_mu = np.linspace(0, len(mu) - 1, points_per_set).round().astype(int)
        keep_lam = np.linspace(0, len(lam) - 1, points_per_set).round().astype(int)
        mu, Hmu, lam, Hlam = mu[keep_mu], Hmu[keep_mu], lam[keep_lam], Hlam[keep_lam]
    q, m = H.shape[1], H.shape[2]
    L = (Hmu[:, None] - Hlam[None, :]) / (mu[:, None] - lam[None, :])[:, :, None, None]
    L = L.transpose(0, 2, 1, 3).reshape(len(mu) * q, len(lam) * m)
    s = np.linalg.svd(L, compute_uv=False)
    return s / s[0]


def estimate_order_bounds(transfer_function, frequency, center_frequency, sampling_frequency,
                          sign_convention=PHYSICIST, min_tolerance=1e-3, max_tolerance=1e-6, max_points_per_set=500):
    """A-priori (min, max) model order for vector fitting of `transfer_function` (num_frequency, q, m).

    min: Loewner singular values above `min_tolerance` (1e-3 ~ a -60 dB maximum fit error); max: above
    `max_tolerance` (1e-6 ~ the data's accuracy floor), capped by the Loewner rank limit and half the number of
    frequency samples. The Loewner sampling uses `max_points_per_set` points per partition, raised to four per phase
    cycle (delay x bandwidth) when the response needs more."""
    H = np.asarray(transfer_function)
    n = H.shape[0]
    cycles = delay_bandwidth_cycles(H, frequency)
    points = int(min(n // 2, max(max_points_per_set, np.ceil(4 * cycles))))
    s = loewner_singular_values(H, frequency, center_frequency, sampling_frequency, sign_convention, points)
    rank_limit = points
    min_order = int(max(2, np.count_nonzero(s > min_tolerance)))
    max_order = int(min(np.count_nonzero(s > max_tolerance), rank_limit, n // 2))
    max_order = max(max_order, min_order + 2)
    return OrderEstimate(min_order, max_order, s, cycles, points)
