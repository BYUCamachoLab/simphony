"""Gaussian process propagation through causal MIMO LTI systems via the
Papoulis equations.

Propagates the mean and covariance of a Gaussian random process through a
discrete-time multi-input multi-output LTI system described by its impulse
response. This is the stochastic analog of the pole-residue state-space
simulation in vector_fitting/z_domain.py.

Shape conventions
-----------------
h       : (K, q, m)        impulse response, K taps, q outputs, m inputs
mu_x    : (T, m)           input mean sequence
mu_y    : (T, q)           output mean sequence
Cx      : (T, T, m, m)     input covariance blocks,  Cx[n1, n2] = Cov(x[n1], x[n2])
Cy      : (T, T, q, q)     output covariance blocks
Rxx/Ryy : same shape       autocorrelation = covariance + mean outer-product

Reference
---------
Papoulis, A., "Probability, Random Variables, and Stochastic Processes," 4th ed.,
McGraw-Hill, 2002. See Chapter 12 (response of LTI systems to random inputs).
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Covariance / autocorrelation transforms
# ---------------------------------------------------------------------------


@jax.jit
def covariance_to_autocorrelation(Cx, mu_x):
    """
    Rxx[n1, n2] = Cx[n1, n2] + mu_x[n1] ⊗ conj(mu_x[n2]).

    Parameters
    ----------
    Cx : jnp.ndarray, shape (T, T, m, m)
        Input covariance blocks.
    mu_x : jnp.ndarray, shape (T, m)
        Input mean sequence.

    Returns
    -------
    Rxx : jnp.ndarray, shape (T, T, m, m)
    """
    outer = jnp.einsum("ti,sj->tsij", mu_x, jnp.conj(mu_x))
    return Cx + outer


@jax.jit
def autocorrelation_to_covariance(Rxx, mu_x):
    """
    Cx[n1, n2] = Rxx[n1, n2] - mu_x[n1] ⊗ conj(mu_x[n2]).

    Parameters
    ----------
    Rxx : jnp.ndarray, shape (T, T, m, m)
        Autocorrelation blocks.
    mu_x : jnp.ndarray, shape (T, m)
        Mean sequence (use mu_y when converting output autocorrelation).

    Returns
    -------
    Cx : jnp.ndarray, shape (T, T, m, m)
    """
    outer = jnp.einsum("ti,sj->tsij", mu_x, jnp.conj(mu_x))
    return Rxx - outer


# ---------------------------------------------------------------------------
# Mean propagation
# ---------------------------------------------------------------------------


@jax.jit
def propagate_mean(h, mu_x):
    """Propagate the mean sequence through a causal MIMO LTI system.

    mu_y[n] = sum_{k=0}^{K-1} h[k] @ mu_x[n-k]

    Uses jax.lax.scan with a sliding buffer so the computation is O(T*K)
    and JIT-compilable.

    Parameters
    ----------
    h : jnp.ndarray, shape (K, q, m)
        Discrete-time impulse response.
    mu_x : jnp.ndarray, shape (T, m)
        Input mean sequence.

    Returns
    -------
    mu_y : jnp.ndarray, shape (T, q)
        Output mean sequence.
    """
    K, q, m = h.shape
    dtype = jnp.result_type(h, mu_x)

    def step(buffer, u_n):
        # Roll newest input to front; oldest drops off the back.
        new_buffer = jnp.roll(buffer, 1, axis=0).at[0].set(u_n)
        y_n = jnp.einsum("kqm,km->q", h, new_buffer)
        return new_buffer, y_n

    init_buffer = jnp.zeros((K, m), dtype=dtype)
    _, mu_y = jax.lax.scan(step, init_buffer, mu_x.astype(dtype))
    return mu_y


# ---------------------------------------------------------------------------
# Covariance propagation (Papoulis equations)
# ---------------------------------------------------------------------------


def _propagate_cross_correlation(h, Rxx):
    """
    First Papoulis step: Rxy[n1, n2] = sum_{k=0}^{K-1} Rxx[n1, n2-k] @ h[k]^H

    For each fixed n1 this is a causal convolution of the row Rxx[n1, :]
    with the conjugate-transposed impulse response. Rows are processed in
    parallel via jax.vmap.

    Parameters
    ----------
    h : jnp.ndarray, shape (K, q, m)
    Rxx : jnp.ndarray, shape (T, T, m, m)

    Returns
    -------
    Rxy : jnp.ndarray, shape (T, T, m, q)
    """
    K, q, m = h.shape
    # h[k]^H  shape: (K, m, q)
    h_H = jnp.conj(h).transpose(0, 2, 1)
    dtype = jnp.result_type(h, Rxx)

    def compute_rxy_row(rxx_row):
        # rxx_row: (T, m, m)  — Rxx[n1, :] for fixed n1
        def step(buffer, rxx_n2):
            new_buffer = jnp.roll(buffer, 1, axis=0).at[0].set(rxx_n2)
            # sum_k  buffer[k] @ h_H[k]  =>  (m, q)
            rxy = jnp.einsum("kij,kjl->il", new_buffer, h_H)
            return new_buffer, rxy

        init = jnp.zeros((K, m, m), dtype=dtype)
        _, rxy_row = jax.lax.scan(step, init, rxx_row.astype(dtype))
        return rxy_row  # (T, m, q)

    return jax.vmap(compute_rxy_row)(Rxx)  # (T, T, m, q)


def _propagate_output_autocorrelation(h, Rxy):
    """
    Second Papoulis step: Ryy[n1, n2] = sum_{k=0}^{K-1} h[k] @ Rxy[n1-k, n2]

    For each fixed n2 this is a causal convolution of the column Rxy[:, n2]
    with the impulse response. Columns are processed in parallel via jax.vmap.

    Parameters
    ----------
    h : jnp.ndarray, shape (K, q, m)
    Rxy : jnp.ndarray, shape (T, T, m, q)

    Returns
    -------
    Ryy : jnp.ndarray, shape (T, T, q, q)
    """
    K, q, m = h.shape
    dtype = jnp.result_type(h, Rxy)

    def compute_ryy_col(rxy_col):
        # rxy_col: (T, m, q)  — Rxy[:, n2] for fixed n2
        def step(buffer, rxy_n1):
            new_buffer = jnp.roll(buffer, 1, axis=0).at[0].set(rxy_n1)
            # sum_k  h[k] @ buffer[k]  =>  (q, q)
            # h: (K, q, m=l),  buffer: (K, m=l, q=o)
            ryy = jnp.einsum("kql,klo->qo", h, new_buffer)
            return new_buffer, ryy

        init = jnp.zeros((K, m, q), dtype=dtype)
        _, ryy_col = jax.lax.scan(step, init, rxy_col.astype(dtype))
        return ryy_col  # (T, q, q)

    # vmap over n2 (axis 1); transpose so n2 is outermost, then restore.
    Ryy = jax.vmap(compute_ryy_col)(Rxy.transpose(1, 0, 2, 3))  # (T_n2, T_n1, q, q)
    return Ryy.transpose(1, 0, 2, 3)  # (T_n1, T_n2, q, q)


def propagate_autocorrelation(h, Rxx):
    """Propagate input autocorrelation through a causal MIMO LTI system.

    Applies both Papoulis steps in sequence:

      Step 1:  Rxy[n1, n2] = sum_{k=0}^{K-1} Rxx[n1, n2-k] @ h[k]^H
      Step 2:  Ryy[n1, n2] = sum_{k=0}^{K-1} h[k] @ Rxy[n1-k, n2]

    Parameters
    ----------
    h : jnp.ndarray, shape (K, q, m)
        Discrete-time impulse response.
    Rxx : jnp.ndarray, shape (T, T, m, m)
        Input autocorrelation blocks.

    Returns
    -------
    Ryy : jnp.ndarray, shape (T, T, q, q)
        Output autocorrelation blocks.
    """
    Rxy = _propagate_cross_correlation(h, Rxx)
    return _propagate_output_autocorrelation(h, Rxy)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def gaussian_process_response(h, mu_x, Cx):
    """Compute output mean and covariance for a Gaussian input through a causal
    MIMO LTI system.

    This is the stochastic analog of state_space_response_discrete in
    vector_fitting/z_domain.py: given the system impulse response and the
    input statistics, return the output statistics.

    Parameters
    ----------
    h : jnp.ndarray, shape (K, q, m)
        Discrete-time impulse response.
    mu_x : jnp.ndarray, shape (T, m)
        Input mean sequence.
    Cx : jnp.ndarray, shape (T, T, m, m)
        Input covariance blocks. Cx[n1, n2] = Cov(x[n1], x[n2]).

    Returns
    -------
    mu_y : jnp.ndarray, shape (T, q)
        Output mean sequence.
    Cy : jnp.ndarray, shape (T, T, q, q)
        Output covariance blocks.
    """
    mu_y = propagate_mean(h, mu_x)
    Rxx = covariance_to_autocorrelation(Cx, mu_x)
    Ryy = propagate_autocorrelation(h, Rxx)
    Cy = autocorrelation_to_covariance(Ryy, mu_y)
    return mu_y, Cy


# ---------------------------------------------------------------------------
# Common input covariance constructors
# ---------------------------------------------------------------------------


def white_noise_covariance(T, m, sigma_sq=1.0):
    """Create a spectrally white (temporally uncorrelated) covariance matrix.

    Cx[n1, n2, i, j] = sigma_sq * delta[n1-n2] * delta[i-j]

    Parameters
    ----------
    T : int
        Number of time steps.
    m : int
        Number of modes.
    sigma_sq : float
        Noise power per mode (default 1.0).

    Returns
    -------
    Cx : jnp.ndarray, shape (T, T, m, m)
    """
    return (
        sigma_sq
        * jnp.eye(T, dtype=complex)[:, :, None, None]
        * jnp.eye(m, dtype=complex)[None, None, :, :]
    )


# ---------------------------------------------------------------------------
# Block ↔ flat matrix conversion utilities
# ---------------------------------------------------------------------------


def covariance_blocks_to_matrix(C):
    """Flatten block-structured covariance (T, T, M, M) → (T*M, T*M).

    The flat layout satisfies C_flat[n1*M+i, n2*M+j] = C[n1, n2, i, j].

    Parameters
    ----------
    C : jnp.ndarray, shape (T, T, M, M)

    Returns
    -------
    jnp.ndarray, shape (T*M, T*M)
    """
    T, _, M, _ = C.shape
    return C.transpose(0, 2, 1, 3).reshape(T * M, T * M)


def covariance_matrix_to_blocks(C_flat, T, M):
    """Unflatten covariance matrix (T*M, T*M) → block structure (T, T, M, M).

    Inverse of covariance_blocks_to_matrix.

    Parameters
    ----------
    C_flat : jnp.ndarray, shape (T*M, T*M)
    T : int
        Number of time steps.
    M : int
        Number of modes per time step.

    Returns
    -------
    jnp.ndarray, shape (T, T, M, M)
    """
    return jnp.asarray(C_flat).reshape(T, M, T, M).transpose(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def main():
    """Verify output variance matches Parseval's identity for white-noise
    input."""
    T = 60
    K = 12
    q, m = 2, 2

    key = jax.random.PRNGKey(0)
    h_real = jax.random.normal(key, (K, q, m))
    h_imag = jax.random.normal(jax.random.fold_in(key, 1), (K, q, m))
    h = (h_real + 1j * h_imag) / jnp.sqrt(2 * K)

    # Zero-mean white noise: Cx[n1,n2] = delta[n1,n2] * I_m
    mu_x = jnp.zeros((T, m), dtype=complex)
    Cx = (
        jnp.eye(T, dtype=complex)[:, :, None, None]
        * jnp.eye(m, dtype=complex)[None, None, :, :]
    )  # (T, T, m, m)

    mu_y, Cy = gaussian_process_response(h, mu_x, Cx)

    # For white noise the output power matrix at large lag should equal
    # sum_k h[k] @ h[k]^H  (Parseval).
    h_gram = jnp.sum(jnp.einsum("kqi,kpi->kqp", h, jnp.conj(h)), axis=0)  # (q, q)
    output_power = Cy[T - 1, T - 1]  # (q, q), last sample has seen all K taps

    print("Expected output power matrix (Parseval):")
    print(h_gram.real)
    print("Computed output power matrix:")
    print(output_power.real)
    print("Max absolute error:", jnp.max(jnp.abs(output_power - h_gram)).item())


if __name__ == "__main__":
    main()
