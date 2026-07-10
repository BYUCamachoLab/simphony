import jax
import jax.numpy as jnp

# TODO: Implement a decorator @jax.jit for the step_function
# (it needs a flag for the simulator to know whether it can be used in jax.jit)


def python_based_scan(f, init, xs=None, length=None):
    """Debug-only Python approximation of ``jax.lax.scan``.

    This helper is intentionally not a production simulation backend. It
    executes the loop eagerly in Python and is not guaranteed to match
    ``jax.lax.scan`` for tracing behavior, compilation behavior, pytree edge
    cases, mutation exposure, or numerical results. Use it only while debugging
    a step function; production simulation results should use the JAX scan path.
    """
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jax.tree_util.tree_map(lambda *values: jnp.stack(values), *ys)


def python_based_while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val
