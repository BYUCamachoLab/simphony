import jax.numpy as jnp

# TODO: Implement a decorator @jax.jit for the step_function 
# (it needs a flag for the simulator to know whether it can be used in jax.jit)

# Our implemenation of jax.lax.scan used for debugging
# and when 
def python_based_scan(f, init, xs=None, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, jnp.stack(ys)