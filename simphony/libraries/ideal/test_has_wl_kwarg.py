"""
Diagnostic analysis of has_wl_kwarg() in _calculate_state_space_coefficients_from_sax_model.

The function is defined inside _calculate_state_space_coefficients_from_sax_model and
determines whether a SAX model depends on wavelength. If it returns False the caller
treats the model as memoryless (constant_over_wavelength = True) and collapses the
entire transfer function into a single D matrix, giving zero temporal dynamics.

This script copies the function verbatim, then systematically tests it against every
model type that might be passed in — including the _get_filtered_sax_model wrapper used
by gaussian_process_s_parameter — to find exactly where it returns False incorrectly.
"""

import sys
sys.path.insert(0, '.')

import inspect
import functools
import sax
from jax import config
config.update("jax_enable_x64", True)

from simphony.libraries.old_ideal import coupler, waveguide
from simphony.libraries.ideal.s_parameters import _get_filtered_sax_model


# ---------------------------------------------------------------------------
# Verbatim copy of has_wl_kwarg from s_parameters.py (line 996)
# ---------------------------------------------------------------------------

def has_wl_kwarg(model):
    sig = inspect.signature(model)

    if "wl" in sig.parameters:
        return True

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        try:
            model(wl=0.0)
            return True
        except TypeError:
            return False

    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def report(label, model):
    sig = inspect.signature(model)
    params = sig.parameters
    has_wl = "wl" in params
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    # If **kwargs present, probe whether wl is actually accepted
    var_kw_accepts_wl = None
    if has_var_kw:
        try:
            model(wl=0.0)
            var_kw_accepts_wl = True
        except TypeError:
            var_kw_accepts_wl = False

    result = has_wl_kwarg(model)

    print(f"\n{'='*60}")
    print(f"Model : {label}")
    print(f"  sig          : {sig}")
    print(f"  has 'wl'     : {has_wl}")
    print(f"  has **kwargs : {has_var_kw}")
    if has_var_kw:
        print(f"  **kwargs accepts wl=0.0 : {var_kw_accepts_wl}")
    print(f"  >>> has_wl_kwarg() returns : {result}  {'✓' if result else '✗ WRONG — will be treated as constant!'}")
    return result


# ---------------------------------------------------------------------------
# 1. Baseline: raw model functions
# ---------------------------------------------------------------------------

print("\n" + "#"*60)
print("# 1. Raw SAX model functions")
print("#"*60)

report("waveguide (raw)", waveguide)
report("coupler   (raw)", coupler)


# ---------------------------------------------------------------------------
# 2. SAX circuit composed from raw models
# ---------------------------------------------------------------------------

print("\n" + "#"*60)
print("# 2. sax.circuit() output")
print("#"*60)

ring_netlist = {
    "instances": {"wg": "waveguide", "dc": "dc"},
    "connections": {"dc,o2": "wg,o0", "dc,o3": "wg,o1"},
    "ports": {"o0": "dc,o0", "o1": "dc,o1"},
}
ring_sax, _ = sax.circuit(
    netlist=ring_netlist,
    models={"waveguide": waveguide, "dc": coupler},
)

report("ring_sax  (sax.circuit output)", ring_sax)


# ---------------------------------------------------------------------------
# 3. _get_filtered_sax_model wrapper
# ---------------------------------------------------------------------------

print("\n" + "#"*60)
print("# 3. _get_filtered_sax_model() wrapper")
print("#"*60)

port_directionality = {"o0": "input", "o1": "output"}
default_modes = ("TE", "TM")

filtered = _get_filtered_sax_model(ring_sax, port_directionality, default_modes)

report("filtered_sax_model (wrapper around ring_sax)", filtered)

# Inspect what _get_filtered_sax_model actually does to the signature
print("\n--- Signature inspection of filtered model ---")
sig_raw = inspect.signature(ring_sax)
sig_filt = inspect.signature(filtered)
print(f"  ring_sax  signature : {sig_raw}")
print(f"  filtered  signature : {sig_filt}")

# Show __wrapped__ chain if present
obj = filtered
depth = 0
while hasattr(obj, '__wrapped__'):
    depth += 1
    print(f"  __wrapped__ depth {depth}: {obj.__wrapped__}")
    obj = obj.__wrapped__

# Show functools.wraps metadata
print(f"  filtered.__module__  : {getattr(filtered, '__module__', 'N/A')}")
print(f"  filtered.__name__    : {getattr(filtered, '__name__', 'N/A')}")
print(f"  filtered.__qualname__: {getattr(filtered, '__qualname__', 'N/A')}")
print(f"  filtered.__wrapped__ : {getattr(filtered, '__wrapped__', 'N/A')}")


# ---------------------------------------------------------------------------
# 4. Probe calling behaviour of the filtered model
# ---------------------------------------------------------------------------

print("\n" + "#"*60)
print("# 4. Calling behaviour of filtered model")
print("#"*60)

# Does it accept wl=0.0?
try:
    result = filtered(wl=0.0)
    print(f"  filtered(wl=0.0) succeeded → {type(result)}")
except TypeError as e:
    print(f"  filtered(wl=0.0) raised TypeError: {e}")
except Exception as e:
    print(f"  filtered(wl=0.0) raised {type(e).__name__}: {e}")

# Does it accept wl as positional?
try:
    result = filtered(0.0)
    print(f"  filtered(0.0) succeeded → {type(result)}")
except TypeError as e:
    print(f"  filtered(0.0) raised TypeError: {e}")
except Exception as e:
    print(f"  filtered(0.0) raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 5. Root cause: what inspect.signature sees for functools.wraps
# ---------------------------------------------------------------------------

print("\n" + "#"*60)
print("# 5. Root-cause analysis: inspect.signature vs functools.wraps")
print("#"*60)

# _get_filtered_sax_model uses @functools.wraps(sax_model) which copies
# sax_model's __wrapped__ attribute. inspect.signature follows __wrapped__
# by default (follow_wrapped=True). Let's check whether that is happening.

print("\nDoes inspect.signature follow __wrapped__?")
sig_follow = inspect.signature(filtered, follow_wrapped=True)
sig_nofollow = inspect.signature(filtered, follow_wrapped=False)
print(f"  follow_wrapped=True  : {sig_follow}")
print(f"  follow_wrapped=False : {sig_nofollow}")

print("\nConclusion:")
has_wl_follow   = "wl" in sig_follow.parameters
has_wl_nofollow = "wl" in sig_nofollow.parameters
has_var_kw_follow   = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_follow.parameters.values())
has_var_kw_nofollow = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_nofollow.parameters.values())
print(f"  follow=True  → 'wl' in params: {has_wl_follow},  **kwargs: {has_var_kw_follow}")
print(f"  follow=False → 'wl' in params: {has_wl_nofollow}, **kwargs: {has_var_kw_nofollow}")


# ---------------------------------------------------------------------------
# 6. Reproduce with a minimal stand-alone example
# ---------------------------------------------------------------------------

print("\n" + "#"*60)
print("# 6. Minimal reproduction")
print("#"*60)

def inner(wl=1.55, length=10.0):
    return {"result": wl * length}

@functools.wraps(inner)
def outer(**kwargs):
    return inner(**kwargs)

print(f"  inner sig: {inspect.signature(inner)}")
print(f"  outer sig: {inspect.signature(outer)}")
print(f"  has_wl_kwarg(inner) = {has_wl_kwarg(inner)}")
print(f"  has_wl_kwarg(outer) = {has_wl_kwarg(outer)}")
print()
print("  When @functools.wraps copies inner's signature onto outer,")
print("  inspect.signature(outer) returns inner's signature (has 'wl').")
print("  But filtered_sax_model is defined with **kwargs, and wraps(ring_sax)")
print("  sets __wrapped__ = ring_sax. inspect.signature follows __wrapped__")
print("  and returns ring_sax's signature — which should have 'wl'.")
print()

# Now reproduce what _get_filtered_sax_model does exactly
@functools.wraps(ring_sax)
def filtered_manual(*args, **kwargs):
    sdict = ring_sax(*args, **kwargs)
    return {k: v for k, v in sdict.items()}

# Override signature explicitly (as _get_filtered_sax_model does)
filtered_manual.__signature__ = inspect.signature(ring_sax)

print(f"  filtered_manual sig: {inspect.signature(filtered_manual)}")
print(f"  has_wl_kwarg(filtered_manual) = {has_wl_kwarg(filtered_manual)}")


# ---------------------------------------------------------------------------
# 7. The exact _get_filtered_sax_model implementation
# ---------------------------------------------------------------------------

print("\n" + "#"*60)
print("# 7. Does _get_filtered_sax_model set __signature__ explicitly?")
print("#"*60)

print(f"  filtered.__signature__ attr: {getattr(filtered, '__signature__', 'NOT SET')}")
print()
print("  inspect.signature() checks for __signature__ FIRST.")
print("  If __signature__ is set, it returns that and ignores __wrapped__.")
print()

# So the question is: what is filtered.__signature__?
sig_explicit = getattr(filtered, '__signature__', None)
if sig_explicit is not None:
    print(f"  __signature__ is SET to: {sig_explicit}")
    print(f"  'wl' in __signature__.parameters: {'wl' in sig_explicit.parameters}")
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in sig_explicit.parameters.values())
    print(f"  **kwargs in __signature__: {has_var_kw}")
else:
    print("  __signature__ is NOT explicitly set.")


# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------

print("\n" + "#"*60)
print("# 8. SUMMARY OF ROOT CAUSE")
print("#"*60)
print()
sig_used = inspect.signature(filtered)
params_used = sig_used.parameters
has_wl_final = "wl" in params_used
has_var_kw_final = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params_used.values())

print(f"  inspect.signature(filtered) = {sig_used}")
print(f"  'wl' in parameters          = {has_wl_final}")
print(f"  **kwargs in parameters      = {has_var_kw_final}")
print()
if not has_wl_final and not has_var_kw_final:
    print("  RESULT: has_wl_kwarg returns False because:")
    print("    - 'wl' is NOT in the visible signature parameters, AND")
    print("    - there are no **kwargs to probe at runtime.")
    print()
    print("  This means _get_filtered_sax_model's __signature__ override")
    print("  produces a signature that lacks both 'wl' and **kwargs.")
    print("  The function then falls through both checks in has_wl_kwarg")
    print("  and returns False, making the model appear constant over wavelength.")
elif has_wl_final:
    print("  RESULT: has_wl_kwarg correctly returns True ('wl' is visible).")
elif has_var_kw_final:
    print("  RESULT: has_wl_kwarg falls into the **kwargs branch.")
    print("  Whether it returns True depends on whether filtered(wl=0.0) succeeds.")
