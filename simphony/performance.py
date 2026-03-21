import numpy as np
import jax.numpy as jnp
import hashlib

def is_array(x):
    return isinstance(x, (np.ndarray, jnp.ndarray))

def hash_array(arr):
    arr = np.asarray(arr)

    h = hashlib.blake2b(digest_size=16)
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())

    return h.hexdigest()

def make_cache_key(*args, **kwargs):
    key_parts = []

    # positional args
    for arg in args:
        if is_array(arg):
            key_parts.append(("array", hash_array(arg)))
        else:
            key_parts.append(("value", arg))

    # keyword args (sorted for determinism)
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if is_array(v):
            key_parts.append((k, "array", hash_array(v)))
        else:
            key_parts.append((k, "value", v))

    return tuple(key_parts)
def hash_key(key_tuple):
    return hashlib.blake2b(
        str(key_tuple).encode(),
        digest_size=16
    ).hexdigest()

import os
import pickle

LOCAL_CACHE_DIR = "./.simphony_cache/vector_fitting_cache"

# TODO: Add a global cache directory and logic for where to look

os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

def persistent_cache(func):
    def wrapper(*args, **kwargs):
        use_cache = kwargs.pop("use_cache", True)

        if not use_cache:
            return func(*args, **kwargs)
        
        key_tuple = make_cache_key(*args, **kwargs)
        key = hash_key(key_tuple)

        filename = os.path.join(LOCAL_CACHE_DIR, key + ".pkl")

        # Load if cached
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)

        # Compute
        result = func(*args, **kwargs)

        # Save
        with open(filename, "wb") as f:
            pickle.dump(result, f)

        return result

    return wrapper

# import hashlib
# import numpy as np
# from jax.numpy import ndarray as jax_ndarray

# def hash_array(arr):
#     """Deterministic hash of array content, shape, dtype."""
#     arr = np.asarray(arr)

#     h = hashlib.blake2b(digest_size=16)  # faster than sha256
#     h.update(arr.tobytes())
#     h.update(str(arr.shape).encode())
#     h.update(str(arr.dtype).encode())

#     return h.hexdigest()

# import os
# import pickle

# CACHE_DIR = "./vf_cache"
# os.makedirs(CACHE_DIR, exist_ok=True)

# def persistent_cache(func):
#     def wrapper(*args, **kwargs):
#         key = str(make_cache_key(*args, **kwargs))
#         filename = os.path.join(CACHE_DIR, key + ".pkl")

#         # Load if exists
#         if os.path.exists(filename):
#             with open(filename, "rb") as f:
#                 return pickle.load(f)

#         # Compute otherwise
#         result = func(*args, **kwargs)

#         # Save
#         with open(filename, "wb") as f:
#             pickle.dump(result, f)

#         return result

#     return wrapper


# from functools import wraps

# def array_cache(maxsize=None):
#     cache = {}

#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             key_parts = []

#             # hash positional args
#             for arg in args:
#                 if isinstance(arg, (np.ndarray, jax_ndarray)):
#                     key_parts.append(hash_array(arg))
#                 else:
#                     key_parts.append(arg)

#             # hash keyword args (sorted for determinism)
#             for k in sorted(kwargs.keys()):
#                 v = kwargs[k]
#                 if isinstance(v, (np.ndarray,jax_ndarray)):
#                     key_parts.append((k, hash_array(v)))
#                 else:
#                     key_parts.append((k, v))

#             key = tuple(key_parts)

#             if key in cache:
#                 return cache[key]

#             result = func(*args, **kwargs)

#             if maxsize is None or len(cache) < maxsize:
#                 cache[key] = result

#             return result

#         return wrapper
#     return decorator


