# TODO: Turn this directory into a proper module
import importlib.resources as pkg_resources

import numpy as np
import jax.numpy as jnp
import hashlib

import os

import pickle

# TODO: Add a global cache directory and logic for where to look

LOCAL_CACHE_DIR = "./.simphony_cache/vector_fitting_cache"
# GLOBAL_CACHE_DIR = ... # IS there any way that you can make this in the python module folder?
# Replace 'your_package_name' with your actual package
GLOBAL_CACHE_DIR = pkg_resources.files("simphony") / "performance" / "vector_fitting_cache"
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

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

def persistent_cache(func):
    def wrapper(*args, **kwargs):
        use_cache = kwargs.pop("use_cache", True)
        save_global = kwargs.pop("save_global", False)

        if not use_cache:
            return func(*args, **kwargs)

        key_tuple = make_cache_key(*args, **kwargs)
        key = hash_key(key_tuple)
        filename = key + ".pkl"

        local_path = os.path.join(LOCAL_CACHE_DIR, filename)
        global_path = os.path.join(GLOBAL_CACHE_DIR, filename)

        # 🔍 1. Check LOCAL cache
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                return pickle.load(f)

        # 🌍 2. Check GLOBAL cache
        if os.path.exists(global_path):
            with open(global_path, "rb") as f:
                result = pickle.load(f)

            # Optional: copy to local cache for faster future access
            os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
            with open(local_path, "wb") as f:
                pickle.dump(result, f)

            return result

        # ⚙️ 3. Compute
        result = func(*args, **kwargs)

        # 💾 4. Always save locally
        os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
        with open(local_path, "wb") as f:
            pickle.dump(result, f)

        # 🌍 5. Optionally save globally
        if save_global:
            try:
                os.makedirs(GLOBAL_CACHE_DIR, exist_ok=True)
                with open(global_path, "wb") as f:
                    pickle.dump(result, f)
            except PermissionError:
                # Installed packages are often read-only
                pass

        return result

    return wrapper

# import numpy as np
# import jax.numpy as jnp
# import hashlib

# import os

# import pickle

# # TODO: Add a global cache directory and logic for where to look

# LOCAL_CACHE_DIR = "./.simphony_cache/vector_fitting_cache"
# GLOBAL_CACHE_DIR = ... # IS there any way that you can make this in the python module folder?


# def is_array(x):
#     return isinstance(x, (np.ndarray, jnp.ndarray))

# def hash_array(arr):
#     arr = np.asarray(arr)

#     h = hashlib.blake2b(digest_size=16)
#     h.update(arr.tobytes())
#     h.update(str(arr.shape).encode())
#     h.update(str(arr.dtype).encode())

#     return h.hexdigest()

# def make_cache_key(*args, **kwargs):
#     key_parts = []

#     # positional args
#     for arg in args:
#         if is_array(arg):
#             key_parts.append(("array", hash_array(arg)))
#         else:
#             key_parts.append(("value", arg))

#     # keyword args (sorted for determinism)
#     for k in sorted(kwargs.keys()):
#         v = kwargs[k]
#         if is_array(v):
#             key_parts.append((k, "array", hash_array(v)))
#         else:
#             key_parts.append((k, "value", v))

#     return tuple(key_parts)
# def hash_key(key_tuple):
#     return hashlib.blake2b(
#         str(key_tuple).encode(),
#         digest_size=16
#     ).hexdigest()

# os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

# def persistent_cache(func):
#     def wrapper(*args, **kwargs):
#         save_global = kwargs.pop("save_global", True)
#         # Now look
        
#         use_cache = kwargs.pop("use_cache", True)

#         if not use_cache:
#             return func(*args, **kwargs)
        
#         key_tuple = make_cache_key(*args, **kwargs)
#         key = hash_key(key_tuple)

#         filename = os.path.join(LOCAL_CACHE_DIR, key + ".pkl")

#         # Load if cached
#         # TODO: make sure that you look in the local and global caches before giving up
#         if os.path.exists(filename):
#             with open(filename, "rb") as f:
#                 return pickle.load(f)

#         # Compute
#         result = func(*args, **kwargs)

#         # Save
#         with open(filename, "wb") as f:
#             pickle.dump(result, f)

#         return result

#     return wrapper


