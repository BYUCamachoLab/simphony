# Simphony Code Structure & Organization Review

## 1. Dead Code / Commented-Out Code (Critical)

There are approximately **4,150 lines** of dead code across the project:

- **`simphony/performance/performance.py`**: 112 lines of live code followed by **141 lines** of commented-out duplicates (lines 113-253).
- **`simphony/component/component.py`**: ~190 of its 331 lines are commented-out old function drafts, experimental plotting code, and abandoned class definitions (lines 62-187).
- **`simphony/libraries/ideal/s_parameters.py`**: ~180 lines of commented-out old class definition at the bottom (lines 526-707).
- **`simphony/time_domain/vector_fitting/old/`**: An entire `old/` directory with 5 files and ~2,400 lines of superseded code.
- **`simphony/time_domain/old_simulation.py`** (886 lines), **`quicker_delete.py`** (249 lines): Orphaned files with no imports.
- **`simphony/libraries/old_ideal.py`** (200 lines): Another orphaned old file.

**Recommendation**: Delete all of it. Git history preserves everything. Commented-out code makes the codebase harder to read and misleads contributors.

---

## 2. Caching System (`performance/performance.py`)

### a) Security — Pickle deserialization
The cache uses `pickle.load()` on disk files. If the local `.simphony_cache/` directory is in a shared or writable location, this is an arbitrary code execution vector. Consider using `numpy.savez`/`numpy.load` with `allow_pickle=False`, `safetensors`, or `json` + `numpy`.

### b) No cache invalidation
There is no TTL, max-size limit, or eviction policy. The local cache grows unboundedly. If the underlying function's behavior changes (e.g., a bug fix in vector fitting), stale cached results silently persist. The `CACHE_VERSION = "v1"` string is hardcoded and not tied to the package version.

**Recommendation**: Tie `CACHE_VERSION` to the actual code version. When the package version changes, old cache entries should be ignored or cleaned up.

### c) Side effect at import time
Line 18 (`os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)`) runs at import time, creating a `.simphony_cache` directory in whatever the CWD happens to be. Importing a library should not create directories.

**Recommendation**: Defer directory creation to first cache write. Use `platformdirs.user_cache_dir("simphony")`.

### d) Missing `functools.wraps`
The `persistent_cache` decorator doesn't preserve the wrapped function's name, docstring, or signature.

### e) No thread safety
Concurrent processes writing to the same cache file could corrupt it. Use atomic writes (write to temp file, then `os.rename`).

### f) Implicit `use_cache` and `save_global` kwargs
These are popped from `**kwargs` before forwarding, silently shadowing any parameter with the same name on the wrapped function.

---

## 3. Module Organization

### a) `performance/` directory is misnamed and unstructured
The TODO on line 1 says "Turn this directory into a proper module" — still not done. No `__init__.py`. Should be called `cache` or folded into a `utils` subpackage.

### b) `libraries/ideal/s_parameters.py` is doing too much
At 707 lines (525 live), this file handles SAX model wrapping, port directionality conversion, mode mux/demux netlist construction, vector fitting integration, block mode design, and sample mode design. Consider splitting into:
- `s_parameter_design.py`
- `block_mode_design.py`
- `sample_mode_design.py`
- `port_utils.py`

### c) Tests embedded inside the package
`simphony/time_domain/tests/` and `simphony/time_domain/examples/` are inside the distributed package. Move to top-level `tests/` and `examples/`.

### d) Inconsistent naming
- `SSFM.py` and `SSFM_old.py` use uppercase filenames, breaking Python module naming conventions.
- `_netlist.py` vs `netlist.py` split may be adding indirection without clarity.

---

## 4. Class Hierarchy & Design

### a) Component base class raises in `__init__`
`Component.__init__` raises `ValueError("Component is a base class")`. Use `abc.ABC` and `@abstractmethod` instead for proper enforcement and clearer error messages.

### b) Empty base class `Signal`
`component.py` line 189: `class Signal: ...` — placeholder with no behavior. Implement or remove.

### c) No use of `abc.ABC` / `@abstractmethod`
Base classes don't enforce that subclasses implement required methods. Adding ABC usage would catch missing implementations early.

### d) `SParameterComponent` marked for removal
Line 280: `# TODO: Get rid of this`. The migration path should be documented.

---

## 5. Code Quality Issues

### a) Debug `print()` statements in production code
`s_parameters.py` lines 247 and 260 contain debug prints that should be `logging.debug()` or removed.

### b) Incomplete implementation shipped
`_sample_mode_design()` (lines 228-232) returns literal `...` (ellipsis) objects. This will produce confusing runtime errors.

### c) Duplicate imports
`s_parameters.py` imports `ArrayLike` and `sax` twice (lines 1+9, 5+8).

### d) Typos
- `component.py` line 286: "onsidered" -> "considered"
- `component.py` line 302: "recieve" -> "receive"

---

## 6. Architectural Recommendations

### a) Use a proper logging framework
Replace `print()` and `warnings.warn()` with Python's `logging` module.

### b) Introduce a configuration object for cache settings
```python
@dataclass
class CacheConfig:
    enabled: bool = True
    local_dir: Path = platformdirs.user_cache_dir("simphony")
    max_size_mb: int = 500
    version: str = __version__
```

### c) Consider a registry pattern for component models
The `models` dict is passed around loosely. A typed `ModelRegistry` could provide validation and prevent name collisions.

### d) Consolidate signal types
The three signal modules define nearly identical classes differing only in array shape. Consider a shared base class or a single generic `Signal` parameterized by simulation mode.

---

## Priority Actions

| Priority | Action | Impact |
|----------|--------|--------|
| **High** | Delete all dead/commented-out code and `old/` directories | -4,000 lines, dramatically improves readability |
| **High** | Fix cache to not create dirs at import time; use `platformdirs` | Eliminates surprising side effects |
| **High** | Replace `pickle` in cache with safer serialization | Security fix |
| **Medium** | Split `s_parameters.py` into focused modules | Better maintainability |
| **Medium** | Use `abc.ABC` / `@abstractmethod` for base classes | Clearer contracts, better error messages |
| **Medium** | Remove debug `print()` statements, add logging | Production quality |
| **Medium** | Move embedded tests/examples out of the package | Smaller install size |
| **Low** | Add cache eviction and size limits | Prevents unbounded disk usage |
| **Low** | Fix duplicate imports and typos | Code quality polish |
| **Low** | Implement or remove `_sample_mode_design` stub | Prevent confusing runtime errors |
