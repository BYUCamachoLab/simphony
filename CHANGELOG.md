# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.7.2](https://github.com/BYUCamachoLab/simphony/tree/v0.7.2) - <small>2024-01-09</small>

Minor bug fixes.

### Fixed
- Bug in loss of waveguide in ideal library was fixed.
  
---

## [0.7.1](https://github.com/BYUCamachoLab/simphony/tree/v0.7.1) - <small>2024-01-09</small>

Minor bug fixes.

### Fixed
- Bug in some of the tests after the models were rewritten.
- Bug in string formatting for parameterized filenames in SiEPIC libraries.
  
---

## [0.7.0](https://github.com/BYUCamachoLab/simphony/tree/v0.7.0) - <small>2024-01-06</small>

This is a refactor of simphony that uses [SAX](https://github.com/flaport/sax) 
as the s-parameter solver backend.

### Added

- Quantum simulations tools for simphony. The main advantage here is the use of
  classical s-parameter models for quantum simulations.

### Changed

- The s-parameter solver backend is now SAX.
- Simphony inherits the ability to run simulations on GPU's using 
  [JAX](https://jax.readthedocs.io/en/latest/). This means that simphony code
  now uses ``jax.numpy`` in place of ``numpy``.
- SiPANN wrappers are now built into simphony, instead of into SiPANN. In order
  to use them, you must still install SiPANN.

### Removed

- The majority of the package has been reimplemented or reorganized or made to
  depend on other packages. This means that the majority of the code has been
  removed, and this version of simphony is not compatible with previous 
  releases.

---

## [0.6.1](https://github.com/BYUCamachoLab/simphony/tree/v0.6.1) - <small>2022-02-18</small>

This patch makes the CMRR take beam splitting into account.

---

## [0.6.0](https://github.com/BYUCamachoLab/simphony/tree/v0.6.0) - <small>2022-01-12</small>

This version includes Relative Intensity Noise modeling and CMRR.

### Added

- Laser RIN model
- Differential Detector CMRR

### Changed

- coupling_loss now specified in dB

---

## [0.5.1](https://github.com/BYUCamachoLab/simphony/tree/v0.5.1) - <small>2021-11-09</small>

This patch update makes filtering for simulation devices work again.

---

## [0.5.0](https://github.com/BYUCamachoLab/simphony/tree/v0.5.0) - <small>2021-10-20</small>

This update adds a new way to run simulations. Sources and Detectors can be
connected to circuits during a simulation context. When the context ends,
the sources and detectors disconnect from the circuit so it can be used again.

### Added
- Simulation Contexts
- Noise simulations

---

## [0.4.0](https://github.com/BYUCamachoLab/simphony/tree/v0.4.0) - <small>2021-06-08</small>

This update changes the way components are connected and
pins are handled for the end user, and refactors the API 
substantially.

### Added
- SiPANN Wrapper included in simphony now
- Fix to [issue 17](https://github.com/BYUCamachoLab/SiPANN/issues/17)
  in SiPANN, now devices work in simphony regardless of
  number of pins.

### Changed
- refactored API
- simulator result units match initial units
- reorganize modules
- general code clean up

---

## 0.3.1 - <small>2020-05-19</small>

Some developer updates are mainly included in this release, along with a switch
back to the MIT License (from the GPLv3+ License).

### Added
- Pre-commit hooks added to assist with code quality.
- Bump2version (pip) now manages version numbering throughout the codebase.
- [#42](https://github.com/BYUCamachoLab/simphony/pull/42) Added the ability to
    simultaneously sweep over multiple inputs.

### Changed
- Switched back to the MIT License from the GPLv3+.

### Removed
- Some of the source files taken from the SiEPIC-Tools and SiEPIC EBeam PDK 
    packages, which were causing issues with the black codeformatter and were
    unnecessary (as we only use the data files for the compact models anyway).
- The (unimplemented) classes `SinglePortSweepSimulation` and 
    `MultiInputSweepSimulation` were removed in favor of adding single/multi
    input abilities to the base class `SweepSimulation`.

---

## [0.3.0](https://github.com/BYUCamachoLab/simphony/tree/v0.3.0) - <small>2020-05-18</small>

This version is a complete codebase rewrite implementing a much more
human-friendly way of defining circuits and running simulations. 

Circuits are now defined and stored in a clear, easy-to-understand 
object-oriented way. Integration with other packages should be easy, and the
creation of model libraries or other things that extend the functionality of
Simphony should fit easily into the existing framework.

NOTE: THIS VERSION IS NOT BACKWARDS COMPATIBLE.

### Added
- [#31](https://github.com/sequoiap/simphony/issues/31) Simphony was placed 
    under continuous integration to ensure code health.
- [#11](https://github.com/BYUCamachoLab/simphony/issues/11) Examples and 
    documentation were created and are hosted online at 
    https://simphonyphotonics.readthedocs.io/
- Nearly the entire SiEPIC EBeam PDK Compact Model Library is available.
- Parser compatible with circuit files exported by SiEPIC-Tools in KLayout
    allows Simphony to perform simulations on files created in KLayout.

### Changed
- Entire package was rewritten from previous versions.

### Removed
- Simphony simulation engine completely rewritten. Former scripts are not 
    compatible with this version and will need to be rewritten.

---

## 0.2.1 - <small>2019-08-07</small>

Some minor updates are included in this version.

### Added
- [#10](https://github.com/sequoiap/simphony/issues/10) ENHANCEMENT: Extras can
    now be passed in to the ebeam waveguide model to modify default parameters
    for central wavelength, effective index, group index, and dispersion.

### Changed
- README updated

### Removed
- 

---

## [0.2.0](https://github.com/BYUCamachoLab/simphony/tree/v0.2.0) - <small>2019-08-07</small>

Due to a complete code overhaul, this version is incompatible with previous 
versions of Simphony. Typically, when a release is incompatible, the MAJOR
version number is bumped. However, since this software is still in the first
stages of development, bumping the MINOR version symbolizes major changes
until the version 1 is officially released.

### Added
- Since we only need two of the 'connect' algorithms from scikit-rf, they were
    brought over into this library to reduce the number of dependencies.
- Added test cases using the pytest framework
- Models are now cached during simulation to avoid repeating calculations
- Logging is now the method of choice for getting info messages from the program
- Simulation runtime is now logged at the INFO level
- Documentation! Built using Sphinx.

### Changed
- Simphony is now segmented into three modules: core, simulation, and DeviceLibrary
- New component implementation methods (simphony.core.base.ComponentModel)
- Rebuilt device library to match new component implementations
- Device library no long reads data files for s-parameters, as they've been 
    converted and are now stored as .npz files.
- [#6](https://github.com/sequoiap/simphony/pull/6): ENHANCEMENT: Framework 
    rebuild, allows for easier implementation of custom component models and
    libraries.

### Removed
- Dependencies on scikit-rf, jsons, and matplotlib have been removed.
- Netlist export functionality no longer exists since netlists can be 
    scripted in Python.

---

## 0.1.5 - <small>2019-06-02</small>

### Added
- Waveguide parameters now calculated based off new regression model provided by Easton Potokar.
- [#3](https://github.com/sequoiap/simphony/pull/3): ENHANCEMENT: Add functionality for multi-input simulations

### Changed
- Dependencies are slightly more flexible now, not pinned to a specific version of software.

### Removed
- Dependencies on TensorFlow, Keras, and h5py have been removed since the waveguide model has been replaced.
- Removed the "settings" gui, as it's been migrated over to SiEPIC-Simphony.
- Persistent settings that should only really be in the SiEPIC-Simphony integration package were removed from here and implemented there instead.

---

## 0.1.4 - <small>2019-05-31</small>

### Added
- N/A

### Changed
- Dependencies are slightly more flexible now, not pinned to a specific version of software

### Removed
- circuit_simulation.py, a GUI primarily from when package was part of SiEPIC-Tools
- monte_carlo_simulation.py, a GUI primarily from when package was part of SiEPIC-Tools

---

## 0.1.2 - <small>2019-05-30</small>

### Added
- setup.py includes data files in build now

### Changed
- N/A

### Removed
- N/A
