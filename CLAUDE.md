# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simphony is a simulator for photonic circuits, allowing design and simulation of photonic integrated circuits with Python. It provides:
- A SPICE-like method for defining photonic circuits
- Subnetwork growth algorithms (20x speedup vs other software)
- Support for classical and quantum simulations
- Multiple simulation modes: S-parameter, sample-mode, block-mode, steady-state
- Component libraries including ideal models, SiEPIC, and SiPANN

Current version: 0.7.2 | Python 3.9+ | MIT License

## Development Commands

### Setup
```bash
python3 -m venv env
source env/bin/activate
make install  # Installs dev, doc, and test dependencies + pre-commit hooks
```

### Testing
```bash
make test           # Run all tests with coverage report
make test-force     # Run tests and regenerate test data
pytest              # Run pytest directly
pytest tests/path/to/test.py::TestClass::test_method  # Single test
```

### Code Quality
```bash
make format         # Black code formatter
make lint           # Flake8 linter
make mypy           # Type checking
make precommit      # Run pre-commit hooks on all files
```

### Documentation
```bash
make doc            # Build documentation (Jupyter Book)
make serve          # Start HTTP server at docs/_build/html
```

### Versioning
```bash
make patch          # Bump patch version
make minor          # Bump minor version
make major          # Bump major version
```

## Architecture Overview

### Core Modules

**Circuit Definition** (`simphony/circuit/`)
- `circuit.py`: Main Circuit class for defining photonic circuits with components and connections
- `netlist.py`: Netlisting utilities for converting between circuit representations (YAML, networkx graphs, SAX netlists)
- Netlists support SPICE-like hierarchical design with instances and connections

**Components** (`simphony/component/`)
- `component.py`: Base Component class and specialized variants (SParameterComponent, SteadyStateComponent, BlockModeComponent, SampleModeComponent)
- `pcell.py`: Parametric Cell (PCell) for procedurally-generated components
- `port.py`: Port definitions for component connectivity
- Components are the building blocks instantiated in circuits

**Simulations** (`simphony/simulation/`)
- `simulation.py`: Base Simulation and SimulationResult classes, SimulationMode enum (S_PARAMETER, SAMPLE_MODE, BLOCK_MODE, STEADY_STATE)
- `s_parameter.py`: S-parameter frequency-domain simulation
- `sample_mode.py`: Time-domain sample-mode simulation (JAX-based)
- `block_mode.py`: Block-mode signal simulation with vector fitting
- `steady_state.py`: Steady-state coherent simulation

**Signals** (`simphony/signal/`)
- `block_mode.py`: BlockModeOpticalSignal, BlockModeElectricalSignal, BlockModeLogicSignal dataclasses
- `sample_mode.py`: SampleModeOpticalSignal, SampleModeElectricalSignal, SampleModeLogicSignal dataclasses
- `steady_state.py`: SteadyStateOpticalSignal for coherent signals

**Component Libraries** (`simphony/libraries/`)
- `ideal/`: Ideal components (sources, couplers, modulators, filters, photonic circuits, waveguides)
- `siepic/`: SiEPIC PDK components (from SiEPIC-Tools)
- `sipann.py`: SiPANN component library integration
- `_internal/`: Shared utilities for port labeling and signal definitions

**Utilities** (`simphony/classical.py`, `simphony/quantum.py`, `simphony/utils.py`)
- `classical.py`: Classical simulation devices (Laser, Detector)
- `quantum.py`: Quantum simulation with QuantumState and covariance matrix representation
- `utils.py`: Coordinate conversion, signal utilities, YAML serialization

### Key Architectural Patterns

**SAX Integration**: Uses SAX (Symbolic Algebra xylose) for underlying circuit calculations. Models are SAX-compatible callables that compute S-parameters or other responses.

**Netlist System**: Circuits are represented as dictionaries with:
- `instances`: Named component instantiations with parameters
- `connections`: Net connections between ports
- `ports`: External circuit ports
- Settings can be attached to netlists and propagated to simulations

**PCell Framework**: PCells are components that generate netlists dynamically based on parameters. They implement a design space that can be instantiated at simulation time.

**Multi-Mode Simulation**: The framework supports different signal representations (optical, electrical, logic) and simulation methodologies, selected via SimulationMode enum.

**Signal Dataclasses**: Use Flax struct.dataclass for immutable signal representation with JAX compatibility.

**Vector Fitting**: Block-mode simulations use pole-residue models fitted via vector fitting algorithm (`time_domain/vector_fitting/`) for frequency-dependent behavior.

## Code Style & Conventions

- **Pre-commit hooks** (black, isort, autoflake, docformatter) enforce code style automatically on commit
- **Import ordering**: isort configured with black profile (see pyproject.toml [tool.isort])
- **Line length**: 88 characters (black default)
- **Type hints**: Used throughout for JAX arrays via `jax.typing.ArrayLike`
- **Docstrings**: NumPy style docstrings with parameter descriptions
- **Naming**: PascalCase for classes, snake_case for functions/variables

## Important Dependencies

- **JAX**: Numerical computing with automatic differentiation (required, >=0.4.18)
- **SAX**: Symbolic circuit calculations (required, >=0.10.3)
- **NumPy/SciPy**: Numerical operations and signal processing
- **Flax**: Struct dataclasses for JAX-compatible immutable objects
- **lark**: Parser for netlist DSL (>=1.1.5)
- **pandas**: Data handling (>=2.0.0)

## Testing Infrastructure

- Tests in `tests/` using pytest framework
- Test data in `tests/data/`
- Library-specific tests in `tests/libraries/`
- Plugin tests in `tests/plugins/`
- Configuration in `pyproject.toml [tool.pytest.ini_options]`

## Documentation

- Built with Jupyter Book (`docs/` folder)
- Tutorials in `docs/tutorials/` (intro, mzi, filters, recursive, quantum)
- API reference auto-generated from docstrings
- Contributing guides in `docs/dev/`

## Common Development Scenarios

**Adding a new ideal component**: Create a function in `simphony/libraries/ideal/` that returns a SAX-compatible model, following the pattern of existing components in `s_parameters.py` or `photonic_circuits.py`.

**Creating a PCell**: Extend PCell in `simphony/component/pcell.py`, implementing the design space and netlist generation logic. Register in appropriate library module.

**Implementing a simulation mode**: Extend Simulation base class and handle the specific signal types and computation method. Register SimulationMode enum if new.

**Working with netlists**: Use utilities from `simphony/circuit/netlist.py` (netlist_to_graph, add_settings_to_netlist, instantiate_netlist) to manipulate circuit representations.

## Current Development Focus

Repository is on `instantiated_circuit` branch with ongoing work on:
- Block-mode simulations with JAX
- Multimode transient analysis
- InstantiatedCircuit graph-based circuit representation
- Performance optimizations via caching and vector fitting
