# `dla-ideal-solver`: Diffusion-Limited Aggregation Solver

[![DOI](https://zenodo.org/badge/1073943609.svg)](https://doi.org/10.5281/zenodo.17318133)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/dla-ideal-solver.svg)](https://pypi.org/project/dla-ideal-solver/)
[![PyPI downloads](https://img.shields.io/pypi/dm/dla-ideal-solver.svg)](https://pypi.org/project/dla-ideal-solver/)
[![PyPI status](https://img.shields.io/pypi/status/dla-ideal-solver.svg)](https://pypi.org/project/dla-ideal-solver/)

[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Numba](https://img.shields.io/badge/accelerated-numba-orange.svg)](https://numba.pydata.org/)
[![netCDF4](https://img.shields.io/badge/netCDF4-1.5.0+-blue.svg)](https://unidata.github.io/netcdf4-python/)
[![tqdm](https://img.shields.io/badge/tqdm-4.60.0+-green.svg)](https://tqdm.github.io/)
[![Pillow](https://img.shields.io/badge/Pillow-8.0.0+-yellow.svg)](https://python-pillow.org/)

High-performance Diffusion-Limited Aggregation (DLA) solver with Numba JIT compilation and parallel rendering.

## Physics

Simulates particle aggregation through random walks on a 2D lattice:

1. **Random Walk**: Particles perform 4-neighbor random walks
2. **Sticking Rule**: Particles stick when adjacent to existing aggregate
3. **Growth**: Dendritic structures emerge from stochastic aggregation

**Fractal Analysis**: Mass-radius relationship $M(R) \propto R^D$ gives fractal dimension $D \approx 1.71$ (2D DLA)

## Features

- **Numba JIT**: Compiled random walk for 100x speedup
- **Parallel rendering**: Multi-core GIF generation
- **NetCDF output**: Compact compressed format
- **4 test cases**: Classic, competitive, controlled, dense
- **Fractal analysis**: Automatic $D$ calculation

## Installation

```bash
# From PyPI
pip install dla-ideal-solver

# From source
git clone https://github.com/sandyherho/dla-ideal-solver.git
cd dla-ideal-solver
pip install -e .
```

## Quick Start

**Command line:**
```bash
# Run single case
dla-simulate case1

# Run all cases
dla-simulate --all

# Custom cores
dla-simulate case1 --cores 8
```

**Python API:**
```python
from dla_ideal import DLASolver

solver = DLASolver(N=512, n_cores=8)

result = solver.solve(
    n_walkers=10000,
    n_seeds=1,
    max_iter=100000,
    injection_mode='random'
)

print(f"Particles: {result['n_particles']}")
print(f"Aggregates: {result['n_aggregates']}")
print(f"Fractal dimension: {result['fractal_dimension']:.3f}")
```

## Test Cases

| Case | Description | Seeds | Walkers | Physics |
|------|-------------|-------|---------|---------|
| 1 | Classic DLA | 1 | 10k | Baseline dendritic |
| 2 | Multiple Seeds | 12 | 15k | Competition & fusion |
| 3 | Radial Injection | 1 | 10k | Controlled growth |
| 4 | High Density | 1 | 25k | Dense packing |

## Configuration

Key parameters:

```text
lattice_size = 512          # Grid size (N×N)
n_walkers = 10000           # Number of particles
n_seeds = 1                 # Initial sticky particles
max_iterations = 100000     # Safety limit
injection_mode = random     # 'random' or 'radial'
injection_radius = 180      # For radial mode
snapshot_interval = 100     # Frames per N particles
```

## Output

**NetCDF variables:**
- `grid(x,y)`: Final aggregate
- `snapshots(time,x,y)`: Growth evolution
- `glued_counts(time)`: Particle timeline
- `radii, masses`: Fractal analysis data

**Attributes:**
- `fractal_dimension`: $D$ from $M(R)$ fit
- `n_aggregates`: Number of clusters
- `n_particles`: Total stuck particles

**Reading data:**
```python
import netCDF4 as nc

data = nc.Dataset('outputs/case1_classic_dla.nc')
grid = data['grid'][:]
snapshots = data['snapshots'][:]
D = data.fractal_dimension
print(f"Fractal dimension: {D:.3f}")
```

## Mathematical Background

The fractal dimension $D$ is computed from the mass-radius scaling relationship:

$$M(R) = \int_0^R \rho(r) \, dV \propto R^D$$

where $M(R)$ is the mass within radius $R$ from the aggregate center. For 2D DLA:

$$\log M(R) = D \log R + \text{const}$$

The slope $D$ is obtained via linear regression on $\log$-$\log$ scale. Theoretical predictions give $D \approx 1.71$ for 2D DLA structures.

## Citation

```bibtex
@software{dla_solver_2025,
  author = {Herho, Sandy H. S. and Kaban, Siti N. and 
            Trilaksono, Nurjanna J. and Suwarman, Rusmawan and
            Irawan, Dasapta E.},
  title = {DLA Solver: Diffusion-Limited Aggregation with Numba},
  year = {2025},
  version = {0.0.1},
  license = {MIT}
}
```

## Authors

- Sandy H. S. Herho (sandy.herho@email.ucr.edu)
- Siti N. Kaban
- Nurjanna J. Trilaksono
- Rusmawan Suwarman
- Dasapta E. Irawan

## License

MIT License - See [LICENSE](LICENSE) for details.
