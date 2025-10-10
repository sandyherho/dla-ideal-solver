"""
Diffusion-Limited Aggregation (DLA) Solver with Numba Acceleration
Implements random walk on 2D lattice with sticky particles
"""

import numpy as np
from numba import jit, prange
import numba
import os
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm


@jit(nopython=True, cache=True)
def compute_fractal_dimension_mass_radius(grid: np.ndarray, 
                                          center_x: int, 
                                          center_y: int,
                                          max_radius: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mass-radius relationship for fractal dimension."""
    n_radii = min(50, max_radius)
    radii = np.logspace(0.5, np.log10(max_radius), n_radii)
    masses = np.zeros(n_radii)
    
    for ir, r in enumerate(radii):
        r_sq = r * r
        mass = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 2:  # Sticky particle
                    dx = i - center_x
                    dy = j - center_y
                    dist_sq = dx*dx + dy*dy
                    if dist_sq <= r_sq:
                        mass += 1
        masses[ir] = mass
    
    return radii, masses


@jit(nopython=True, parallel=True, cache=True)
def random_walk_step(x: np.ndarray, y: np.ndarray, 
                    status: np.ndarray, grid: np.ndarray,
                    N: int, n_walkers: int) -> int:
    """
    Perform one random walk step for all mobile particles.
    Returns number of newly stuck particles.
    """
    n_glued = 0
    
    # Direction arrays for 4-neighbor movement
    dx = np.array([-1, 0, 1, 0])
    dy = np.array([0, -1, 0, 1])
    
    for i in prange(n_walkers):
        if status[i] == 1:  # Mobile particle
            # Pick random direction
            direction = np.random.randint(0, 4)
            
            # Calculate new position
            x_new = x[i] + dx[direction]
            y_new = y[i] + dy[direction]
            
            # Apply periodic boundaries
            x_new = (N + x_new) % N
            y_new = (N + y_new) % N
            
            # Update lattice
            grid[x[i], y[i]] = 0
            
            # Check for sticky neighbors
            has_sticky_neighbor = False
            for d in range(4):
                nx = (N + x_new + dx[d]) % N
                ny = (N + y_new + dy[d]) % N
                if grid[nx, ny] == 2:
                    has_sticky_neighbor = True
                    break
            
            if has_sticky_neighbor:
                # Stick particle
                grid[x_new, y_new] = 2
                status[i] = 2
                n_glued += 1
            else:
                # Move particle
                grid[x_new, y_new] = 1
                x[i] = x_new
                y[i] = y_new
    
    return n_glued


@jit(nopython=True, cache=True)
def initialize_radial_injection(N: int, n_walkers: int, 
                                radius: float, 
                                center_x: int, 
                                center_y: int) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize walkers on a circle for radial injection."""
    x = np.zeros(n_walkers, dtype=np.int32)
    y = np.zeros(n_walkers, dtype=np.int32)
    
    for i in range(n_walkers):
        theta = 2.0 * np.pi * np.random.random()
        x[i] = int(center_x + radius * np.cos(theta)) % N
        y[i] = int(center_y + radius * np.sin(theta)) % N
    
    return x, y


class DLASolver:
    """Diffusion-Limited Aggregation solver with Numba acceleration."""
    
    def __init__(self, N: int = 512, verbose: bool = True,
                 logger: Optional[Any] = None, n_cores: Optional[int] = None):
        """
        Initialize DLA solver.
        
        Args:
            N: Lattice size (N×N grid)
            verbose: Print progress messages
            logger: Optional logger instance
            n_cores: Number of CPU cores (None = all available)
        """
        self.N = N
        self.verbose = verbose
        self.logger = logger
        
        if n_cores is None:
            n_cores = os.cpu_count()
        numba.set_num_threads(n_cores)
        
        if verbose:
            print(f"  Lattice: {N} × {N}")
            print(f"  Using {n_cores} CPU cores")
    
    def solve(self, n_walkers: int = 10000, n_seeds: int = 1,
              max_iter: int = 100000, injection_mode: str = 'random',
              injection_radius: float = None, show_progress: bool = True,
              snapshot_interval: int = 100) -> Dict[str, Any]:
        """
        Run DLA simulation.
        
        Args:
            n_walkers: Number of random walking particles
            n_seeds: Number of initial sticky seeds
            max_iter: Maximum iterations
            injection_mode: 'random' or 'radial'
            injection_radius: Radius for radial injection (auto if None)
            show_progress: Show progress bar
            snapshot_interval: Save snapshot every N glued particles
        
        Returns:
            Dictionary with results
        """
        N = self.N
        
        # Initialize grid
        grid = np.zeros((N+2, N+2), dtype=np.int32)
        
        # Initialize particle positions and status
        x = np.zeros(n_walkers, dtype=np.int32)
        y = np.zeros(n_walkers, dtype=np.int32)
        status = np.ones(n_walkers, dtype=np.int32)  # 1=mobile, 2=sticky
        
        # Place seed particles
        if n_seeds == 1:
            # Single seed at center
            center_x, center_y = N // 2, N // 2
            grid[center_x, center_y] = 2
        else:
            # Multiple seeds randomly distributed
            for _ in range(n_seeds):
                sx = np.random.randint(0, N)
                sy = np.random.randint(0, N)
                grid[sx, sy] = 2
            center_x, center_y = N // 2, N // 2
        
        # Initialize walker positions
        if injection_mode == 'radial':
            if injection_radius is None:
                injection_radius = N // 4
            x, y = initialize_radial_injection(N, n_walkers, 
                                              injection_radius, 
                                              center_x, center_y)
        else:
            # Random distribution
            x = np.random.randint(0, N, n_walkers)
            y = np.random.randint(0, N, n_walkers)
        
        # Place walkers on grid
        for i in range(n_walkers):
            grid[x[i], y[i]] = 1
        
        # Tracking
        n_glued_total = n_seeds
        iteration = 0
        
        # Snapshot storage
        snapshots = []
        glued_counts = []
        
        # Save initial state
        snapshots.append(grid[0:N, 0:N].copy())
        glued_counts.append(n_seeds)
        
        if self.verbose:
            print(f"  Seeds: {n_seeds}")
            print(f"  Walkers: {n_walkers}")
            print(f"  Injection: {injection_mode}")
            if injection_mode == 'radial':
                print(f"  Injection radius: {injection_radius:.1f}")
        
        if show_progress:
            pbar = tqdm(total=n_walkers, desc="  Growing", unit=" particles")
            pbar.update(n_seeds)
        
        # Main simulation loop
        last_snapshot_count = n_seeds
        
        while n_glued_total < n_walkers and iteration < max_iter:
            n_glued_this_step = random_walk_step(x, y, status, grid, N, n_walkers)
            n_glued_total += n_glued_this_step
            iteration += 1
            
            if show_progress and n_glued_this_step > 0:
                pbar.update(n_glued_this_step)
            
            # Save snapshot at intervals
            if n_glued_total - last_snapshot_count >= snapshot_interval:
                snapshots.append(grid[0:N, 0:N].copy())
                glued_counts.append(n_glued_total)
                last_snapshot_count = n_glued_total
        
        if show_progress:
            pbar.close()
        
        # Final snapshot
        if glued_counts[-1] != n_glued_total:
            snapshots.append(grid[0:N, 0:N].copy())
            glued_counts.append(n_glued_total)
        
        # Compute fractal dimension
        final_grid = grid[0:N, 0:N]
        max_radius = min(N // 2, 200)
        
        radii, masses = compute_fractal_dimension_mass_radius(
            final_grid, center_x, center_y, max_radius
        )
        
        # Fit power law: log(M) = D * log(R) + const
        valid = (masses > 10) & (radii < max_radius * 0.8)
        if np.sum(valid) > 5:
            log_r = np.log(radii[valid])
            log_m = np.log(masses[valid])
            D = np.polyfit(log_r, log_m, 1)[0]
        else:
            D = np.nan
        
        # Count aggregates
        n_aggregates = self._count_aggregates(final_grid)
        
        if self.verbose:
            print(f"  Iterations: {iteration}")
            print(f"  Particles stuck: {n_glued_total}")
            print(f"  Aggregates: {n_aggregates}")
            if not np.isnan(D):
                print(f"  Fractal dimension: {D:.3f}")
        
        return {
            'grid': final_grid,
            'snapshots': np.array(snapshots),
            'glued_counts': np.array(glued_counts),
            'n_particles': n_glued_total,
            'n_aggregates': n_aggregates,
            'fractal_dimension': D,
            'radii': radii[valid] if np.sum(valid) > 0 else radii,
            'masses': masses[valid] if np.sum(valid) > 0 else masses,
            'center': (center_x, center_y),
            'params': {
                'N': N,
                'n_walkers': n_walkers,
                'n_seeds': n_seeds,
                'n_iterations': iteration,
                'injection_mode': injection_mode,
                'injection_radius': injection_radius if injection_mode == 'radial' else None
            }
        }
    
    def _count_aggregates(self, grid: np.ndarray) -> int:
        """Count number of separate aggregates using flood fill."""
        visited = np.zeros_like(grid, dtype=bool)
        n_aggregates = 0
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 2 and not visited[i, j]:
                    self._flood_fill(grid, visited, i, j)
                    n_aggregates += 1
        
        return n_aggregates
    
    def _flood_fill(self, grid: np.ndarray, visited: np.ndarray, 
                   i: int, j: int):
        """Flood fill to mark connected components."""
        stack = [(i, j)]
        
        while stack:
            x, y = stack.pop()
            
            if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
                continue
            if visited[x, y] or grid[x, y] != 2:
                continue
            
            visited[x, y] = True
            
            stack.extend([
                (x-1, y), (x+1, y), (x, y-1), (x, y+1)
            ])
