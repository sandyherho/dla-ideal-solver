"""
Diffusion-Limited Aggregation (DLA) Solver with Numba Acceleration
Implements random walk on 2D lattice with sticky particles
"""

import numpy as np
from numba import jit
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


@jit(nopython=True, cache=True)
def find_aggregate_bounds(grid: np.ndarray, N: int) -> Tuple[int, int, int, int]:
    """Find bounding box of all sticky particles."""
    min_x, max_x = N, 0
    min_y, max_y = N, 0
    
    for i in range(N):
        for j in range(N):
            if grid[i, j] == 2:
                if i < min_x:
                    min_x = i
                if i > max_x:
                    max_x = i
                if j < min_y:
                    min_y = j
                if j > max_y:
                    max_y = j
    
    return min_x, max_x, min_y, max_y


@jit(nopython=True, cache=True)
def reinject_walker(i: int, x: np.ndarray, y: np.ndarray, 
                   grid: np.ndarray, N: int,
                   min_x: int, max_x: int, min_y: int, max_y: int,
                   injection_mode: str, injection_radius: float,
                   center_x: int, center_y: int) -> None:
    """Re-inject a walker near the aggregate."""
    # Clear old position
    grid[x[i], y[i]] = 0
    
    if injection_mode == 'radial':
        # Radial injection
        theta = 2.0 * np.pi * np.random.random()
        x[i] = int(center_x + injection_radius * np.cos(theta)) % N
        y[i] = int(center_y + injection_radius * np.sin(theta)) % N
    else:
        # Random injection near aggregate
        margin = 20
        x_range = max(max_x - min_x + 2 * margin, N // 4)
        y_range = max(max_y - min_y + 2 * margin, N // 4)
        
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        
        x[i] = (cx + np.random.randint(-x_range//2, x_range//2)) % N
        y[i] = (cy + np.random.randint(-y_range//2, y_range//2)) % N
    
    # Make sure position is empty
    max_attempts = 20
    for _ in range(max_attempts):
        if grid[x[i], y[i]] == 0:
            break
        x[i] = np.random.randint(0, N)
        y[i] = np.random.randint(0, N)
    
    grid[x[i], y[i]] = 1


@jit(nopython=True, cache=True)
def random_walk_step(x: np.ndarray, y: np.ndarray, 
                    status: np.ndarray, grid: np.ndarray,
                    walker_age: np.ndarray,
                    N: int, n_walkers: int,
                    reinject_timeout: int,
                    min_x: int, max_x: int, min_y: int, max_y: int,
                    injection_mode: str, injection_radius: float,
                    center_x: int, center_y: int) -> int:
    """
    Perform one random walk step for all mobile particles.
    Returns number of newly stuck particles.
    """
    n_glued = 0
    
    # Direction arrays for 4-neighbor movement
    dx = np.array([-1, 0, 1, 0])
    dy = np.array([0, -1, 0, 1])
    
    # Create temporary arrays to avoid conflicts
    new_grid = grid.copy()
    
    for i in range(n_walkers):
        if status[i] == 1:  # Mobile particle
            walker_age[i] += 1
            
            # Re-inject if walker is taking too long
            if reinject_timeout > 0 and walker_age[i] > reinject_timeout:
                reinject_walker(i, x, y, new_grid, N, 
                               min_x, max_x, min_y, max_y,
                               injection_mode, injection_radius,
                               center_x, center_y)
                walker_age[i] = 0
                continue
            
            # Clear old position
            new_grid[x[i], y[i]] = 0
            
            # Pick random direction
            direction = np.random.randint(0, 4)
            
            # Calculate new position with periodic boundaries
            x_new = (x[i] + dx[direction]) % N
            y_new = (y[i] + dy[direction]) % N
            
            # Check if new position is already occupied by another mobile walker
            if new_grid[x_new, y_new] == 1:
                # Position blocked, stay in place
                new_grid[x[i], y[i]] = 1
                continue
            
            # Check for sticky neighbors
            has_sticky_neighbor = False
            for d in range(4):
                nx = (x_new + dx[d]) % N
                ny = (y_new + dy[d]) % N
                if grid[nx, ny] == 2:  # Check original grid
                    has_sticky_neighbor = True
                    break
            
            if has_sticky_neighbor:
                # Stick particle
                new_grid[x_new, y_new] = 2
                status[i] = 2
                n_glued += 1
                x[i] = x_new
                y[i] = y_new
                walker_age[i] = 0
            else:
                # Move particle
                new_grid[x_new, y_new] = 1
                x[i] = x_new
                y[i] = y_new
    
    # Copy new grid back
    for i in range(N):
        for j in range(N):
            grid[i, j] = new_grid[i, j]
    
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


@jit(nopython=True, cache=True)
def initialize_walkers_no_overlap(N: int, n_walkers: int, 
                                  grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize walkers randomly ensuring no overlap."""
    x = np.zeros(n_walkers, dtype=np.int32)
    y = np.zeros(n_walkers, dtype=np.int32)
    
    for i in range(n_walkers):
        # Find empty position
        max_attempts = 100
        for attempt in range(max_attempts):
            pos_x = np.random.randint(0, N)
            pos_y = np.random.randint(0, N)
            
            if grid[pos_x, pos_y] == 0:
                x[i] = pos_x
                y[i] = pos_y
                grid[pos_x, pos_y] = 1
                break
        else:
            # If we can't find empty position, place anyway
            x[i] = np.random.randint(0, N)
            y[i] = np.random.randint(0, N)
            grid[x[i], y[i]] = 1
    
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
            print(f"  Enhanced: Walker re-injection enabled")
    
    def solve(self, n_walkers: int = 10000, n_seeds: int = 1,
              max_iter: int = 100000, injection_mode: str = 'random',
              injection_radius: float = None, show_progress: bool = True,
              snapshot_interval: int = 100,
              reinject_timeout: int = None) -> Dict[str, Any]:
        """
        Run DLA simulation.
        
        Args:
            n_walkers: Number of random walking particles
            n_seeds: Number of initial sticky seeds
            max_iter: Maximum iterations (increased automatically if needed)
            injection_mode: 'random' or 'radial'
            injection_radius: Radius for radial injection (auto if None)
            show_progress: Show progress bar
            snapshot_interval: Save snapshot every N glued particles
            reinject_timeout: Steps before re-injecting stuck walkers (auto if None)
        
        Returns:
            Dictionary with results
        """
        N = self.N
        
        # Auto-calculate reinject timeout based on lattice size
        if reinject_timeout is None:
            reinject_timeout = max(N * 2, 1000)
        
        # Increase max_iter to ensure completion
        # Each walker needs on average O(N^2) steps to find aggregate
        effective_max_iter = max(max_iter, n_walkers * N // 2)
        
        grid = np.zeros((N, N), dtype=np.int32)
        
        # Initialize particle positions and status
        x = np.zeros(n_walkers, dtype=np.int32)
        y = np.zeros(n_walkers, dtype=np.int32)
        status = np.ones(n_walkers, dtype=np.int32)  # 1=mobile, 2=sticky
        walker_age = np.zeros(n_walkers, dtype=np.int32)  # Steps since last injection
        
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
                while grid[sx, sy] != 0:
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
            # Place walkers on grid
            for i in range(n_walkers):
                if grid[x[i], y[i]] == 0:
                    grid[x[i], y[i]] = 1
        else:
            x, y = initialize_walkers_no_overlap(N, n_walkers, grid)
        
        # Tracking
        n_glued_total = n_seeds
        iteration = 0
        
        # Snapshot storage
        snapshots = []
        glued_counts = []
        
        # Save initial state
        snapshots.append(grid.copy())
        glued_counts.append(n_seeds)
        
        if self.verbose:
            print(f"  Seeds: {n_seeds}")
            print(f"  Walkers: {n_walkers}")
            print(f"  Injection: {injection_mode}")
            if injection_mode == 'radial':
                print(f"  Injection radius: {injection_radius:.1f}")
            print(f"  Re-injection timeout: {reinject_timeout} steps")
        
        if show_progress:
            pbar = tqdm(total=n_walkers, desc="  Growing", unit=" particles")
            pbar.update(n_seeds)
        
        # Main simulation loop
        last_snapshot_count = n_seeds
        update_bounds_interval = 100
        min_x, max_x, min_y, max_y = find_aggregate_bounds(grid, N)
        
        while n_glued_total < n_walkers and iteration < effective_max_iter:
            # Update aggregate bounds periodically
            if iteration % update_bounds_interval == 0:
                min_x, max_x, min_y, max_y = find_aggregate_bounds(grid, N)
            
            n_glued_this_step = random_walk_step(
                x, y, status, grid, walker_age, N, n_walkers,
                reinject_timeout, min_x, max_x, min_y, max_y,
                injection_mode, injection_radius if injection_mode == 'radial' else 0.0,
                center_x, center_y
            )
            
            n_glued_total += n_glued_this_step
            iteration += 1
            
            if show_progress and n_glued_this_step > 0:
                pbar.update(n_glued_this_step)
            
            # Save snapshot at intervals
            if n_glued_total - last_snapshot_count >= snapshot_interval:
                snapshots.append(grid.copy())
                glued_counts.append(n_glued_total)
                last_snapshot_count = n_glued_total
        
        if show_progress:
            pbar.close()
        
        # Final snapshot
        if glued_counts[-1] != n_glued_total:
            snapshots.append(grid.copy())
            glued_counts.append(n_glued_total)
        
        # Compute fractal dimension
        final_grid = grid
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
            print(f"  Particles stuck: {n_glued_total}/{n_walkers}")
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
                'injection_radius': injection_radius if injection_mode == 'radial' else None,
                'reinject_timeout': reinject_timeout
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
