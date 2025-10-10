"""NetCDF Data Handler for DLA Results."""

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime


class DataHandler:
    """NetCDF output handler for DLA simulations."""
    
    @staticmethod
    def save_netcdf(filename: str, result: dict, metadata: dict,
                   output_dir: str = "outputs"):
        """Save DLA simulation results to NetCDF file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with Dataset(filepath, 'w', format='NETCDF4') as nc:
            
            # Dimensions
            N = result['params']['N']
            n_snapshots = len(result['snapshots'])
            n_radii = len(result['radii'])
            
            nc.createDimension('x', N)
            nc.createDimension('y', N)
            nc.createDimension('time', n_snapshots)
            nc.createDimension('radius', n_radii)
            
            # Coordinates
            nc_x = nc.createVariable('x', 'i4', ('x',), zlib=True, complevel=4)
            nc_x[:] = np.arange(N)
            nc_x.units = "lattice_units"
            nc_x.long_name = "x_coordinate"
            
            nc_y = nc.createVariable('y', 'i4', ('y',), zlib=True, complevel=4)
            nc_y[:] = np.arange(N)
            nc_y.units = "lattice_units"
            nc_y.long_name = "y_coordinate"
            
            # Final grid
            nc_grid = nc.createVariable('grid', 'i1', ('x', 'y'),
                                       zlib=True, complevel=6)
            nc_grid[:] = result['grid']
            nc_grid.units = "state"
            nc_grid.long_name = "lattice_state"
            nc_grid.description = "0=empty, 1=mobile, 2=stuck"
            
            # Snapshots
            nc_snaps = nc.createVariable('snapshots', 'i1', ('time', 'x', 'y'),
                                        zlib=True, complevel=6)
            nc_snaps[:] = result['snapshots']
            nc_snaps.units = "state"
            nc_snaps.long_name = "growth_snapshots"
            
            # Particle counts
            nc_counts = nc.createVariable('glued_counts', 'i4', ('time',),
                                         zlib=True, complevel=4)
            nc_counts[:] = result['glued_counts']
            nc_counts.units = "particles"
            nc_counts.long_name = "cumulative_stuck_particles"
            
            # Fractal analysis
            nc_r = nc.createVariable('radii', 'f4', ('radius',),
                                    zlib=True, complevel=4)
            nc_r[:] = result['radii']
            nc_r.units = "lattice_units"
            nc_r.long_name = "radius"
            
            nc_m = nc.createVariable('masses', 'f4', ('radius',),
                                    zlib=True, complevel=4)
            nc_m[:] = result['masses']
            nc_m.units = "particles"
            nc_m.long_name = "mass_within_radius"
            
            # Global attributes - Results
            nc.n_particles = int(result['n_particles'])
            nc.n_aggregates = int(result['n_aggregates'])
            nc.fractal_dimension = float(result['fractal_dimension'])
            nc.center_x = int(result['center'][0])
            nc.center_y = int(result['center'][1])
            
            # Global attributes - Parameters
            params = result['params']
            nc.lattice_size = int(params['N'])
            nc.n_walkers = int(params['n_walkers'])
            nc.n_seeds = int(params['n_seeds'])
            nc.n_iterations = int(params['n_iterations'])
            nc.injection_mode = str(params['injection_mode'])
            if params['injection_radius'] is not None:
                nc.injection_radius = float(params['injection_radius'])
            
            # Metadata
            nc.scenario = metadata.get('scenario_name', 'unknown')
            nc.created = datetime.now().isoformat()
            nc.software = "dla-ideal-solver"
            nc.version = "0.0.1"
            nc.method = "random_walk_lattice"
            nc.Conventions = "CF-1.8"
            nc.title = f"DLA Simulation: {metadata.get('scenario_name', 'unknown')}"
            nc.institution = "Samudera Sains Teknologi (SST) Ltd."
