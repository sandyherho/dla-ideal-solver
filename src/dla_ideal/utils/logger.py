"""Simulation logger for DLA simulations with enhanced timing breakdown."""

import logging
from pathlib import Path
from datetime import datetime


class SimulationLogger:
    """Logger for DLA simulations with clear timing separation."""
    
    def __init__(self, scenario_name: str, log_dir: str = "logs", 
                verbose: bool = True):
        """Initialize simulation logger."""
        self.scenario_name = scenario_name
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{scenario_name}.log"
        
        self.logger = self._setup_logger()
        self.warnings = []
        self.errors = []
    
    def _setup_logger(self):
        """Configure Python logging."""
        logger = logging.getLogger(f"dla_{self.scenario_name}")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        
        handler = logging.FileHandler(self.log_file, mode='w')
        handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger
    
    def info(self, msg: str):
        """Log informational message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
        self.warnings.append(msg)
        if self.verbose:
            print(f"  WARNING: {msg}")
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
        self.errors.append(msg)
        if self.verbose:
            print(f"  ERROR: {msg}")
    
    def log_parameters(self, params: dict):
        """Log simulation parameters."""
        self.info("=" * 60)
        self.info(f"SIMULATION PARAMETERS - {params.get('scenario_name', 'Unknown')}")
        self.info("=" * 60)
        
        for key, value in sorted(params.items()):
            self.info(f"  {key}: {value}")
        
        self.info("=" * 60)
    
    def log_results(self, results: dict):
        """Log simulation results."""
        self.info("=" * 60)
        self.info("SIMULATION RESULTS")
        self.info("=" * 60)
        
        self.info(f"  Particles stuck: {results['n_particles']}")
        self.info(f"  Aggregates formed: {results['n_aggregates']}")
        self.info(f"  Fractal dimension: {results['fractal_dimension']:.3f}")
        self.info(f"  Iterations: {results['params']['n_iterations']}")
        
        self.info("=" * 60)
    
    def log_timing(self, timing: dict):
        """Log timing breakdown with clear categorization."""
        self.info("=" * 70)
        self.info("TIMING BREAKDOWN")
        self.info("=" * 70)
        
        # Separate computation from output times
        computation_times = {}
        output_times = {}
        
        for key, value in timing.items():
            if key in ['solver_init', 'simulation']:
                computation_times[key] = value
            elif key in ['save_netcdf', 'animation', 'gif_rendering']:
                output_times[key] = value
            elif key == 'total':
                total_time = value
            else:
                computation_times[key] = value
        
        # Log computation times
        if computation_times:
            self.info("")
            self.info("  COMPUTATION TIME (Simulation):")
            self.info("  " + "-" * 60)
            for key, value in sorted(computation_times.items()):
                display_name = key.replace('_', ' ').title()
                self.info(f"    {display_name:.<40} {value:>8.2f} s")
            
            computation_total = sum(computation_times.values())
            self.info(f"    {'Subtotal Computation':.<40} {computation_total:>8.2f} s")
        
        # Log output times
        if output_times:
            self.info("")
            self.info("  OUTPUT TIME (Saving Files):")
            self.info("  " + "-" * 60)
            for key, value in sorted(output_times.items()):
                display_name = key.replace('_', ' ').title()
                if key == 'animation':
                    display_name = 'GIF Rendering & Saving'
                elif key == 'save_netcdf':
                    display_name = 'NetCDF Saving'
                self.info(f"    {display_name:.<40} {value:>8.2f} s")
            
            output_total = sum(output_times.values())
            self.info(f"    {'Subtotal Output':.<40} {output_total:>8.2f} s")
        
        # Log total time
        self.info("")
        self.info("  " + "=" * 60)
        if 'total' in timing:
            self.info(f"    {'TOTAL TIME':.<40} {timing['total']:>8.2f} s")
        self.info("  " + "=" * 60)
        
        self.info("=" * 70)
    
    def finalize(self):
        """Write final summary."""
        self.info("=" * 60)
        self.info("SIMULATION SUMMARY")
        self.info("=" * 60)
        
        if self.errors:
            self.info(f"  ERRORS: {len(self.errors)}")
        else:
            self.info("  ERRORS: None")
        
        if self.warnings:
            self.info(f"  WARNINGS: {len(self.warnings)}")
        else:
            self.info("  WARNINGS: None")
        
        self.info(f"  Log file: {self.log_file}")
        self.info("=" * 60)
