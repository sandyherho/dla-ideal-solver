"""Simulation logger for DLA simulations."""

import logging
from pathlib import Path
from datetime import datetime


class SimulationLogger:
    """Logger for DLA simulations."""
    
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
        """Log timing breakdown."""
        self.info("=" * 60)
        self.info("TIMING BREAKDOWN")
        self.info("=" * 60)
        
        for key, value in sorted(timing.items()):
            self.info(f"  {key}: {value:.3f} s")
        
        self.info("=" * 60)
    
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
