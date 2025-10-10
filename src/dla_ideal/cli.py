#!/usr/bin/env python
"""Command Line Interface for DLA Solver."""

import argparse
import sys
from pathlib import Path

from .core.solver import DLASolver
from .io.config_manager import ConfigManager
from .io.data_handler import DataHandler
from .visualization.animator import Animator
from .utils.logger import SimulationLogger
from .utils.timer import Timer


def print_header():
    """Print ASCII art header."""
    print("\n" + "=" * 70)
    print(" " * 10 + "DIFFUSION-LIMITED AGGREGATION SOLVER")
    print(" " * 20 + "Version 0.0.1")
    print("=" * 70)
    print("\n  Authors: Sandy H. S. Herho, Siti N. Kaban,")
    print("           Nurjanna J. Trilaksono, Rusmawan Suwarman,")
    print("           Dasapta E. Irawan")
    print("\n  License: MIT License")
    print("=" * 70 + "\n")


def normalize_scenario_name(scenario_name: str) -> str:
    """Convert scenario name to clean filename format."""
    clean = scenario_name.lower()
    clean = clean.replace(' - ', '_')
    clean = clean.replace('-', '_')
    clean = clean.replace(' ', '_')
    
    while '__' in clean:
        clean = clean.replace('__', '_')
    
    clean = clean.rstrip('_')
    return clean


def run_scenario(config: dict, output_dir: str = "outputs",
                verbose: bool = True, n_cores: int = None):
    """Run complete DLA simulation scenario."""
    scenario_name = config.get('scenario_name', 'simulation')
    clean_name = normalize_scenario_name(scenario_name)
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'=' * 60}")
    
    logger = SimulationLogger(clean_name, "logs", verbose)
    timer = Timer()
    timer.start("total")
    
    try:
        logger.log_parameters(config)
        
        with timer.time_section("solver_init"):
            if verbose:
                print("\n[1/3] Initializing DLA solver...")
            
            solver = DLASolver(
                N=config.get('lattice_size', 512),
                verbose=verbose,
                logger=logger,
                n_cores=n_cores
            )
        
        with timer.time_section("simulation"):
            if verbose:
                print("\n[2/3] Running DLA simulation...")
            
            result = solver.solve(
                n_walkers=config.get('n_walkers', 10000),
                n_seeds=config.get('n_seeds', 1),
                max_iter=config.get('max_iterations', 100000),
                injection_mode=config.get('injection_mode', 'random'),
                injection_radius=config.get('injection_radius', None),
                show_progress=verbose,
                snapshot_interval=config.get('snapshot_interval', 100)
            )
            
            logger.log_results(result)
            
            if verbose:
                print(f"      Particles: {result['n_particles']}")
                print(f"      Aggregates: {result['n_aggregates']}")
                if not result['fractal_dimension'] is None:
                    print(f"      D = {result['fractal_dimension']:.3f}")
        
        if config.get('save_netcdf', True):
            with timer.time_section("save_netcdf"):
                if verbose:
                    print("\n[3/3] Saving NetCDF and creating animation...")
                
                filename = f"{clean_name}.nc"
                DataHandler.save_netcdf(filename, result, config, output_dir)
                
                if verbose:
                    print(f"      Saved: {output_dir}/{filename}")
        
        if config.get('save_animation', True):
            with timer.time_section("animation"):
                filename = f"{clean_name}.gif"
                
                Animator.create_gif(
                    result,
                    filename,
                    output_dir,
                    scenario_name,
                    config.get('fps', 20),
                    config.get('colormap', 'hot')
                )
                
                if verbose:
                    print(f"      Saved: {output_dir}/{filename}")
        
        timer.stop("total")
        logger.log_timing(timer.get_times())
        
        if verbose:
            print(f"\n{'=' * 60}")
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print(f"  Total time: {timer.get_times()['total']:.2f} s")
            print(f"{'=' * 60}\n")
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"SIMULATION FAILED: {str(e)}")
            print(f"{'=' * 60}\n")
        raise
    
    finally:
        logger.finalize()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='DLA Solver with Numba Acceleration',
        epilog='Example: dla-simulate case1 --cores 8'
    )
    
    parser.add_argument(
        'case',
        nargs='?',
        choices=['case1', 'case2', 'case3', 'case4'],
        help='Test case to run'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all test cases sequentially'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    parser.add_argument(
        '--cores',
        type=int,
        default=None,
        help='Number of CPU cores (default: all)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print_header()
    
    if args.config:
        config = ConfigManager.load(args.config)
        run_scenario(config, args.output_dir, verbose, args.cores)
    
    elif args.all:
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        config_files = sorted(configs_dir.glob('case*.txt'))
        
        if not config_files:
            print("ERROR: No configuration files found in configs/")
            sys.exit(1)
        
        for i, cfg_file in enumerate(config_files, 1):
            if verbose:
                print(f"\n[Case {i}/{len(config_files)}] Running {cfg_file.stem}...")
            
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
    
    elif args.case:
        cfg_name = args.case
        configs_dir = Path(__file__).parent.parent.parent / 'configs'
        cfg_file = configs_dir / f'{cfg_name}.txt'
        
        if cfg_file.exists():
            config = ConfigManager.load(str(cfg_file))
            run_scenario(config, args.output_dir, verbose, args.cores)
        else:
            print(f"ERROR: Configuration file not found: {cfg_file}")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
