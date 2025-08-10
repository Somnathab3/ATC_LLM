#!/usr/bin/env python3
"""
ATC LLM CLI - Command Line Interface for LLM-BlueSky CDR System

This module provides a unified command-line interface for all major functions
of the Air Traffic Control LLM-driven Conflict Detection and Resolution system.

Usage:
    atc-llm --help                     # Show all available commands
    atc-llm simulate --help            # Show simulation options
    atc-llm batch --help               # Show batch processing options
    atc-llm compare --help             # Show comparison options
    atc-llm test --help                # Show testing options
    atc-llm health-check              # Run system health check
    atc-llm start-server              # Start API server
    atc-llm scat-baseline --help       # Generate SCAT baseline analysis
    atc-llm scat-llm-run --help        # Run LLM simulation with real-time option
    atc-llm enhanced-reporting --help  # Run enhanced reporting demonstration
"""

import argparse
import logging
import sys
import contextlib
import inspect
from pathlib import Path
from typing import Dict, Any, List

# Add src and bin to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "bin"))


def setup_logging(verbose: bool = False):
    """Configure logging for CLI operations."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@contextlib.contextmanager
def argv_context(new_argv: List[str]):
    """Context manager for safely modifying sys.argv."""
    original_argv = sys.argv.copy()
    try:
        sys.argv = new_argv
        yield
    finally:
        sys.argv = original_argv


def validate_path_exists(path: str, path_type: str = "path") -> bool:
    """Validate that a path exists."""
    if not Path(path).exists():
        print(f"[ERROR] {path_type.capitalize()} does not exist: {path}")
        return False
    return True


def safe_import(module_name: str, description: str = "module"):
    """Safely import a module with error handling."""
    try:
        if module_name == "repo_healthcheck":
            from repo_healthcheck import main
            return main
        elif module_name == "complete_llm_demo":
            from complete_llm_demo import main
            return main
        elif module_name == "batch_scat_llm_processor":
            from batch_scat_llm_processor import main
            return main
        elif module_name == "production_batch_processor":
            from production_batch_processor import main
            return main
        elif module_name == "demo_baseline_vs_llm":
            from demo_baseline_vs_llm import main
            return main
        elif module_name == "verify_llm_communication":
            from verify_llm_communication import main
            return main
        elif module_name == "visualize_conflicts":
            from visualize_conflicts import main
            return main
        elif module_name == "scat_baseline":
            from scat_baseline import main
            return main
        elif module_name == "scat_llm_run":
            from scat_llm_run import main
            return main
        else:
            raise ImportError(f"Unknown module: {module_name}")
    except ImportError as e:
        print(f"[ERROR] Failed to import {description}: {e}")
        print(f"   Make sure the required module is available in the bin directory")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error importing {description}: {e}")
        return None


def cmd_health_check(args: argparse.Namespace) -> int:
    """Run system health check to verify all components are working."""
    setup_logging(args.verbose)
    
    health_main = safe_import("repo_healthcheck", "health check module")
    if not health_main:
        return 1
    
    try:
        health_main()
        print("[OK] System health check completed successfully!")
        return 0
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_simulate_basic(args: argparse.Namespace) -> int:
    """Run basic simulation with generated scenarios."""
    setup_logging(args.verbose)
    
    demo_main = safe_import("complete_llm_demo", "demo module")
    if not demo_main:
        return 1
    
    try:
        print(f"[INIT] Starting basic simulation...")
        print(f"   Aircraft: {args.aircraft}")
        print(f"   Duration: {args.duration} minutes")
        print(f"   LLM Model: {args.llm_model}")
        
        demo_main()
        print("[OK] Basic simulation completed successfully!")
        return 0
    except Exception as e:
        print(f"[ERROR] Basic simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_simulate_scat(args: argparse.Namespace) -> int:
    """Run simulation with SCAT real-world data."""
    setup_logging(args.verbose)
    
    # Validate SCAT directory exists
    if not validate_path_exists(args.scat_dir, "SCAT directory"):
        return 1
    
    batch_main = safe_import("batch_scat_llm_processor", "SCAT batch processor")
    if not batch_main:
        return 1
    
    try:
        print(f"[INIT] Starting SCAT simulation...")
        print(f"   SCAT Directory: {args.scat_dir}")
        print(f"   Max Flights: {args.max_flights}")
        print(f"   Scenarios per Flight: {args.scenarios_per_flight}")
        print(f"   Output Directory: {args.output_dir}")
        
        # Use context manager for argv manipulation
        new_argv = [
            'batch_scat_llm_processor.py',
            '--scat-dir', args.scat_dir,
            '--max-flights', str(args.max_flights),
            '--scenarios-per-flight', str(args.scenarios_per_flight),
            '--output-dir', args.output_dir,
            '--llm-model', args.llm_model
        ]
        if args.verbose:
            new_argv.append('--verbose')
        
        with argv_context(new_argv):
            result = batch_main()
        
        if result == 0:
            print("[OK] SCAT simulation completed successfully!")
        else:
            print("[ERROR] SCAT simulation failed!")
        return result or 0
    except Exception as e:
        print(f"[ERROR] SCAT simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_batch_production(args: argparse.Namespace) -> int:
    """Run production batch processing with all safety checks."""
    setup_logging(args.verbose)
    
    # Validate SCAT directory exists
    if not validate_path_exists(args.scat_dir, "SCAT directory"):
        return 1
    
    prod_main = safe_import("production_batch_processor", "production batch processor")
    if not prod_main:
        return 1
    
    try:
        print(f"[FACTORY] Starting production batch processing...")
        print(f"   SCAT Directory: {args.scat_dir}")
        print(f"   Max Flights: {args.max_flights}")
        print(f"   Scenarios per Flight: {args.scenarios_per_flight}")
        
        # Use context manager for argv manipulation
        new_argv = [
            'production_batch_processor.py',
            '--scat-dir', args.scat_dir,
            '--max-flights', str(args.max_flights),
            '--scenarios-per-flight', str(args.scenarios_per_flight),
            '--output-dir', args.output_dir
        ]
        if args.skip_checks:
            new_argv.append('--skip-checks')
        if args.verbose:
            new_argv.append('--verbose')
            
        with argv_context(new_argv):
            result = prod_main()
        
        if result == 0:
            print("[OK] Production batch processing completed successfully!")
        else:
            print("[ERROR] Production batch processing failed!")
        return result or 0
    except Exception as e:
        print(f"[ERROR] Production batch processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_compare_baseline_llm(args: argparse.Namespace) -> int:
    """Compare baseline vs LLM performance."""
    setup_logging(args.verbose)
    
    # Validate SCAT path exists
    if not validate_path_exists(args.scat_path, "SCAT path"):
        return 1
    
    compare_main = safe_import("demo_baseline_vs_llm", "comparison module")
    if not compare_main:
        return 1
    
    try:
        print(f"[STATS] Starting baseline vs LLM comparison...")
        print(f"   SCAT Path: {args.scat_path}")
        print(f"   Max Flights: {args.max_flights}")
        print(f"   Time Window: {args.time_window} minutes")
        
        # Use context manager for argv manipulation
        new_argv = [
            'demo_baseline_vs_llm.py',
            '--scat-path', args.scat_path,
            '--max-flights', str(args.max_flights),
            '--time-window', str(args.time_window),
            '--output', args.output
        ]
        
        with argv_context(new_argv):
            compare_main()
        
        print("[OK] Baseline vs LLM comparison completed successfully!")
        return 0
    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_test_suite(args: argparse.Namespace) -> int:
    """Run comprehensive test suite."""
    setup_logging(args.verbose)
    
    try:
        import subprocess
        
        print(f"[EXPERIMENT] Running test suite...")
        
        if args.coverage:
            cmd = ['python', '-m', 'pytest', '--cov=src', '--cov-report=html', '--cov-report=term']
        else:
            cmd = ['python', '-m', 'pytest']
            
        if args.verbose:
            cmd.append('-v')
            
        if args.test_pattern:
            cmd.extend(['-k', args.test_pattern])
        
        # Use Path.cwd() if parent directory doesn't exist
        cwd_path = Path(__file__).parent
        if not cwd_path.exists():
            cwd_path = Path.cwd()
            
        result = subprocess.run(cmd, cwd=cwd_path)
        
        # Handle different exit codes properly
        if result.returncode == 0:
            print("[OK] Test suite completed successfully!")
            if args.coverage:
                print("[STATS] Coverage report generated in htmlcov/")
        elif result.returncode == 5:
            # pytest exit code 5 means no tests were collected/run
            print("[WARN]  No tests found matching the criteria")
            if args.test_pattern:
                print(f"   Pattern used: {args.test_pattern}")
            return 0  # This is not an error condition
        else:
            print("[ERROR] Some tests failed!")
            
        return result.returncode
    except FileNotFoundError:
        print("[ERROR] pytest not found. Please install pytest with: pip install pytest")
        if args.coverage:
            print("   Also install pytest-cov with: pip install pytest-cov")
        return 1
    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_start_server(args: argparse.Namespace) -> int:
    """Start the FastAPI server."""
    setup_logging(args.verbose)
    
    try:
        import uvicorn
        
        print(f"[GLOBAL] Starting API server...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Debug: {args.debug}")
        
        # Check if the API service module exists
        try:
            from src.api.service import app  # noqa: F401
            print("[OK] API service module loaded successfully")
        except ImportError as e:
            print(f"[ERROR] API service module not found: {e}")
            print("   Make sure src/api/service.py exists and is properly configured")
            return 1
        except Exception as e:
            print(f"[ERROR] Error loading API service: {e}")
            return 1
        
        uvicorn.run(
            "src.api.service:app",
            host=args.host,
            port=args.port,
            reload=args.debug,
            log_level="debug" if args.verbose else "info"
        )
        
        return 0
    except ImportError:
        print("[ERROR] uvicorn not found. Please install uvicorn with: pip install uvicorn")
        return 1
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_verify_llm(args: argparse.Namespace) -> int:
    """Verify LLM connectivity and functionality."""
    setup_logging(args.verbose)
    
    verify_main = safe_import("verify_llm_communication", "LLM verification module")
    if not verify_main:
        return 1
    
    try:
        print(f"[BOT] Verifying LLM connectivity...")
        print(f"   Model: {args.model}")
        
        # Use context manager for argv manipulation
        with argv_context(['verify_llm_communication.py']):
            verify_main()
        
        print("[OK] LLM verification completed successfully!")
        return 0
            
    except Exception as e:
        print(f"[ERROR] LLM verification failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_visualize_conflicts(args: argparse.Namespace) -> int:
    """Generate conflict visualization plots."""
    setup_logging(args.verbose)
    
    # Validate data file exists
    if not validate_path_exists(args.data_file, "data file"):
        return 1
    
    viz_main = safe_import("visualize_conflicts", "visualization module")
    if not viz_main:
        return 1
    
    try:
        print(f"[CHART] Generating conflict visualizations...")
        print(f"   Data file: {args.data_file}")
        
        # Call visualization main function with proper arguments
        # The updated visualize_conflicts script only needs --data-file
        with argv_context(['visualize_conflicts.py', '--data-file', args.data_file]):
            viz_main()
            
        print("[OK] Conflict visualizations generated successfully!")
        return 0
    except Exception as e:
        print(f"[ERROR] Visualization generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_scat_baseline(args: argparse.Namespace) -> int:
    """Generate SCAT baseline analysis."""
    setup_logging(args.verbose)
    
    # Validate SCAT root directory exists
    if not validate_path_exists(args.root, "SCAT root directory"):
        return 1
    
    baseline_main = safe_import("scat_baseline", "SCAT baseline module")
    if not baseline_main:
        return 1
    
    try:
        print(f"[BASELINE] Generating SCAT baseline analysis...")
        print(f"   SCAT root: {args.root}")
        print(f"   Ownship: {args.ownship}")
        print(f"   Radius: {args.radius}")
        print(f"   Altitude window: {args.altwin}")
        
        # Build command line arguments
        baseline_args = [
            'scat_baseline.py',
            '--root', args.root,
            '--ownship', args.ownship,
            '--radius', args.radius,
            '--altwin', str(args.altwin)
        ]
        
        if args.output:
            baseline_args.extend(['--output', args.output])
        if args.verbose:
            baseline_args.append('--verbose')
        
        with argv_context(baseline_args):
            result = baseline_main()
        
        if result == 0:
            print("[OK] SCAT baseline analysis completed successfully!")
        else:
            print("[ERROR] SCAT baseline analysis failed!")
        return result or 0
        
    except Exception as e:
        print(f"[ERROR] SCAT baseline analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_scat_llm_run(args: argparse.Namespace) -> int:
    """Run SCAT LLM simulation."""
    setup_logging(args.verbose)
    
    # Validate SCAT root directory exists
    if not validate_path_exists(args.root, "SCAT root directory"):
        return 1
    
    llm_run_main = safe_import("scat_llm_run", "SCAT LLM runner module")
    if not llm_run_main:
        return 1
    
    try:
        print(f"[LLM] Starting SCAT LLM simulation...")
        print(f"   SCAT root: {args.root}")
        print(f"   Ownship: {args.ownship}")
        print(f"   Intruders: {args.intruders}")
        print(f"   Real-time: {args.realtime}")
        print(f"   Time step: {args.dt_min} minutes")
        
        # Build command line arguments
        llm_args = [
            'scat_llm_run.py',
            '--root', args.root,
            '--ownship', args.ownship,
            '--intruders', args.intruders,
            '--dt-min', str(args.dt_min)
        ]
        
        if args.realtime:
            llm_args.append('--realtime')
        if args.duration:
            llm_args.extend(['--duration', str(args.duration)])
        if args.output:
            llm_args.extend(['--output', args.output])
        if args.verbose:
            llm_args.append('--verbose')
        
        with argv_context(llm_args):
            result = llm_run_main()
        
        if result == 0:
            print("[OK] SCAT LLM simulation completed successfully!")
        else:
            print("[ERROR] SCAT LLM simulation failed!")
        return result or 0
        
    except Exception as e:
        print(f"[ERROR] SCAT LLM simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_enhanced_reporting(args: argparse.Namespace) -> int:
    """Enhanced reporting demonstration command."""
    print(f"[INFO] Starting Enhanced Reporting Demo...")
    print(f"[INFO] Flights: {args.flights}, Intruders per flight: {args.intruders}")
    print(f"[INFO] Output directory: {args.output}")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("[INFO] Verbose logging enabled")
    
    try:
        # Import required modules
        from cdr.pipeline import CDRPipeline
        from cdr.schemas import ConfigurationSettings, MonteCarloParameters
        from cdr.monte_carlo_intruders import MonteCarloIntruderGenerator
        from cdr.scat_adapter import SCATAdapter
        import time
        from pathlib import Path
        
        print("[INFO] Creating configuration...")
        
        # Create configuration for enhanced reporting demo
        config = ConfigurationSettings(
            # Simulation settings
            polling_interval_min=1.0,
            lookahead_time_min=10.0,
            snapshot_interval_min=0.5,
            
            # Intruder detection
            max_intruders_in_prompt=5,
            intruder_proximity_nm=100.0,
            intruder_altitude_diff_ft=5000.0,
            
            # Safety parameters
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            safety_buffer_factor=1.2,
            
            # LLM settings
            llm_enabled=True,
            llm_model_name="llama-3.1-8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            
            # Resolution parameters
            max_resolution_angle_deg=45.0,
            max_altitude_change_ft=2000.0,
            
            # Enhanced validation
            enforce_ownship_only=True,
            max_climb_rate_fpm=3000.0,
            max_descent_rate_fpm=2500.0,
            realistic_turn_rates=True,
            check_fuel_burn=False,
            validate_airspace=False
        )
        
        print("[INFO] Initializing pipeline with enhanced reporting...")
        
        # Initialize pipeline with enhanced reporting
        pipeline = CDRPipeline(config)
        
        # Initialize SCAT adapter for real flight data
        scat_adapter = SCATAdapter()
        available_flights = scat_adapter.get_available_flights()
        
        if not available_flights:
            print("[ERROR] No SCAT flights available. Please check SCAT data.")
            return 1
        
        print(f"[INFO] Found {len(available_flights)} available SCAT flights")
        
        # Use first available flights (limit to requested number)
        selected_flights = available_flights[:args.flights]
        
        # Monte Carlo intruder generator
        mc_generator = MonteCarloIntruderGenerator()
        
        # Monte Carlo parameters for varied scenarios
        mc_params = MonteCarloParameters(
            num_scenarios=args.flights,
            intruders_per_scenario=args.intruders,
            max_spawn_distance_nm=80.0,
            min_spawn_distance_nm=20.0,
            altitude_variance_ft=3000.0,
            speed_variance_kt=100.0,
            spawn_time_window_min=15.0,
            conflict_probability=0.7,  # High probability for demonstration
            realistic_aircraft_types=True,
            airway_based_generation=False,
            weather_influence=False
        )
        
        print(f"[INFO] Monte Carlo Parameters: {mc_params.num_scenarios} scenarios, "
                f"{mc_params.intruders_per_scenario} intruders/scenario")
        
        # Generate scenarios with real flight data
        scenarios = []
        for i, flight_id in enumerate(selected_flights):
            flight_record = scat_adapter.get_flight_record(flight_id)
            if flight_record:
                scenario = mc_generator.generate_scenario_for_flight(
                    flight_record, mc_params, f"ENHANCED_DEMO_{i+1:03d}"
                )
                scenarios.append(scenario)
                print(f"[INFO] Generated scenario {scenario.scenario_id} for flight {flight_id}")
            else:
                print(f"[WARNING] Could not get flight record for {flight_id}")
        
        if not scenarios:
            print("[ERROR] No scenarios generated. Exiting.")
            return 1
        
        print(f"[INFO] Generated {len(scenarios)} scenarios for demonstration")
        
        # Run batch simulation with enhanced reporting
        print("[INFO] Starting batch simulation with enhanced metrics collection...")
        start_time = time.time()
        
        # Run scenarios
        for i, scenario in enumerate(scenarios, 1):
            print(f"[INFO] Running scenario {i}/{len(scenarios)}: {scenario.scenario_id}")
            
            # Get flight record
            flight_record = scat_adapter.get_flight_record(selected_flights[i-1])
            
            # Run single scenario
            metrics = pipeline._run_single_scenario(scenario, flight_record, max_cycles=50)
            
            # Log scenario results
            conflicts_detected = metrics.get('conflicts_detected', 0)
            successful_resolutions = metrics.get('successful_resolutions', 0)
            safety_violations = metrics.get('safety_violations', 0)
            min_separation = metrics.get('minimum_separation_nm', 999.0)
            
            print(f"[INFO] Scenario {scenario.scenario_id} completed:")
            print(f"[INFO]   - Conflicts detected: {conflicts_detected}")
            print(f"[INFO]   - Successful resolutions: {successful_resolutions}")
            print(f"[INFO]   - Safety violations: {safety_violations}")
            print(f"[INFO]   - Minimum separation: {min_separation:.2f} NM")
            
            if conflicts_detected > 0:
                success_rate = (successful_resolutions / conflicts_detected) * 100
                print(f"[INFO]   - Success rate: {success_rate:.1f}%")
        
        simulation_time = time.time() - start_time
        print(f"[INFO] Batch simulation completed in {simulation_time:.2f} seconds")
        
        # Generate enhanced reports
        print("[INFO] Generating enhanced reports...")
        csv_path, json_path = pipeline.generate_enhanced_reports(args.output)
        
        # Get summary statistics
        summary_stats = pipeline.get_enhanced_summary_statistics()
        
        # Display summary
        print("=" * 80)
        print("ENHANCED REPORTING SUMMARY")
        print("=" * 80)
        print(f"Total scenarios processed: {summary_stats.get('total_scenarios', 0)}")
        print(f"Total conflicts detected: {summary_stats.get('total_conflicts', 0)}")
        print(f"Conflicts resolved: {summary_stats.get('resolved_conflicts', 0)}")
        print(f"Overall success rate: {summary_stats.get('overall_success_rate', 0):.1f}%")
        print(f"Average time to action: {summary_stats.get('average_time_to_action_sec', 0):.2f} seconds")
        print(f"Average minimum separation: {summary_stats.get('average_min_separation_nm', 0):.2f} NM")
        print(f"Separation violations: {summary_stats.get('separation_violations', 0)}")
        
        # Engine usage breakdown
        engine_usage = summary_stats.get('engine_usage', {})
        print("Engine Usage Breakdown:")
        print(f"  - Horizontal: {engine_usage.get('horizontal', 0)}")
        print(f"  - Vertical: {engine_usage.get('vertical', 0)}")
        print(f"  - Deterministic: {engine_usage.get('deterministic', 0)}")
        print(f"  - Fallback: {engine_usage.get('fallback', 0)}")
        
        print(f"Average path deviation: {summary_stats.get('average_path_deviation_nm', 0):.2f} NM")
        print(f"Average resolution effectiveness: {summary_stats.get('average_resolution_effectiveness', 0):.3f}")
        
        print("=" * 80)
        print("REPORT FILES GENERATED")
        print("=" * 80)
        print(f"CSV Report: {csv_path}")
        print(f"JSON Report: {json_path}")
        print("")
        print("Features demonstrated:")
        print("✓ Per-conflict metrics: Resolved(Y/N), Min-sep(NM), Time-to-action, Engine used")
        print("✓ Per-scenario success rates and comprehensive tracking")
        print("✓ Waypoint vs heading resolution classification")
        print("✓ Timing analysis and operational impact assessment")
        print("✓ CSV/JSON batch run outputs with detailed metrics")
        print("✓ Reality comparison framework (prepared for SCAT vs BlueSky path analysis)")
        print("=" * 80)
        
        # Clean up
        pipeline.stop()
        
        print("[OK] Enhanced reporting demonstration completed successfully!")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Enhanced reporting demonstration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_wolfgang_metrics(args: argparse.Namespace) -> int:
    """Calculate Wolfgang (2011) aviation CDR metrics from CSV data."""
    setup_logging(args.verbose)
    
    try:
        from src.cdr.wolfgang_metrics import WolfgangMetricsCalculator
        
        print(f"[INIT] Starting Wolfgang (2011) metrics calculation...")
        print(f"   Sep threshold: {args.sep_threshold_nm} NM")
        print(f"   Alt threshold: {args.alt_threshold_ft} ft")
        print(f"   Margin: {args.margin_min} min")
        print(f"   Target separation: {args.sep_target_nm} NM")
        
        # Initialize calculator
        calculator = WolfgangMetricsCalculator(
            sep_threshold_nm=args.sep_threshold_nm,
            alt_threshold_ft=args.alt_threshold_ft,
            margin_min=args.margin_min,
            sep_target_nm=args.sep_target_nm
        )
        
        # Load input files
        from pathlib import Path
        
        if args.events:
            calculator.load_events_csv(Path(args.events))
        if args.baseline_sep:
            calculator.load_baseline_separation_csv(Path(args.baseline_sep))
        if args.resolved_sep:
            calculator.load_resolved_separation_csv(Path(args.resolved_sep))
        if args.planned_track:
            calculator.load_planned_track_csv(Path(args.planned_track))
        if args.resolved_track:
            calculator.load_resolved_track_csv(Path(args.resolved_track))
        
        # Compute metrics
        calculator.compute_metrics()
        
        # Export results
        output_path = Path(args.output)
        calculator.export_to_csv(output_path)
        
        # Print summary
        calculator.print_summary()
        
        print(f"\n[OK] Wolfgang metrics calculation completed successfully!")
        print(f"   Results saved to: {output_path}")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Wolfgang metrics calculation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog='atc-llm',
        description='ATC LLM-BlueSky CDR System - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  atc-llm health-check                                    # Check system health
  atc-llm simulate basic --aircraft 5 --duration 30      # Basic simulation
  atc-llm simulate scat --scat-dir /path/to/scat          # SCAT simulation
  atc-llm batch production --max-flights 10              # Production batch
  atc-llm compare --scat-path /path/to/scat               # Compare performance
  atc-llm test --coverage                                 # Run tests with coverage
  atc-llm server --port 8080                             # Start API server
  atc-llm scat-baseline --root /path/to/scat --ownship 100000.json --radius 100nm --altwin 5000
  atc-llm scat-llm-run --root /path/to/scat --ownship 100000.json --realtime --dt-min 1
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Health check command
    health_parser = subparsers.add_parser('health-check', help='Check system health')
    health_parser.set_defaults(func=cmd_health_check)
    
    # Simulation commands
    sim_parser = subparsers.add_parser('simulate', help='Run simulations')
    sim_subparsers = sim_parser.add_subparsers(dest='sim_type', help='Simulation types')
    
    # Basic simulation
    basic_parser = sim_subparsers.add_parser('basic', help='Run basic simulation')
    basic_parser.add_argument('--aircraft', type=int, default=5,
                             help='Number of aircraft (default: 5)')
    basic_parser.add_argument('--duration', type=int, default=30,
                             help='Simulation duration in minutes (default: 30)')
    basic_parser.add_argument('--llm-model', default='llama3.1:8b',
                             help='LLM model to use (default: llama3.1:8b)')
    basic_parser.set_defaults(func=cmd_simulate_basic)
    
    # SCAT simulation
    scat_parser = sim_subparsers.add_parser('scat', help='Run SCAT data simulation')
    scat_parser.add_argument('--scat-dir', required=True,
                            help='Directory containing SCAT files')
    scat_parser.add_argument('--max-flights', type=int, default=3,
                            help='Maximum flights to process (default: 3)')
    scat_parser.add_argument('--scenarios-per-flight', type=int, default=5,
                            help='Scenarios per flight (default: 5)')
    scat_parser.add_argument('--output-dir', default='Output',
                            help='Output directory (default: Output)')
    scat_parser.add_argument('--llm-model', default='llama3.1:8b',
                            help='LLM model to use (default: llama3.1:8b)')
    scat_parser.set_defaults(func=cmd_simulate_scat)
    
    # Batch processing commands
    batch_parser = subparsers.add_parser('batch', help='Run batch processing')
    batch_subparsers = batch_parser.add_subparsers(dest='batch_type', help='Batch types')
    
    # Production batch
    prod_parser = batch_subparsers.add_parser('production', help='Production batch processing')
    
    # Try to use a reasonable default SCAT directory
    default_scat_dir = 'scenarios/scat'  # Use relative path by default
    if Path(project_root / 'scenarios' / 'scat').exists():
        default_scat_dir = str(project_root / 'scenarios' / 'scat')
    elif Path('F:/SCAT_extracted').exists():
        default_scat_dir = 'F:/SCAT_extracted'
    elif Path('/data/SCAT_extracted').exists():
        default_scat_dir = '/data/SCAT_extracted'
        
    prod_parser.add_argument('--scat-dir', default=default_scat_dir,
                            help=f'SCAT directory (default: {default_scat_dir})')
    prod_parser.add_argument('--max-flights', type=int, default=5,
                            help='Maximum flights to process (default: 5)')
    prod_parser.add_argument('--scenarios-per-flight', type=int, default=5,
                            help='Scenarios per flight (default: 5)')
    prod_parser.add_argument('--output-dir', default='Output',
                            help='Output directory (default: Output)')
    prod_parser.add_argument('--skip-checks', action='store_true',
                            help='Skip prerequisite checks')
    prod_parser.set_defaults(func=cmd_batch_production)
    
    # Comparison commands
    comp_parser = subparsers.add_parser('compare', help='Compare system performance')
    
    # Try to use a reasonable default SCAT path
    default_scat_path = 'scenarios/scat'  # Use relative path by default
    if Path(project_root / 'scenarios' / 'scat').exists():
        default_scat_path = str(project_root / 'scenarios' / 'scat')
    elif Path('F:/SCAT_extracted').exists():
        default_scat_path = 'F:/SCAT_extracted'
    elif Path('/data/SCAT_extracted').exists():
        default_scat_path = '/data/SCAT_extracted'
        
    comp_parser.add_argument('--scat-path', default=default_scat_path,
                           help=f'Path to SCAT dataset (default: {default_scat_path})')
    comp_parser.add_argument('--max-flights', type=int, default=5,
                           help='Maximum flights to process (default: 5)')
    comp_parser.add_argument('--time-window', type=int, default=30,
                           help='Time window in minutes (default: 30)')
    comp_parser.add_argument('--output', default='demo_results.json',
                           help='Output file (default: demo_results.json)')
    comp_parser.set_defaults(func=cmd_compare_baseline_llm)
    
    # Testing commands
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--coverage', action='store_true',
                           help='Generate coverage report')
    test_parser.add_argument('--test-pattern', type=str,
                           help='Run tests matching pattern')
    test_parser.set_defaults(func=cmd_test_suite)
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='127.0.0.1',
                              help='Server host (default: 127.0.0.1)')
    server_parser.add_argument('--port', type=int, default=8000,
                              help='Server port (default: 8000)')
    server_parser.add_argument('--debug', action='store_true',
                              help='Enable debug mode with auto-reload')
    server_parser.set_defaults(func=cmd_start_server)
    
    # LLM verification command
    llm_parser = subparsers.add_parser('verify-llm', help='Verify LLM connectivity')
    llm_parser.add_argument('--model', default='llama3.1:8b',
                           help='LLM model to test (default: llama3.1:8b)')
    llm_parser.set_defaults(func=cmd_verify_llm)
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Generate conflict visualizations')
    viz_parser.add_argument('--data-file', required=True,
                           help='Data file to visualize')
    viz_parser.set_defaults(func=cmd_visualize_conflicts)
    
    # SCAT baseline command
    scat_baseline_parser = subparsers.add_parser('scat-baseline', help='Generate SCAT baseline analysis')
    scat_baseline_parser.add_argument('--root', required=True,
                                     help='Root directory containing SCAT files')
    scat_baseline_parser.add_argument('--ownship', required=True,
                                     help='Ownship SCAT file (e.g., 100000.json)')
    scat_baseline_parser.add_argument('--radius', default='100nm',
                                     help='Search radius (default: 100nm)')
    scat_baseline_parser.add_argument('--altwin', type=int, default=5000,
                                     help='Altitude window in feet (default: 5000)')
    scat_baseline_parser.add_argument('--output',
                                     help='Output file prefix (default: auto-generated)')
    scat_baseline_parser.set_defaults(func=cmd_scat_baseline)
    
    # SCAT LLM run command
    scat_llm_parser = subparsers.add_parser('scat-llm-run', help='Run LLM simulation with SCAT data')
    scat_llm_parser.add_argument('--root', required=True,
                                help='Root directory containing SCAT files')
    scat_llm_parser.add_argument('--ownship', required=True,
                                help='Ownship file or ID')
    scat_llm_parser.add_argument('--intruders', default='auto',
                                help='Intruders: "auto" or specific file (default: auto)')
    scat_llm_parser.add_argument('--realtime', action='store_true',
                                help='Enable real-time simulation')
    scat_llm_parser.add_argument('--dt-min', type=float, default=1.0,
                                help='Time step in minutes (default: 1.0)')
    scat_llm_parser.add_argument('--duration', type=float,
                                help='Maximum simulation duration in minutes')
    scat_llm_parser.add_argument('--output',
                                help='Output file for results (default: auto-generated)')
    scat_llm_parser.set_defaults(func=cmd_scat_llm_run)
    
    # Enhanced reporting command
    enhanced_reporting_parser = subparsers.add_parser('enhanced-reporting', 
                                                     help='Run enhanced reporting demonstration')
    enhanced_reporting_parser.add_argument('--flights', type=int, default=3,
                                          help='Number of flights to simulate (default: 3)')
    enhanced_reporting_parser.add_argument('--intruders', type=int, default=5,
                                          help='Number of intruders per flight (default: 5)')
    enhanced_reporting_parser.add_argument('--output', default='reports/enhanced_demo',
                                          help='Output directory for reports (default: reports/enhanced_demo)')
    enhanced_reporting_parser.add_argument('--verbose', action='store_true',
                                          help='Enable verbose logging')
    enhanced_reporting_parser.set_defaults(func=cmd_enhanced_reporting)
    
    # Wolfgang metrics command
    wolfgang_parser = subparsers.add_parser('wolfgang-metrics', 
                                           help='Calculate Wolfgang (2011) aviation CDR metrics')
    wolfgang_parser.add_argument('--events', type=str,
                                help='events.csv file path')
    wolfgang_parser.add_argument('--baseline-sep', type=str,
                                help='baseline_sep.csv file path')
    wolfgang_parser.add_argument('--resolved-sep', type=str,
                                help='resolved_sep.csv file path')
    wolfgang_parser.add_argument('--planned-track', type=str,
                                help='planned_track.csv file path')
    wolfgang_parser.add_argument('--resolved-track', type=str,
                                help='resolved_track.csv file path')
    wolfgang_parser.add_argument('--output', type=str, default='metrics_wolfgang.csv',
                                help='Output CSV file path (default: metrics_wolfgang.csv)')
    wolfgang_parser.add_argument('--sep-threshold-nm', type=float, default=5.0,
                                help='Horizontal LoS threshold in NM (default: 5.0)')
    wolfgang_parser.add_argument('--alt-threshold-ft', type=float, default=1000.0,
                                help='Vertical LoS threshold in feet (default: 1000.0)')
    wolfgang_parser.add_argument('--margin-min', type=float, default=5.0,
                                help='Time margin for alerting in minutes (default: 5.0)')
    wolfgang_parser.add_argument('--sep-target-nm', type=float, default=5.0,
                                help='Target separation for RE normalization in NM (default: 5.0)')
    wolfgang_parser.add_argument('--verbose', action='store_true',
                                help='Enable verbose logging')
    wolfgang_parser.set_defaults(func=cmd_wolfgang_metrics)
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if hasattr(args, 'sim_type') and not args.sim_type:
        parser.parse_args(['simulate', '--help'])
        return 1
        
    if hasattr(args, 'batch_type') and not args.batch_type:
        parser.parse_args(['batch', '--help'])
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[WARN]  Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
