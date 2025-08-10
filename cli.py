#!/usr/bin/env python3
"""
ATC LLM CLI - Unified Command Line Interface for LLM-BlueSky CDR System

This module provides a single canonical command-line interface for all functions
of the Air Traffic Control LLM-driven Conflict Detection and Resolution system.

Usage:
    atc-llm health-check               # System health check
    atc-llm simulate --help            # Unified simulation (basic, SCAT, enhanced)
    atc-llm batch --help               # Batch processing operations
    atc-llm metrics --help             # Wolfgang + basic metrics calculation
    atc-llm report --help              # Enhanced reporting generation
    atc-llm verify-llm --help          # LLM connectivity verification
    atc-llm visualize --help           # Conflict visualization (optional)
"""

import argparse
import logging
import sys
import contextlib
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def setup_logging(verbose: bool = False):
    """Configure logging for CLI operations."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_path_exists(path: str, path_type: str = "path") -> bool:
    """Validate that a path exists."""
    if not Path(path).exists():
        print(f"[ERROR] {path_type.capitalize()} does not exist: {path}")
        return False
    return True


def cmd_health_check(args: argparse.Namespace) -> int:
    """Run system health check to verify all components are working."""
    setup_logging(args.verbose)
    
    try:
        print("[1/4] Module imports")
        
        # Test core module imports
        try:
            from src.cdr.pipeline import CDRPipeline
            from src.cdr.schemas import ConfigurationSettings
            from src.cdr.scat_adapter import SCATAdapter
            from src.cdr.bluesky_io import BlueSkyClient
            from src.cdr.llm_client import LLMClient
            print("   [OK] Core modules imported successfully")
        except Exception as e:
            print(f"   [ERROR] Module import failed: {e}")
            return 1
        
        print("[2/4] SCAT adapter")
        
        # Check SCAT adapter functionality
        scat_dir = "sample_data"
        if Path(scat_dir).exists():
            try:
                scat_adapter = SCATAdapter(scat_dir)
                summary = scat_adapter.get_flight_summary()
                print(f"   [OK] SCAT adapter working, flights: {summary.get('total_flights', 0)}")
            except Exception as e:
                print(f"   [WARNING] SCAT test failed: {e}")
        else:
            print("   [WARNING] No SCAT data directory found, skipping")
        
        print("[3/4] LLM connectivity")
        
        # Basic configuration for LLM test
        try:
            config = ConfigurationSettings(
                polling_interval_min=1.0,
                lookahead_time_min=10.0,
                snapshot_interval_min=1.5,
                max_intruders_in_prompt=5,
                intruder_proximity_nm=100.0,
                intruder_altitude_diff_ft=5000.0,
                trend_analysis_window_min=2.0,
                min_horizontal_separation_nm=5.0,
                min_vertical_separation_ft=1000.0,
                llm_enabled=True,
                llm_model_name="llama3.1:8b",
                llm_temperature=0.1,
                llm_max_tokens=2048,
                safety_buffer_factor=1.2,
                max_resolution_angle_deg=45.0,
                max_altitude_change_ft=2000.0,
                max_waypoint_diversion_nm=80.0,
                enforce_ownship_only=True,
                max_climb_rate_fpm=3000.0,
                max_descent_rate_fpm=3000.0,
                min_flight_level=100,
                max_flight_level=600,
                max_heading_change_deg=90.0,
                enable_dual_llm=True,
                horizontal_retry_count=2,
                vertical_retry_count=2,
                bluesky_host="localhost",
                bluesky_port=1337,
                bluesky_timeout_sec=5.0,
                fast_time=True,
                sim_accel_factor=1.0
            )
            
            # Test LLM connectivity
            llm_client = LLMClient(config)
            test_response = llm_client.generate_response("Test connectivity")
            if test_response:
                print(f"   [OK] LLM connectivity successful")
            else:
                print("   [WARNING] LLM response failed")
                
        except Exception as e:
            print(f"   [WARNING] LLM test failed: {e}")
        
        print("[4/4] Pipeline initialization")
        
        # Test pipeline initialization
        try:
            pipeline = CDRPipeline(config)
            print("   [OK] Pipeline initialized successfully")
        except Exception as e:
            print(f"   [WARNING] Pipeline test failed: {e}")
        
        print("[OK] System health check completed successfully!")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_simulate(args: argparse.Namespace) -> int:
    """Unified simulation command supporting basic, SCAT, and enhanced modes."""
    setup_logging(args.verbose)
    
    try:
        # Import core modules directly
        from src.cdr.pipeline import CDRPipeline
        from src.cdr.schemas import ConfigurationSettings, FlightRecord, AircraftState, MonteCarloParameters
        from src.cdr.monte_carlo_intruders import BatchIntruderGenerator
        from src.cdr.scat_adapter import SCATAdapter
        from datetime import datetime, timedelta
        from pathlib import Path
        import random
        
        print(f"[INIT] Starting unified simulation...")
        print(f"   Mode: {'SCAT-based' if args.scat_dir else 'Generated scenarios'}")
        if args.scat_dir:
            print(f"   SCAT Directory: {args.scat_dir}")
            print(f"   Ownship: {args.ownship}")
        print(f"   Duration: {args.duration_min} minutes")
        print(f"   Time Step: {args.dt_min} minutes")
        print(f"   Real-time: {args.real_time}")
        print(f"   LLM Model: {args.llm_model}")
        print(f"   Constraint Mode: {getattr(args, 'constraint_mode', 'both')}")
        if args.nav_aware:
            print(f"   Navigation-aware routing: Enabled")
        if args.spawn_dynamic_intruders:
            print(f"   Dynamic Intruders: Enabled")
        
        # Create modern configuration with user options
        config = ConfigurationSettings(
            polling_interval_min=args.dt_min,
            lookahead_time_min=10.0,
            snapshot_interval_min=args.dt_min,
            max_intruders_in_prompt=5,
            intruder_proximity_nm=100.0,
            intruder_altitude_diff_ft=5000.0,
            trend_analysis_window_min=2.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_enabled=True,
            llm_model_name=args.llm_model,
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=45.0,
            max_altitude_change_ft=2000.0,
            max_waypoint_diversion_nm=80.0 if args.nav_aware else 50.0,
            enforce_ownship_only=True,
            max_climb_rate_fpm=3000.0,
            max_descent_rate_fpm=3000.0,
            min_flight_level=100,
            max_flight_level=600,
            max_heading_change_deg=90.0,
            enable_dual_llm=True,
            horizontal_retry_count=2 if not hasattr(args, 'constraint_mode') or args.constraint_mode != 'vertical-only' else 0,
            vertical_retry_count=2 if not hasattr(args, 'constraint_mode') or args.constraint_mode != 'horizontal-only' else 0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0,
            fast_time=not args.real_time,
            sim_accel_factor=1.0 if args.real_time else 10.0
        )
        
        # Initialize pipeline
        print("[PIPELINE] Initializing CDR pipeline...")
        pipeline = CDRPipeline(config)
        
        if args.scat_dir:
            # SCAT-based simulation
            print(f"[SCAT] Loading SCAT data from {args.scat_dir}...")
            if not validate_path_exists(args.scat_dir, "SCAT directory"):
                return 1
                
            scat_adapter = SCATAdapter(args.scat_dir)
            
            if args.ownship:
                # Extract flight records for specific ownship
                flight_states = scat_adapter.extract_states_for_callsign(args.ownship)
                if not flight_states:
                    print(f"[ERROR] No states found for ownship {args.ownship}")
                    return 1
                
                # Convert to FlightRecord
                waypoints = [(state.latitude, state.longitude) for state in flight_states[:10]]  # First 10 points
                altitudes = [state.altitude for state in flight_states[:10]]
                timestamps = [state.timestamp for state in flight_states[:10]]
                
                flight_record = FlightRecord(
                    flight_id=args.ownship,
                    callsign=args.ownship,
                    aircraft_type=getattr(flight_states[0], 'aircraft_type', 'B737'),
                    waypoints=waypoints,
                    altitudes_ft=altitudes,
                    timestamps=timestamps,
                    cruise_speed_kt=450,
                    climb_rate_fpm=2000.0,
                    descent_rate_fpm=-1500.0,
                    scenario_type="scat_replay",
                    complexity_level=3
                )
                
                flight_records = [flight_record]
            else:
                # Use first few available flights
                available_flights = scat_adapter.get_available_callsigns()[:getattr(args, 'max_flights', 3)]
                flight_records = []
                
                for callsign in available_flights:
                    flight_states = scat_adapter.extract_states_for_callsign(callsign)
                    if flight_states:
                        waypoints = [(state.latitude, state.longitude) for state in flight_states[:10]]
                        altitudes = [state.altitude for state in flight_states[:10]]
                        timestamps = [state.timestamp for state in flight_states[:10]]
                        
                        flight_record = FlightRecord(
                            flight_id=callsign,
                            callsign=callsign,
                            aircraft_type=getattr(flight_states[0], 'aircraft_type', 'B737'),
                            waypoints=waypoints,
                            altitudes_ft=altitudes,
                            timestamps=timestamps,
                            cruise_speed_kt=450,
                            climb_rate_fpm=2000.0,
                            descent_rate_fpm=-1500.0,
                            scenario_type="scat_replay",
                            complexity_level=3
                        )
                        flight_records.append(flight_record)
                        
                        if len(flight_records) >= getattr(args, 'max_flights', 3):
                            break
        else:
            # Generated scenario simulation
            flight_record = FlightRecord(
                flight_id="OWNSHIP",
                callsign="OWNSHIP",
                aircraft_type="B737",
                waypoints=[
                    (51.5074, -0.1278),  # London
                    (52.5200, 13.4050),  # Berlin
                    (48.8566, 2.3522),   # Paris
                    (41.9028, 12.4964),  # Rome
                ],
                altitudes_ft=[35000, 35000, 35000, 35000],
                timestamps=[
                    datetime.now(),
                    datetime.now() + timedelta(minutes=15),
                    datetime.now() + timedelta(minutes=30),
                    datetime.now() + timedelta(minutes=45)
                ],
                cruise_speed_kt=450,
                climb_rate_fpm=2000.0,
                descent_rate_fpm=-1500.0,
                scenario_type="generated_demo",
                complexity_level=2
            )
            flight_records = [flight_record]
        
        # Create Monte Carlo parameters
        monte_carlo_params = MonteCarloParameters(
            scenarios_per_flight=getattr(args, 'scenarios_per_flight', 1),
            intruder_count_range=(getattr(args, 'aircraft', 3) - 1, getattr(args, 'aircraft', 5) - 1),
            conflict_zone_radius_nm=50.0,
            non_conflict_zone_radius_nm=200.0,
            altitude_spread_ft=10000.0,
            time_window_min=args.duration_min,
            conflict_timing_variance_min=5.0,
            conflict_probability=0.4 if args.spawn_dynamic_intruders else 0.3,
            speed_variance_kt=50.0,
            heading_variance_deg=45.0,
            realistic_aircraft_types=True,
            airway_based_generation=args.nav_aware,
            weather_influence=False
        )
        
        # Calculate max cycles based on duration and time step
        max_cycles = int(args.duration_min / args.dt_min)
        
        # Run simulation
        print(f"[SIMULATION] Running for {max_cycles} cycles...")
        batch_result = pipeline.run_for_flights(
            flight_records=flight_records,
            max_cycles=max_cycles,
            monte_carlo_params=monte_carlo_params
        )
        
        # Generate reports
        output_dir = Path("reports/unified_simulation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path, json_path = pipeline.generate_enhanced_reports(str(output_dir))
        summary = pipeline.get_enhanced_summary_statistics()
        
        print("[OK] Unified simulation completed successfully!")
        print(f"   Flights processed: {len(flight_records)}")
        print(f"   Total conflicts: {summary.get('total_conflicts', 0)}")
        print(f"   Resolved conflicts: {summary.get('resolved_conflicts', 0)}")
        print(f"   Success rate: {summary.get('overall_success_rate', 0):.1f}%")
        print(f"   Reports saved to: {output_dir}")
        print(f"     CSV: {csv_path}")
        print(f"     JSON: {json_path}")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Run batch processing operations."""
    setup_logging(args.verbose)
    
    try:
        from src.cdr.pipeline import CDRPipeline
        from src.cdr.schemas import ConfigurationSettings, MonteCarloParameters
        from src.cdr.scat_adapter import SCATAdapter
        from pathlib import Path
        
        print(f"[BATCH] Starting batch processing...")
        print(f"   SCAT Directory: {args.scat_dir}")
        print(f"   Max Flights: {args.max_flights}")
        print(f"   Scenarios per Flight: {args.scenarios_per_flight}")
        print(f"   Output Directory: {args.output_dir}")
        
        # Validate SCAT directory
        if not validate_path_exists(args.scat_dir, "SCAT directory"):
            return 1
        
        # Initialize SCAT adapter
        scat_adapter = SCATAdapter(args.scat_dir)
        available_flights = scat_adapter.get_available_callsigns()[:args.max_flights]
        
        if not available_flights:
            print("[ERROR] No flight data found in SCAT directory")
            return 1
        
        print(f"[BATCH] Processing {len(available_flights)} flights...")
        
        # Create configuration for batch processing
        config = ConfigurationSettings(
            polling_interval_min=1.0,
            lookahead_time_min=10.0,
            snapshot_interval_min=1.0,
            max_intruders_in_prompt=5,
            intruder_proximity_nm=100.0,
            intruder_altitude_diff_ft=5000.0,
            trend_analysis_window_min=2.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_enabled=True,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=45.0,
            max_altitude_change_ft=2000.0,
            max_waypoint_diversion_nm=80.0,
            enforce_ownship_only=True,
            max_climb_rate_fpm=3000.0,
            max_descent_rate_fpm=3000.0,
            min_flight_level=100,
            max_flight_level=600,
            max_heading_change_deg=90.0,
            enable_dual_llm=True,
            horizontal_retry_count=2,
            vertical_retry_count=2,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0,
            fast_time=True,
            sim_accel_factor=10.0
        )
        
        # Initialize pipeline
        pipeline = CDRPipeline(config)
        
        # Process flights in batch
        from src.cdr.schemas import FlightRecord
        from datetime import datetime, timedelta
        
        flight_records = []
        for callsign in available_flights:
            flight_states = scat_adapter.extract_states_for_callsign(callsign)
            if flight_states:
                waypoints = [(state.latitude, state.longitude) for state in flight_states[:20]]
                altitudes = [state.altitude for state in flight_states[:20]]
                timestamps = [state.timestamp for state in flight_states[:20]]
                
                flight_record = FlightRecord(
                    flight_id=callsign,
                    callsign=callsign,
                    aircraft_type=getattr(flight_states[0], 'aircraft_type', 'B737'),
                    waypoints=waypoints,
                    altitudes_ft=altitudes,
                    timestamps=timestamps,
                    cruise_speed_kt=450,
                    climb_rate_fpm=2000.0,
                    descent_rate_fpm=-1500.0,
                    scenario_type="batch_scat",
                    complexity_level=3
                )
                flight_records.append(flight_record)
        
        # Monte Carlo parameters for batch processing
        monte_carlo_params = MonteCarloParameters(
            scenarios_per_flight=args.scenarios_per_flight,
            intruder_count_range=(3, 7),
            conflict_zone_radius_nm=75.0,
            non_conflict_zone_radius_nm=200.0,
            altitude_spread_ft=15000.0,
            time_window_min=60.0,
            conflict_timing_variance_min=10.0,
            conflict_probability=0.4,
            speed_variance_kt=75.0,
            heading_variance_deg=60.0,
            realistic_aircraft_types=True,
            airway_based_generation=True,
            weather_influence=False
        )
        
        # Run batch simulation
        batch_result = pipeline.run_for_flights(
            flight_records=flight_records,
            max_cycles=30,
            monte_carlo_params=monte_carlo_params
        )
        
        # Generate reports
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path, json_path = pipeline.generate_enhanced_reports(str(output_dir))
        summary = pipeline.get_enhanced_summary_statistics()
        
        print("[OK] Batch processing completed successfully!")
        print(f"   Flights processed: {len(flight_records)}")
        print(f"   Total scenarios: {len(flight_records) * args.scenarios_per_flight}")
        print(f"   Total conflicts: {summary.get('total_conflicts', 0)}")
        print(f"   Resolved conflicts: {summary.get('resolved_conflicts', 0)}")
        print(f"   Success rate: {summary.get('overall_success_rate', 0):.1f}%")
        print(f"   Reports saved to: {output_dir}")
        print(f"     CSV: {csv_path}")
        print(f"     JSON: {json_path}")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_metrics(args: argparse.Namespace) -> int:
    """Calculate Wolfgang and basic metrics."""
    setup_logging(args.verbose)
    
    try:
        from src.cdr.wolfgang_metrics import WolfgangMetricsCalculator
        from pathlib import Path
        
        print("[METRICS] Calculating Wolfgang aviation CDR metrics...")
        
        # Initialize calculator
        calculator = WolfgangMetricsCalculator(
            sep_threshold_nm=args.sep_threshold_nm,
            alt_threshold_ft=args.alt_threshold_ft,
            margin_min=args.margin_min,
            sep_target_nm=args.sep_target_nm
        )
        
        # Load data files if provided
        if args.events:
            if not validate_path_exists(args.events, "Events file"):
                return 1
            calculator.load_events_data(args.events)
        
        if args.baseline_sep:
            if not validate_path_exists(args.baseline_sep, "Baseline separation file"):
                return 1
            calculator.load_baseline_separation_data(args.baseline_sep)
        
        if args.resolved_sep:
            if not validate_path_exists(args.resolved_sep, "Resolved separation file"):
                return 1
            calculator.load_resolved_separation_data(args.resolved_sep)
        
        if args.planned_track:
            if not validate_path_exists(args.planned_track, "Planned track file"):
                return 1
            calculator.load_planned_track_data(args.planned_track)
        
        if args.resolved_track:
            if not validate_path_exists(args.resolved_track, "Resolved track file"):
                return 1
            calculator.load_resolved_track_data(args.resolved_track)
        
        # Calculate metrics
        print("[METRICS] Computing Wolfgang metrics...")
        metrics = calculator.calculate_all_metrics()
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        calculator.save_metrics_csv(str(output_path))
        
        # Print summary
        calculator.print_summary()
        
        print(f"\n[OK] Metrics calculation completed successfully!")
        print(f"   Results saved to: {output_path}")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Metrics calculation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Generate enhanced reporting."""
    setup_logging(args.verbose)
    
    try:
        from src.cdr.pipeline import CDRPipeline
        from src.cdr.schemas import ConfigurationSettings, FlightRecord, MonteCarloParameters
        from datetime import datetime, timedelta
        from pathlib import Path
        import random
        import json
        
        print(f"[REPORT] Generating enhanced reports...")
        print(f"   Flights: {args.flights}")
        print(f"   Intruders per flight: {args.intruders}")
        print(f"   Output directory: {args.output}")
        
        # Create configuration
        config = ConfigurationSettings(
            polling_interval_min=1.0,
            lookahead_time_min=10.0,
            snapshot_interval_min=1.0,
            max_intruders_in_prompt=args.intruders,
            intruder_proximity_nm=100.0,
            intruder_altitude_diff_ft=5000.0,
            trend_analysis_window_min=2.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_enabled=True,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=45.0,
            max_altitude_change_ft=2000.0,
            max_waypoint_diversion_nm=80.0,
            enforce_ownship_only=True,
            max_climb_rate_fpm=3000.0,
            max_descent_rate_fpm=3000.0,
            min_flight_level=100,
            max_flight_level=600,
            max_heading_change_deg=90.0,
            enable_dual_llm=True,
            horizontal_retry_count=2,
            vertical_retry_count=2,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0,
            fast_time=True,
            sim_accel_factor=5.0
        )
        
        # Generate sample flights for reporting demo
        flight_records = []
        for i in range(args.flights):
            # Generate random waypoints around Europe
            base_lat, base_lon = 50.0 + random.uniform(-5, 5), 10.0 + random.uniform(-10, 10)
            waypoints = []
            altitudes = []
            timestamps = []
            
            for j in range(5):
                lat = base_lat + random.uniform(-1, 1)
                lon = base_lon + random.uniform(-2, 2)
                alt = 30000 + random.randint(-5000, 10000)
                ts = datetime.now() + timedelta(minutes=j*10)
                
                waypoints.append((lat, lon))
                altitudes.append(alt)
                timestamps.append(ts)
            
            flight_record = FlightRecord(
                flight_id=f"DEMO{i+1:03d}",
                callsign=f"DEMO{i+1:03d}",
                aircraft_type=random.choice(["B737", "A320", "B777", "A350"]),
                waypoints=waypoints,
                altitudes_ft=altitudes,
                timestamps=timestamps,
                cruise_speed_kt=450 + random.randint(-50, 50),
                climb_rate_fpm=2000.0,
                descent_rate_fpm=-1500.0,
                scenario_type="enhanced_demo",
                complexity_level=random.randint(2, 4)
            )
            flight_records.append(flight_record)
        
        # Monte Carlo parameters for enhanced reporting
        monte_carlo_params = MonteCarloParameters(
            scenarios_per_flight=3,
            intruder_count_range=(args.intruders-1, args.intruders+1),
            conflict_zone_radius_nm=60.0,
            non_conflict_zone_radius_nm=150.0,
            altitude_spread_ft=8000.0,
            time_window_min=45.0,
            conflict_timing_variance_min=8.0,
            conflict_probability=0.5,
            speed_variance_kt=60.0,
            heading_variance_deg=50.0,
            realistic_aircraft_types=True,
            airway_based_generation=True,
            weather_influence=True
        )
        
        # Initialize pipeline
        pipeline = CDRPipeline(config)
        
        # Run simulation for reporting
        print("[REPORT] Running simulation for enhanced reporting...")
        batch_result = pipeline.run_for_flights(
            flight_records=flight_records,
            max_cycles=20,
            monte_carlo_params=monte_carlo_params
        )
        
        # Generate enhanced reports
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path, json_path = pipeline.generate_enhanced_reports(str(output_dir))
        summary = pipeline.get_enhanced_summary_statistics()
        
        # Generate additional reports
        enhanced_summary = {
            "demonstration_metadata": {
                "timestamp": datetime.now().isoformat(),
                "flights_processed": len(flight_records),
                "scenarios_per_flight": monte_carlo_params.scenarios_per_flight,
                "total_scenarios": len(flight_records) * monte_carlo_params.scenarios_per_flight,
                "simulation_cycles": 20,
                "llm_model": config.llm_model_name
            },
            "performance_metrics": summary,
            "flight_summaries": [
                {
                    "flight_id": fr.flight_id,
                    "aircraft_type": fr.aircraft_type,
                    "complexity_level": fr.complexity_level
                } for fr in flight_records
            ]
        }
        
        enhanced_json_path = output_dir / "enhanced_summary.json"
        with open(enhanced_json_path, 'w') as f:
            json.dump(enhanced_summary, f, indent=2, default=str)
        
        print("[OK] Enhanced reporting completed successfully!")
        print(f"   Flights processed: {len(flight_records)}")
        print(f"   Total conflicts: {summary.get('total_conflicts', 0)}")
        print(f"   Resolved conflicts: {summary.get('resolved_conflicts', 0)}")
        print(f"   Success rate: {summary.get('overall_success_rate', 0):.1f}%")
        print(f"   Reports saved to: {output_dir}")
        print(f"     CSV: {csv_path}")
        print(f"     JSON: {json_path}")
        print(f"     Enhanced Summary: {enhanced_json_path}")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Enhanced reporting failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_verify_llm(args: argparse.Namespace) -> int:
    """Verify LLM connectivity and functionality."""
    setup_logging(args.verbose)
    
    try:
        from src.cdr.llm_client import LLMClient
        from src.cdr.schemas import ConfigurationSettings
        
        print(f"[LLM] Verifying LLM connectivity...")
        print(f"   Model: {args.model}")
        
        # Create configuration
        config = ConfigurationSettings(
            polling_interval_min=1.0,
            lookahead_time_min=10.0,
            snapshot_interval_min=1.0,
            max_intruders_in_prompt=5,
            intruder_proximity_nm=100.0,
            intruder_altitude_diff_ft=5000.0,
            trend_analysis_window_min=2.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_enabled=True,
            llm_model_name=args.model,
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=45.0,
            max_altitude_change_ft=2000.0,
            max_waypoint_diversion_nm=80.0,
            enforce_ownship_only=True,
            max_climb_rate_fpm=3000.0,
            max_descent_rate_fpm=3000.0,
            min_flight_level=100,
            max_flight_level=600,
            max_heading_change_deg=90.0,
            enable_dual_llm=True,
            horizontal_retry_count=2,
            vertical_retry_count=2,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0,
            fast_time=True,
            sim_accel_factor=1.0
        )
        
        # Test LLM connectivity
        llm_client = LLMClient(config)
        
        print("[LLM] Checking available models...")
        models = llm_client.get_available_models()
        print(f"   Available models: {models}")
        
        if args.model not in models:
            print(f"[WARNING] Requested model '{args.model}' not found in available models")
            if models:
                print(f"   Consider using: {models[0]}")
            return 1
        
        print(f"[LLM] Testing model '{args.model}' connectivity...")
        test_prompt = "Hello, this is a connectivity test. Please respond with 'Connection successful'."
        response = llm_client.generate_response(test_prompt)
        
        if response:
            print(f"   [OK] Model response received: {response[:100]}...")
            
            print("[LLM] Testing CDR-specific prompt...")
            cdr_prompt = """Given an aircraft conflict scenario, provide a resolution.
Ownship: B737 at FL350, heading 090, speed 450kt
Intruder: A320 at FL360, heading 270, speed 460kt
Predicted conflict in 8 minutes. Suggest resolution:"""
            
            cdr_response = llm_client.generate_response(cdr_prompt)
            if cdr_response:
                print(f"   [OK] CDR response received: {cdr_response[:150]}...")
                print("[OK] LLM verification completed successfully!")
                return 0
            else:
                print("   [ERROR] CDR-specific prompt failed")
                return 1
        else:
            print("   [ERROR] No response from LLM")
            return 1
            
    except Exception as e:
        print(f"[ERROR] LLM verification failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_visualize(args: argparse.Namespace) -> int:
    """Generate conflict visualizations (optional)."""
    setup_logging(args.verbose)
    
    try:
        from pathlib import Path
        
        print(f"[VISUALIZE] Generating visualizations...")
        print(f"   Data file: {args.data_file}")
        
        # Validate data file
        if not validate_path_exists(args.data_file, "Data file"):
            return 1
        
        # Try to import visualization module
        try:
            from src.cdr.visualization import ConflictVisualizer
            
            # Initialize visualizer
            visualizer = ConflictVisualizer()
            
            # Load data and generate visualizations
            visualizer.load_data(args.data_file)
            
            output_dir = Path("reports/visualizations")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate various visualizations
            visualizer.generate_conflict_map(str(output_dir / "conflict_map.html"))
            visualizer.generate_trajectory_plot(str(output_dir / "trajectories.png"))
            visualizer.generate_timeline_chart(str(output_dir / "timeline.html"))
            visualizer.generate_3d_view(str(output_dir / "3d_view.html"))
            
            print("[OK] Visualization generation completed successfully!")
            print(f"   Outputs saved to: {output_dir}")
            print(f"     Conflict Map: {output_dir / 'conflict_map.html'}")
            print(f"     Trajectories: {output_dir / 'trajectories.png'}")
            print(f"     Timeline: {output_dir / 'timeline.html'}")
            print(f"     3D View: {output_dir / '3d_view.html'}")
            
        except ImportError:
            print("[WARNING] Visualization module not available")
            print("   Basic text-based analysis only...")
            
            # Simple text-based analysis
            with open(args.data_file, 'r') as f:
                data = f.read()
            
            print(f"   File size: {len(data)} characters")
            print(f"   Lines: {len(data.splitlines())}")
            
            if 'conflict' in data.lower():
                conflicts = data.lower().count('conflict')
                print(f"   Conflicts mentioned: {conflicts} times")
            
        return 0
        
    except Exception as e:
        print(f"[ERROR] Visualization generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the unified argument parser with canonical subcommands."""
    parser = argparse.ArgumentParser(
        prog='atc-llm',
        description='ATC LLM-BlueSky CDR System - Unified Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  atc-llm health-check                                    # System health check
  atc-llm simulate --duration-min 30 --aircraft 5        # Basic simulation
  atc-llm simulate --scat-dir /path/to/scat --ownship NAX3580 --real-time
  atc-llm batch --scat-dir /path/to/scat --max-flights 10 # Batch processing
  atc-llm metrics --events events.csv --output metrics.csv # Wolfgang metrics
  atc-llm report --flights 5 --output reports/enhanced   # Enhanced reporting
  atc-llm verify-llm --model llama3.1:8b                 # LLM verification
  atc-llm visualize --data-file results.json             # Conflict visualization
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dump-config', action='store_true',
                       help='Print the effective configuration as JSON and exit')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Health check command
    health_parser = subparsers.add_parser('health-check', help='Check system health')
    health_parser.set_defaults(func=cmd_health_check)
    
    # Unified simulation command
    sim_parser = subparsers.add_parser('simulate', help='Run unified simulation (basic, SCAT, enhanced)')
    
    # Core simulation options
    sim_parser.add_argument('--scat-dir', type=str,
                           help='Path to SCAT directory (enables SCAT-based simulation)')
    sim_parser.add_argument('--ownship', type=str,
                           help='Ownship callsign when using SCAT data')
    sim_parser.add_argument('--duration-min', type=float, default=30.0,
                           help='Simulation duration in minutes (default: 30.0)')
    sim_parser.add_argument('--dt-min', type=float, default=1.0,
                           help='Time step in minutes (default: 1.0)')
    
    # Time mode options
    time_group = sim_parser.add_mutually_exclusive_group()
    time_group.add_argument('--real-time', action='store_true',
                           help='Run in real-time mode')
    time_group.add_argument('--fast-time', action='store_true', default=True,
                           help='Run in fast-time mode (default)')
    
    # Constraint mode options
    constraint_group = sim_parser.add_mutually_exclusive_group()
    constraint_group.add_argument('--horizontal-only', action='store_const', 
                                 dest='constraint_mode', const='horizontal-only',
                                 help='Only horizontal conflict resolution')
    constraint_group.add_argument('--vertical-only', action='store_const',
                                 dest='constraint_mode', const='vertical-only', 
                                 help='Only vertical conflict resolution')
    
    # Advanced options
    sim_parser.add_argument('--nav-aware', action='store_true',
                           help='Enable navigation-aware routing (DCT to named fixes)')
    sim_parser.add_argument('--spawn-dynamic-intruders', action='store_true',
                           help='Enable dynamic intruder spawning during simulation')
    sim_parser.add_argument('--generate-metrics', action='store_true',
                           help='Generate metrics after simulation')
    
    # Legacy compatibility options
    sim_parser.add_argument('--aircraft', type=int, default=5,
                           help='Number of aircraft for generated scenarios (default: 5)')
    sim_parser.add_argument('--max-flights', type=int, default=3,
                           help='Maximum flights to process from SCAT data (default: 3)')
    sim_parser.add_argument('--scenarios-per-flight', type=int, default=5,
                           help='Scenarios per flight for Monte Carlo (default: 5)')
    sim_parser.add_argument('--llm-model', default='llama3.1:8b',
                           help='LLM model to use (default: llama3.1:8b)')
    
    sim_parser.set_defaults(func=cmd_simulate)
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Run batch processing operations')
    batch_parser.add_argument('--scat-dir', required=True,
                             help='Directory containing SCAT files')
    batch_parser.add_argument('--max-flights', type=int, default=5,
                             help='Maximum flights to process (default: 5)')
    batch_parser.add_argument('--scenarios-per-flight', type=int, default=5,
                             help='Scenarios per flight (default: 5)')
    batch_parser.add_argument('--output-dir', default='Output',
                             help='Output directory (default: Output)')
    batch_parser.add_argument('--generate-metrics', action='store_true',
                             help='Generate metrics after batch processing')
    batch_parser.set_defaults(func=cmd_batch)
    
    # Metrics command (Wolfgang + basic)
    metrics_parser = subparsers.add_parser('metrics', help='Calculate Wolfgang and basic metrics')
    metrics_parser.add_argument('--events', type=str,
                               help='events.csv file path')
    metrics_parser.add_argument('--baseline-sep', type=str,
                               help='baseline_sep.csv file path')
    metrics_parser.add_argument('--resolved-sep', type=str,
                               help='resolved_sep.csv file path')
    metrics_parser.add_argument('--planned-track', type=str,
                               help='planned_track.csv file path')
    metrics_parser.add_argument('--resolved-track', type=str,
                               help='resolved_track.csv file path')
    metrics_parser.add_argument('--output', type=str, default='metrics_wolfgang.csv',
                               help='Output CSV file path (default: metrics_wolfgang.csv)')
    metrics_parser.add_argument('--sep-threshold-nm', type=float, default=5.0,
                               help='Horizontal LoS threshold in NM (default: 5.0)')
    metrics_parser.add_argument('--alt-threshold-ft', type=float, default=1000.0,
                               help='Vertical LoS threshold in feet (default: 1000.0)')
    metrics_parser.add_argument('--margin-min', type=float, default=5.0,
                               help='Time margin for alerting in minutes (default: 5.0)')
    metrics_parser.add_argument('--sep-target-nm', type=float, default=5.0,
                               help='Target separation for RE normalization in NM (default: 5.0)')
    metrics_parser.set_defaults(func=cmd_metrics)
    
    # Enhanced reporting command
    report_parser = subparsers.add_parser('report', help='Generate enhanced reporting')
    report_parser.add_argument('--flights', type=int, default=3,
                              help='Number of flights to simulate (default: 3)')
    report_parser.add_argument('--intruders', type=int, default=5,
                              help='Number of intruders per flight (default: 5)')
    report_parser.add_argument('--output', default='reports/enhanced_demo',
                              help='Output directory for reports (default: reports/enhanced_demo)')
    report_parser.set_defaults(func=cmd_report)
    
    # LLM verification command
    llm_parser = subparsers.add_parser('verify-llm', help='Verify LLM connectivity and functionality')
    llm_parser.add_argument('--model', default='llama3.1:8b',
                           help='LLM model to test (default: llama3.1:8b)')
    llm_parser.set_defaults(func=cmd_verify_llm)
    
    # Visualization command (optional)
    viz_parser = subparsers.add_parser('visualize', help='Generate conflict visualizations (optional)')
    viz_parser.add_argument('--data-file', required=True,
                           help='Data file to visualize')
    viz_parser.set_defaults(func=cmd_visualize)
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle --dump-config before any other processing
    if args.dump_config:
        return dump_config(args)
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[WARN] Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def dump_config(args: argparse.Namespace) -> int:
    """Dump the effective configuration as JSON."""
    import json
    
    try:
        from src.cdr.schemas import ConfigurationSettings
        
        # Create configuration with environment variable support
        config = ConfigurationSettings()
        
        # Override with CLI-specific values if provided
        config_dict = config.model_dump()
        
        if hasattr(args, 'llm_model') and args.llm_model:
            config_dict['llm_model_name'] = args.llm_model
        if hasattr(args, 'duration_min') and args.duration_min:
            config_dict['polling_interval_min'] = args.duration_min / 30  # Estimate
        if hasattr(args, 'dt_min') and args.dt_min:
            config_dict['snapshot_interval_min'] = args.dt_min
        if hasattr(args, 'real_time') and args.real_time:
            config_dict['fast_time'] = False
        if hasattr(args, 'constraint_mode') and args.constraint_mode:
            if args.constraint_mode == 'horizontal-only':
                config_dict['vertical_retry_count'] = 0
                config_dict['vertical_engine_enabled'] = False
            elif args.constraint_mode == 'vertical-only':
                config_dict['horizontal_retry_count'] = 0
                config_dict['horizontal_engine_enabled'] = False
        if hasattr(args, 'nav_aware') and args.nav_aware:
            config_dict['max_waypoint_diversion_nm'] = 80.0
        
        print(json.dumps(config_dict, indent=2, default=str))
        return 0
        
    except Exception as e:
        print(f"[ERROR] Failed to dump configuration: {e}")
        return 1
        return 1


if __name__ == '__main__':
    sys.exit(main())
