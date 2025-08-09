#!/usr/bin/env python3
"""
Test script for batch flight processing functionality.
"""

import sys
from pathlib import Path

# Add the parent directory to Python path so we can import src modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_batch")

def test_monte_carlo_intruders():
    """Test Monte Carlo intruder generation."""
    try:
        from src.cdr.monte_carlo_intruders import MonteCarloIntruderGenerator, FlightPathAnalyzer
        from src.cdr.schemas import FlightRecord, MonteCarloParameters
        
        # Create a simple test flight
        flight_record = FlightRecord(
            flight_id="TEST001",
            callsign="TEST001",
            aircraft_type="B737",
            waypoints=[(51.5, -0.1), (52.0, 0.5), (52.5, 1.0)],
            altitudes_ft=[35000, 35000, 35000],
            timestamps=[
                datetime.now(),
                datetime.now() + timedelta(minutes=30),
                datetime.now() + timedelta(minutes=60)
            ],
            cruise_speed_kt=420.0,
            climb_rate_fpm=2000.0,
            descent_rate_fpm=-1500.0,
            scenario_type="test",
            complexity_level=1
        )
        
        # Create Monte Carlo parameters
        params = MonteCarloParameters(
            scenarios_per_flight=2,
            intruder_count_range=(1, 3),
            conflict_zone_radius_nm=50.0,
            non_conflict_zone_radius_nm=200.0,
            altitude_spread_ft=10000.0,
            time_window_min=60.0,
            conflict_timing_variance_min=10.0,
            conflict_probability=0.5,
            speed_variance_kt=50.0,
            heading_variance_deg=45.0,
            realistic_aircraft_types=True,
            airway_based_generation=False,
            weather_influence=False
        )
        
        # Test generator
        generator = MonteCarloIntruderGenerator(params)
        scenarios = generator.generate_scenarios_for_flight(flight_record)
        
        logger.info(f"[OK] Generated {len(scenarios)} scenarios for test flight")
        
        # Test flight path analyzer
        analyzer = FlightPathAnalyzer(flight_record)
        logger.info(f"[OK] Flight path analyzer created with {len(analyzer.path_points)} points")
        
        if scenarios:
            # Test intrusion detection
            intrusions = analyzer.detect_intrusions_along_path(scenarios[0].intruder_states)
            logger.info(f"[OK] Detected {len(intrusions)} intrusions in first scenario")
        
        logger.info("[OK] Monte Carlo intruder generation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Monte Carlo test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_pipeline():
    """Test batch pipeline functionality."""
    try:
        from src.cdr.pipeline import CDRPipeline
        from src.cdr.schemas import ConfigurationSettings, FlightRecord, MonteCarloParameters
        from datetime import datetime, timedelta
        
        # Create test configuration
        config = ConfigurationSettings(
            polling_interval_min=1.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            fast_time=True,
            sim_accel_factor=1.0
        )
        
        # Create test flights
        flight_records = [
            FlightRecord(
                flight_id="TEST001",
                callsign="TEST001",
                aircraft_type="B737",
                waypoints=[(51.5, -0.1), (52.0, 0.5)],
                altitudes_ft=[35000, 35000],
                timestamps=[datetime.now(), datetime.now() + timedelta(minutes=30)],
                cruise_speed_kt=420.0,
                climb_rate_fpm=2000.0,
                descent_rate_fpm=-1500.0
            ),
            FlightRecord(
                flight_id="TEST002",
                callsign="TEST002",
                aircraft_type="A320",
                waypoints=[(50.5, -1.1), (51.0, -0.5)],
                altitudes_ft=[36000, 36000],
                timestamps=[datetime.now(), datetime.now() + timedelta(minutes=25)],
                cruise_speed_kt=400.0,
                climb_rate_fpm=2000.0,
                descent_rate_fpm=-1500.0
            )
        ]
        
        # Create Monte Carlo parameters
        monte_carlo_params = MonteCarloParameters(
            scenarios_per_flight=2,
            intruder_count_range=(1, 2),
            conflict_zone_radius_nm=50.0,
            non_conflict_zone_radius_nm=200.0,
            altitude_spread_ft=10000.0,
            time_window_min=60.0,
            conflict_timing_variance_min=10.0,
            conflict_probability=0.3,
            speed_variance_kt=50.0,
            heading_variance_deg=45.0,
            realistic_aircraft_types=True,
            airway_based_generation=False,
            weather_influence=False
        )
        
        # Test pipeline creation
        pipeline = CDRPipeline(config)
        logger.info("[OK] CDR Pipeline created successfully")
        
        # Test run_for_flights method exists and is callable
        if hasattr(pipeline, 'run_for_flights'):
            logger.info("[OK] run_for_flights method exists")
            
            # Note: We don't actually run the simulation in this test
            # as it would require BlueSky connection
            logger.info("[OK] Batch pipeline test PASSED (method verification)")
            return True
        else:
            logger.error("[FAIL] run_for_flights method not found")
            return False
        
    except Exception as e:
        logger.error(f"[FAIL] Batch pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_schema_validation():
    """Test schema validation for new classes."""
    try:
        from src.cdr.schemas import FlightRecord, MonteCarloParameters, IntruderScenario, BatchSimulationResult
        from datetime import datetime, timedelta
        
        # Test FlightRecord
        flight_record = FlightRecord(
            flight_id="TEST001",
            callsign="TEST001",
            aircraft_type="B737",
            waypoints=[(51.5, -0.1), (52.0, 0.5)],
            altitudes_ft=[35000, 35000],
            timestamps=[datetime.now(), datetime.now() + timedelta(minutes=30)],
            cruise_speed_kt=420.0,
            climb_rate_fpm=2000.0,
            descent_rate_fpm=-1500.0
        )
        logger.info("[OK] FlightRecord validation passed")
        
        # Test MonteCarloParameters
        params = MonteCarloParameters(
            scenarios_per_flight=10,
            intruder_count_range=(1, 5),
            conflict_zone_radius_nm=50.0,
            non_conflict_zone_radius_nm=200.0,
            altitude_spread_ft=10000.0,
            time_window_min=60.0,
            conflict_timing_variance_min=10.0,
            conflict_probability=0.3,
            speed_variance_kt=50.0,
            heading_variance_deg=45.0,
            realistic_aircraft_types=True,
            airway_based_generation=False,
            weather_influence=False
        )
        logger.info("[OK] MonteCarloParameters validation passed")
        
        # Test BatchSimulationResult
        batch_result = BatchSimulationResult(
            simulation_id="test_sim",
            start_time=datetime.now(),
            flight_records=["TEST001"],
            scenarios_per_flight=10,
            total_scenarios=10,
            total_conflicts_detected=0,
            total_resolutions_attempted=0,
            successful_resolutions=0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            average_resolution_time_sec=0.0,
            minimum_separation_achieved_nm=5.0,
            safety_violations=0
        )
        logger.info("[OK] BatchSimulationResult validation passed")
        
        logger.info("[OK] Schema validation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Schema validation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("=== Testing Batch Flight Processing Implementation ===")
    
    tests = [
        ("Schema Validation", test_schema_validation),
        ("Monte Carlo Intruders", test_monte_carlo_intruders),
        ("Batch Pipeline", test_batch_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"[OK] {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"[FAIL] {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"[FAIL] {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logger.info("[SUCCESS] All tests PASSED!")
        return 0
    else:
        logger.error(f"[ERROR] {failed} test(s) FAILED")
        return 1

if __name__ == "__main__":
    exit(main())
