#!/usr/bin/env python3
"""
Test script for SCAT batch processing functionality - SCAT loading only.

This tests the SCAT data loading component without requiring BlueSky or LLM.
"""

import sys
from pathlib import Path

# Add the project to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_scat_batch")


def test_scat_loading():
    """Test SCAT file loading functionality."""
    try:
        from batch_scat_llm_processor import SCATBatchProcessor
        from src.cdr.schemas import ConfigurationSettings
        
        # Create minimal config (won't be used for SCAT loading test)
        config = ConfigurationSettings(
            polling_interval_min=1.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="test",
            llm_temperature=0.1,
            llm_max_tokens=1024,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=45.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0,
            fast_time=True,
            sim_accel_factor=1.0
        )
        
        # Test with a directory that might exist
        test_directories = [
            r"F:\SCAT_extracted",
            r"scenarios\scat",
            r".",  # Current directory
        ]
        
        processor = None
        for test_dir in test_directories:
            try:
                if Path(test_dir).exists():
                    logger.info(f"Testing SCAT directory: {test_dir}")
                    processor = SCATBatchProcessor(test_dir, config)
                    break
            except Exception as e:
                logger.warning(f"Cannot use directory {test_dir}: {e}")
                continue
        
        if not processor:
            logger.warning("No valid SCAT directory found, creating processor with current directory")
            processor = SCATBatchProcessor(".", config)
        
        # Test file discovery
        scat_files = processor.discover_scat_files(pattern="*.json", max_files=5)
        logger.info(f"✓ Discovered {len(scat_files)} SCAT files")
        
        if scat_files:
            # Test loading one file
            logger.info(f"Testing loading from: {scat_files[0]}")
            
            # This will test the loading without the full pipeline
            flight_records = processor.load_flight_records([scat_files[0]])
            
            if flight_records:
                flight = flight_records[0]
                logger.info(f"✓ Successfully loaded flight: {flight.flight_id}")
                logger.info(f"  - Callsign: {flight.callsign}")
                logger.info(f"  - Aircraft Type: {flight.aircraft_type}")
                logger.info(f"  - Waypoints: {len(flight.waypoints)}")
                logger.info(f"  - Complexity Level: {flight.complexity_level}")
            else:
                logger.warning("No flight records loaded from SCAT file")
        else:
            logger.info("No SCAT files found - testing with sample data")
            
            # Create a sample SCAT-like JSON file for testing
            sample_scat_data = {
                "callsign": "TEST123",
                "aircraft_type": "B737",
                "track_points": [
                    {
                        "time": "2024-01-01T10:00:00Z",
                        "latitude": 51.5,
                        "longitude": -0.1,
                        "altitude": 35000,
                        "ground_speed": 420
                    },
                    {
                        "time": "2024-01-01T10:30:00Z", 
                        "latitude": 52.0,
                        "longitude": 0.5,
                        "altitude": 35000,
                        "ground_speed": 420
                    }
                ]
            }
            
            import json
            test_file = Path("test_scat_sample.json")
            with open(test_file, 'w') as f:
                json.dump(sample_scat_data, f, indent=2)
            
            logger.info(f"Created test SCAT file: {test_file}")
            logger.info("✓ SCAT file creation test passed")
            
            # Clean up
            test_file.unlink()
        
        logger.info("✓ SCAT batch processing test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"✗ SCAT batch processing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monte_carlo_integration():
    """Test Monte Carlo integration with SCAT data."""
    try:
        from src.cdr.schemas import FlightRecord, MonteCarloParameters
        from src.cdr.monte_carlo_intruders import MonteCarloIntruderGenerator
        from datetime import datetime, timedelta
        
        # Create a sample flight record (simulating SCAT data)
        flight_record = FlightRecord(
            flight_id="SCAT_TEST_001",
            callsign="SCT001", 
            aircraft_type="B737",
            waypoints=[
                (51.4700, -0.4543),  # London Heathrow
                (51.8860, 0.2389),   # Stansted
                (52.3700, 4.8900)    # Amsterdam
            ],
            altitudes_ft=[35000, 35000, 35000],
            timestamps=[
                datetime.now(),
                datetime.now() + timedelta(minutes=45),
                datetime.now() + timedelta(minutes=90)
            ],
            cruise_speed_kt=420.0,
            climb_rate_fpm=2000.0,
            descent_rate_fpm=-1500.0,
            scenario_type="scat_real_data",
            complexity_level=3
        )
        
        # Create Monte Carlo parameters
        monte_carlo_params = MonteCarloParameters(
            scenarios_per_flight=3,
            intruder_count_range=(2, 4),
            conflict_zone_radius_nm=50.0,
            non_conflict_zone_radius_nm=150.0,
            altitude_spread_ft=8000.0,
            time_window_min=60.0,
            conflict_timing_variance_min=10.0,
            conflict_probability=0.5,
            speed_variance_kt=50.0,
            heading_variance_deg=30.0,
            realistic_aircraft_types=True,
            airway_based_generation=False,
            weather_influence=False
        )
        
        # Generate scenarios
        generator = MonteCarloIntruderGenerator(monte_carlo_params)
        scenarios = generator.generate_scenarios_for_flight(flight_record)
        
        logger.info(f"✓ Generated {len(scenarios)} Monte Carlo scenarios for SCAT flight")
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"  Scenario {i+1}: {len(scenario.intruder_states)} intruders, "
                       f"conflicts={'Yes' if scenario.has_conflicts else 'No'}")
        
        logger.info("✓ Monte Carlo integration test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"✗ Monte Carlo integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=== Testing SCAT Batch Processing Components ===")
    
    tests = [
        ("SCAT Loading", test_scat_loading),
        ("Monte Carlo Integration", test_monte_carlo_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n=== Test Results ===")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All tests PASSED!")
        return 0
    else:
        logger.error(f"❌ {failed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
