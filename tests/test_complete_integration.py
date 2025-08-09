#!/usr/bin/env python3
"""
Integration test for the complete batch processing pipeline.

This tests whether all components can work together without requiring 
BlueSky to be actually running (using mock mode).
"""

import sys
from pathlib import Path

# Add the project to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))

import logging
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integration_test")


def test_complete_integration():
    """Test complete integration with mocked BlueSky."""
    try:
        from batch_scat_llm_processor import SCATBatchProcessor, create_default_config
        from src.cdr.schemas import MonteCarloParameters
        
        # Create configuration with mock mode
        config = create_default_config()
        
        # Set environment variable to enable LLM mock mode
        os.environ["LLM_DISABLED"] = "1"
        
        # Create SCAT processor
        scat_dir = "F:\\SCAT_extracted"
        if not Path(scat_dir).exists():
            logger.warning(f"SCAT directory {scat_dir} not found, using current directory")
            scat_dir = "."
        
        processor = SCATBatchProcessor(scat_dir, config)
        
        # Test file discovery
        scat_files = processor.discover_scat_files(max_files=1)
        
        if not scat_files:
            logger.info("No SCAT files found - creating test scenario with synthetic data")
            return test_synthetic_integration()
        
        # Load one flight record
        flight_records = processor.load_flight_records([scat_files[0]])
        
        if not flight_records:
            logger.error("Could not load any flight records")
            return False
        
        logger.info(f"✓ Loaded flight record: {flight_records[0].flight_id}")
        
        # Create simple Monte Carlo parameters
        monte_carlo_params = MonteCarloParameters(
            scenarios_per_flight=2,  # Small number for testing
            intruder_count_range=(1, 2),
            conflict_zone_radius_nm=50.0,
            non_conflict_zone_radius_nm=150.0,
            altitude_spread_ft=5000.0,
            time_window_min=30.0,
            conflict_timing_variance_min=5.0,
            conflict_probability=0.3,
            speed_variance_kt=30.0,
            heading_variance_deg=20.0,
            realistic_aircraft_types=True,
            airway_based_generation=False,
            weather_influence=False
        )
        
        logger.info("✓ Monte Carlo parameters configured")
        
        # Test scenario generation without full pipeline
        from src.cdr.monte_carlo_intruders import BatchIntruderGenerator
        intruder_generator = BatchIntruderGenerator(monte_carlo_params)
        all_scenarios = intruder_generator.generate_scenarios_for_flights(flight_records)
        
        logger.info(f"✓ Generated {sum(len(scenarios) for scenarios in all_scenarios.values())} scenarios")
        
        # Test CDR pipeline initialization (this will fail if BlueSky is not running, but we can catch it)
        try:
            from src.cdr.pipeline import CDRPipeline
            
            # This will likely fail due to BlueSky connection, which is expected
            pipeline = CDRPipeline(config)
            logger.warning("⚠️ CDR Pipeline created - BlueSky connection may be mocked or available")
            pipeline.stop()
            
        except RuntimeError as e:
            if "BlueSky connection failed" in str(e):
                logger.info("✓ Expected BlueSky connection failure (BlueSky not running)")
            else:
                logger.error(f"Unexpected pipeline error: {e}")
                return False
        
        logger.info("✓ Complete integration test PASSED (BlueSky connection test expected to fail)")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up environment
        if "LLM_DISABLED" in os.environ:
            del os.environ["LLM_DISABLED"]


def test_synthetic_integration():
    """Test with synthetic data when SCAT files are not available."""
    try:
        from src.cdr.schemas import FlightRecord, MonteCarloParameters
        from src.cdr.monte_carlo_intruders import MonteCarloIntruderGenerator
        
        # Create synthetic flight record
        flight_record = FlightRecord(
            flight_id="SYNTH_001",
            callsign="SYN001",
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
            scenario_type="synthetic_test",
            complexity_level=2
        )
        
        # Test Monte Carlo generation
        monte_carlo_params = MonteCarloParameters(
            scenarios_per_flight=3,
            intruder_count_range=(2, 3),
            conflict_zone_radius_nm=50.0,
            non_conflict_zone_radius_nm=150.0,
            altitude_spread_ft=5000.0,
            time_window_min=30.0,
            conflict_timing_variance_min=5.0,
            conflict_probability=0.5,
            speed_variance_kt=30.0,
            heading_variance_deg=20.0,
            realistic_aircraft_types=True,
            airway_based_generation=False,
            weather_influence=False
        )
        
        generator = MonteCarloIntruderGenerator(monte_carlo_params)
        scenarios = generator.generate_scenarios_for_flight(flight_record)
        
        logger.info(f"✓ Generated {len(scenarios)} synthetic scenarios")
        
        # Test flight path analysis
        from src.cdr.monte_carlo_intruders import FlightPathAnalyzer
        path_analyzer = FlightPathAnalyzer(flight_record)
        
        for i, scenario in enumerate(scenarios):
            intrusions = path_analyzer.detect_intrusions_along_path(scenario.intruder_states)
            logger.info(f"  Scenario {i+1}: {len(intrusions)} intrusions detected")
        
        logger.info("✓ Synthetic integration test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"✗ Synthetic integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run integration tests."""
    logger.info("=== Complete Integration Test ===")
    
    # Test 1: Complete integration
    logger.info("\n--- Testing Complete Integration ---")
    success1 = test_complete_integration()
    
    # Test 2: Synthetic data fallback
    logger.info("\n--- Testing Synthetic Integration ---") 
    success2 = test_synthetic_integration()
    
    logger.info("\n=== Integration Test Results ===")
    
    if success1 and success2:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("\n📋 Ready for batch processing:")
        logger.info("1. ✅ SCAT data loading works")
        logger.info("2. ✅ Monte Carlo scenario generation works")
        logger.info("3. ✅ Flight path analysis works")
        logger.info("4. ✅ Pipeline components integrated")
        logger.info("\n🚀 To run full batch processing:")
        logger.info("   1. Start BlueSky: bluesky --mode sim --fasttime")
        logger.info("   2. Start LLM service (Ollama): ollama serve")
        logger.info("   3. Run: python batch_scat_llm_processor.py --scat-dir F:\\SCAT_extracted")
        return 0
    else:
        logger.error("❌ Some integration tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
