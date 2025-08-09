#!/usr/bin/env python3
"""
Example script showing how to run batch flight processing programmatically.
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add the project to path
script_dir = Path(__file__).parent
project_root = script_dir
sys.path.insert(0, str(project_root))

import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("batch_example")

def run_batch_flights_example():
    """Example of running batch flight processing."""
    
    from src.cdr.pipeline import CDRPipeline
    from src.cdr.schemas import (
        ConfigurationSettings, FlightRecord, MonteCarloParameters
    )
    
    # Step 1: Configure the system
    config = ConfigurationSettings(
        polling_interval_min=1.0,
        lookahead_time_min=10.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        llm_model_name="llama-3.1-8b",
        llm_temperature=0.1,
        llm_max_tokens=2048,
        safety_buffer_factor=1.2,
        max_resolution_angle_deg=45.0,
        max_altitude_change_ft=2000.0,
        bluesky_host="localhost",
        bluesky_port=1337,
        bluesky_timeout_sec=5.0,
        fast_time=True,
        sim_accel_factor=1.0
    )
    
    # Step 2: Create flight records (normally loaded from SCAT files)
    flight_records = [
        FlightRecord(
            flight_id="FLIGHT001",
            callsign="AAL123",
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
            scenario_type="example",
            complexity_level=2
        ),
        FlightRecord(
            flight_id="FLIGHT002", 
            callsign="BAW456",
            aircraft_type="A320",
            waypoints=[
                (48.8566, 2.3522),   # Paris CDG
                (50.9013, 4.4844),   # Brussels
                (52.3089, 4.7639)    # Amsterdam
            ],
            altitudes_ft=[37000, 37000, 37000],
            timestamps=[
                datetime.now() + timedelta(minutes=10),
                datetime.now() + timedelta(minutes=50),
                datetime.now() + timedelta(minutes=85)
            ],
            cruise_speed_kt=410.0,
            climb_rate_fpm=2000.0,
            descent_rate_fpm=-1500.0,
            scenario_type="example",
            complexity_level=3
        )
    ]
    
    # Step 3: Configure Monte Carlo parameters
    monte_carlo_params = MonteCarloParameters(
        scenarios_per_flight=5,  # Generate 5 scenarios per flight
        intruder_count_range=(2, 4),  # 2-4 intruders per scenario
        conflict_zone_radius_nm=30.0,
        non_conflict_zone_radius_nm=150.0,
        altitude_spread_ft=8000.0,
        time_window_min=45.0,
        conflict_timing_variance_min=8.0,
        conflict_probability=0.4,  # 40% chance of conflicts
        speed_variance_kt=40.0,
        heading_variance_deg=30.0,
        realistic_aircraft_types=True,
        airway_based_generation=False,
        weather_influence=False
    )
    
    # Step 4: Run the batch simulation
    logger.info(f"Starting batch simulation with {len(flight_records)} flights")
    logger.info(f"Total scenarios to process: {len(flight_records) * monte_carlo_params.scenarios_per_flight}")
    
    try:
        pipeline = CDRPipeline(config)
        
        # This is the main batch processing call
        batch_result = pipeline.run_for_flights(
            flight_records=flight_records,
            monte_carlo_params=monte_carlo_params
        )
        
        # Step 5: Analyze results
        logger.info("=== BATCH SIMULATION RESULTS ===")
        logger.info(f"Simulation ID: {batch_result.simulation_id}")
        logger.info(f"Total scenarios processed: {batch_result.total_scenarios}")
        logger.info(f"Conflicts detected: {batch_result.total_conflicts_detected}")
        logger.info(f"Resolutions attempted: {batch_result.total_resolutions_attempted}")
        logger.info(f"Successful resolutions: {batch_result.successful_resolutions}")
        
        if batch_result.total_resolutions_attempted > 0:
            success_rate = (batch_result.successful_resolutions / batch_result.total_resolutions_attempted) * 100
            logger.info(f"Resolution success rate: {success_rate:.1f}%")
        
        logger.info(f"Average resolution time: {batch_result.average_resolution_time_sec:.2f} seconds")
        logger.info(f"Minimum separation achieved: {batch_result.minimum_separation_achieved_nm:.2f} NM")
        logger.info(f"Safety violations: {batch_result.safety_violations}")
        
        # Step 6: Flight-level breakdown
        logger.info("\n=== FLIGHT-LEVEL RESULTS ===")
        for flight_id, metrics in batch_result.flight_results.items():
            logger.info(f"{flight_id}: {metrics['conflicts_detected']} conflicts, "
                       f"{metrics['successful_resolutions']}/{metrics['resolutions_attempted']} resolved")
        
        return batch_result
        
    except Exception as e:
        logger.error(f"Batch simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Clean up
        if 'pipeline' in locals() and pipeline:
            try:
                pipeline.stop()
            except Exception:
                pass

def load_flights_from_scat_directory() -> List['FlightRecord']:
    """Example of loading flights from SCAT files."""
    
    from scripts.enhanced_batch_scat_simulation import SCATFlightLoader
    from src.cdr.schemas import FlightRecord
    
    # Load flights from SCAT directory
    scat_dir = r"F:\SCAT_extracted"  # Adjust path as needed
    
    try:
        flight_loader = SCATFlightLoader(scat_dir)
        flight_records = flight_loader.load_multiple_flights(
            file_pattern="*.json",
            max_flights=3  # Limit for example
        )
        
        logger.info(f"Loaded {len(flight_records)} flights from SCAT directory")
        
        for flight in flight_records:
            logger.info(f"- {flight.flight_id}: {len(flight.waypoints)} waypoints, "
                       f"complexity {flight.complexity_level}")
        
        return flight_records
        
    except Exception as e:
        logger.error(f"Failed to load SCAT flights: {e}")
        return []

if __name__ == "__main__":
    logger.info("=== Batch Flight Processing Example ===")
    
    # Option 1: Use example flights (works without SCAT files)
    logger.info("\n1. Running with example flights...")
    result = run_batch_flights_example()
    
    # Option 2: Load from SCAT directory (requires SCAT files)
    logger.info("\n2. Loading flights from SCAT directory...")
    scat_flights = load_flights_from_scat_directory()
    
    if scat_flights:
        logger.info(f"Successfully loaded {len(scat_flights)} SCAT flights")
        logger.info("You can now run batch simulation with these flights using:")
        logger.info("pipeline.run_for_flights(scat_flights, monte_carlo_params)")
    else:
        logger.info("No SCAT flights loaded - check directory path")
    
    logger.info("\n=== Example Complete ===")
