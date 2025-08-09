"""Demonstration script for ASAS vs LLM conflict detection and resolution comparison.

This script:
- Sets up both ASAS and LLM systems
- Runs comparison scenarios with realistic traffic
- Shows detailed performance metrics
- Demonstrates proper BlueSky command execution
"""

import logging
import sys
import time
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.cdr.schemas import ConfigurationSettings, AircraftState
from src.cdr.bluesky_io import BlueSkyClient
from src.cdr.comparison_tool import ASASLLMComparator
from src.cdr.enhanced_llm_client import EnhancedLLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('asas_llm_demo.log')
    ]
)
logger = logging.getLogger(__name__)

def create_test_configuration() -> ConfigurationSettings:
    """Create test configuration for the comparison."""
    return ConfigurationSettings(
        # LLM settings
        llm_model_name="llama3.1:8b",
        llm_temperature=0.1,
        llm_max_tokens=2048,
        
        # Detection settings
        polling_interval_min=1.0,
        lookahead_time_min=10.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        
        # Resolution settings
        safety_buffer_factor=1.1,
        max_resolution_angle_deg=30.0,
        max_altitude_change_ft=2000.0,
        
        # BlueSky settings
        bluesky_host="127.0.0.1",
        bluesky_port=5555,
        bluesky_timeout_sec=5.0,
        fast_time=True,
        sim_accel_factor=1.0
    )

def create_conflict_scenario() -> List[AircraftState]:
    """Create a realistic conflict scenario with multiple aircraft."""
    from datetime import datetime
    
    # Scenario: Two aircraft on converging paths
    aircraft_states = [
        AircraftState(
            aircraft_id="UAL123",
            latitude=52.3736,  # London area
            longitude=4.8896,  # Amsterdam area
            altitude_ft=35000,
            heading_deg=90,    # Eastbound
            ground_speed_kt=450,
            vertical_speed_fpm=0,
            aircraft_type="B777",
            timestamp=datetime.fromisoformat("2025-08-08T12:00:00+00:00")
        ),
        AircraftState(
            aircraft_id="KLM456",
            latitude=52.3736,  # Same latitude
            longitude=5.0896,  # 0.2 degrees east
            altitude_ft=35000,  # Same altitude - conflict!
            heading_deg=270,   # Westbound - head-on conflict
            ground_speed_kt=420,
            vertical_speed_fpm=0,
            aircraft_type="A330",
            timestamp=datetime.fromisoformat("2025-08-08T12:00:00+00:00")
        ),
        AircraftState(
            aircraft_id="BAW789",
            latitude=52.2736,  # Slightly south
            longitude=4.9896,  # Middle longitude
            altitude_ft=36000,  # Different altitude - safe
            heading_deg=45,    # Northeast
            ground_speed_kt=460,
            vertical_speed_fpm=0,
            aircraft_type="A350",
            timestamp=datetime.fromisoformat("2025-08-08T12:00:00+00:00")
        )
    ]
    
    return aircraft_states

def create_vertical_conflict_scenario() -> List[AircraftState]:
    """Create a vertical conflict scenario."""
    from datetime import datetime
    
    aircraft_states = [
        AircraftState(
            aircraft_id="DLH100",
            latitude=50.0000,
            longitude=8.0000,
            altitude_ft=37000,
            heading_deg=180,   # Southbound
            ground_speed_kt=480,
            vertical_speed_fpm=0,
            aircraft_type="A380",
            timestamp=datetime.fromisoformat("2025-08-08T12:05:00+00:00")
        ),
        AircraftState(
            aircraft_id="AFR200",
            latitude=49.9000,  # Slightly south
            longitude=8.0000,  # Same longitude
            altitude_ft=37500,  # Only 500ft difference - vertical conflict
            heading_deg=0,     # Northbound
            ground_speed_kt=460,
            vertical_speed_fpm=0,
            aircraft_type="B787",
            timestamp=datetime.fromisoformat("2025-08-08T12:05:00+00:00")
        )
    ]
    
    return aircraft_states

def test_enhanced_llm_prompts():
    """Test the enhanced LLM prompts with proper formatting."""
    logger.info("=== Testing Enhanced LLM Prompts ===")
    
    config = create_test_configuration()
    llm_client = EnhancedLLMClient(config)
    
    # Test connection
    if not llm_client.validate_llm_connection():
        logger.error("LLM connection test failed")
        return False
    
    # Test detection prompt
    aircraft_states = create_conflict_scenario()
    ownship = aircraft_states[0]
    traffic = aircraft_states[1:]
    
    detection_prompt = llm_client.build_enhanced_detect_prompt(ownship, traffic, config)
    logger.info("Detection prompt generated successfully")
    logger.debug(f"Prompt length: {len(detection_prompt)} characters")
    
    # Test resolution prompt (simulate conflicts)
    from datetime import datetime
    from src.cdr.schemas import ConflictPrediction
    mock_conflicts = [
        ConflictPrediction(
            ownship_id="UAL123",
            intruder_id="KLM456",
            time_to_cpa_min=3.5,
            distance_at_cpa_nm=2.1,
            altitude_diff_ft=0,
            is_conflict=True,
            severity_score=0.85,
            conflict_type="horizontal",
            prediction_time=datetime.fromisoformat("2025-08-08T12:00:00+00:00"),
            confidence=0.9
        )
    ]
    
    resolution_prompt = llm_client.build_enhanced_resolve_prompt(ownship, mock_conflicts, config)
    logger.info("Resolution prompt generated successfully")
    logger.debug(f"Prompt length: {len(resolution_prompt)} characters")
    
    # Test actual LLM calls
    try:
        logger.info("Testing live LLM detection call...")
        response = llm_client._post_ollama(detection_prompt)
        parsed_detection = llm_client.parse_enhanced_detection_response(str(response))
        logger.info(f"Detection result: {parsed_detection.get('conflict', False)} conflicts detected")
        
        if parsed_detection.get('conflict', False):
            logger.info("Testing live LLM resolution call...")
            resolution_response = llm_client._post_ollama(resolution_prompt)
            parsed_resolution = llm_client.parse_enhanced_resolution_response(str(resolution_response))
            logger.info(f"Resolution command: {parsed_resolution.get('bluesky_command', 'None')}")
            
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return False
    
    logger.info("Enhanced LLM prompts test completed successfully")
    return True

def run_asas_llm_comparison():
    """Run the complete ASAS vs LLM comparison."""
    logger.info("=== Starting ASAS vs LLM Comparison ===")
    
    config = create_test_configuration()
    
    # Initialize BlueSky client
    bluesky_client = BlueSkyClient(config)
    
    if not bluesky_client.connect():
        logger.error("Failed to connect to BlueSky - comparison cannot proceed")
        return False
    
    # Initialize comparator
    comparator = ASASLLMComparator(bluesky_client, config)
    
    # Setup systems
    if not comparator.setup_comparison():
        logger.error("Failed to setup comparison systems")
        return False
    
    # Run multiple scenarios
    scenarios = [
        ("horizontal_conflict", create_conflict_scenario()),
        ("vertical_conflict", create_vertical_conflict_scenario()),
    ]
    
    for scenario_name, aircraft_states in scenarios:
        logger.info(f"Running scenario: {scenario_name}")
        
        try:
            # Clear BlueSky simulation
            bluesky_client.sim_reset()
            time.sleep(1)
            
            # Create aircraft in BlueSky
            for aircraft in aircraft_states:
                aircraft_type = aircraft.aircraft_type or "B737"  # Default aircraft type
                success = bluesky_client.create_aircraft(
                    aircraft.aircraft_id,
                    aircraft_type,
                    aircraft.latitude,
                    aircraft.longitude,
                    aircraft.heading_deg,
                    aircraft.altitude_ft,
                    aircraft.ground_speed_kt
                )
                if not success:
                    logger.warning(f"Failed to create aircraft {aircraft.aircraft_id}")
            
            # Allow simulation to stabilize
            time.sleep(2)
            
            # Run comparison
            result = comparator.run_comparison_scenario(aircraft_states, scenario_name)
            
            # Print scenario results
            logger.info(f"Scenario {scenario_name} results:")
            logger.info(f"  ASAS conflicts: {result.asas_conflicts_detected}")
            logger.info(f"  LLM conflicts: {result.llm_conflicts_detected}")
            logger.info(f"  Agreement rate: {result.detection_agreement_rate:.1f}%")
            logger.info(f"  ASAS resolutions: {result.asas_successful_resolutions}/{result.asas_resolutions_attempted}")
            logger.info(f"  LLM resolutions: {result.llm_successful_resolutions}/{result.llm_resolutions_attempted}")
            
        except Exception as e:
            logger.error(f"Error in scenario {scenario_name}: {e}")
    
    # Print overall summary
    comparator.print_comparison_summary()
    
    # Save results
    json_file, csv_file = comparator.save_comparison_results()
    logger.info(f"Results saved to {json_file} and {csv_file}")
    
    logger.info("=== ASAS vs LLM Comparison Complete ===")
    return True

def demonstrate_bluesky_commands():
    """Demonstrate proper BlueSky command execution."""
    logger.info("=== Demonstrating BlueSky Commands ===")
    
    config = create_test_configuration()
    bluesky_client = BlueSkyClient(config)
    llm_client = EnhancedLLMClient(config)
    
    if not bluesky_client.connect():
        logger.error("Failed to connect to BlueSky")
        return False
    
    # Reset simulation
    bluesky_client.sim_reset()
    time.sleep(1)
    
    # Create test aircraft
    from datetime import datetime
    test_aircraft = AircraftState(
        aircraft_id="TEST001",
        latitude=52.0,
        longitude=4.0,
        altitude_ft=35000,
        heading_deg=90,
        ground_speed_kt=450,
        vertical_speed_fpm=0,
        aircraft_type="B737",
        timestamp=datetime.fromisoformat("2025-08-08T12:00:00+00:00")
    )
    
    aircraft_type = test_aircraft.aircraft_type or "B737"  # Default aircraft type
    success = bluesky_client.create_aircraft(
        test_aircraft.aircraft_id,
        aircraft_type,
        test_aircraft.latitude,
        test_aircraft.longitude,
        test_aircraft.heading_deg,
        test_aircraft.altitude_ft,
        test_aircraft.ground_speed_kt
    )
    
    if not success:
        logger.error("Failed to create test aircraft")
        return False
    
    logger.info("Test aircraft created successfully")
    
    # Test various BlueSky commands
    test_commands = [
        "TEST001 HDG 120",  # Turn to heading 120
        "TEST001 ALT 37000",  # Climb to FL370
        "TEST001 SPD 480",    # Increase speed
    ]
    
    for command in test_commands:
        logger.info(f"Executing command: {command}")
        
        # Sanitize command using enhanced LLM client
        sanitized = llm_client._sanitize_bluesky_command(command)
        logger.info(f"Sanitized command: {sanitized}")
        
        # Execute command
        success = llm_client.execute_bluesky_command(sanitized, bluesky_client)
        logger.info(f"Command execution: {'SUCCESS' if success else 'FAILED'}")
        
        # Wait a moment between commands
        time.sleep(1)
    
    # Get aircraft states to verify commands
    states = bluesky_client.get_aircraft_states()
    if "TEST001" in states:
        state = states["TEST001"]
        logger.info(f"Final aircraft state:")
        logger.info(f"  Position: {state.get('lat', 0):.4f}, {state.get('lon', 0):.4f}")
        logger.info(f"  Altitude: {state.get('alt_ft', 0):.0f} ft")
        logger.info(f"  Heading: {state.get('hdg_deg', 0):.0f}°")
        logger.info(f"  Speed: {state.get('spd_kt', 0):.0f} kt")
    
    logger.info("BlueSky commands demonstration complete")
    return True

def main():
    """Main demonstration function."""
    logger.info("Starting ASAS vs LLM Comprehensive Demonstration")
    
    try:
        # Test 1: Enhanced LLM prompts
        if not test_enhanced_llm_prompts():
            logger.error("Enhanced LLM prompts test failed")
            return 1
        
        # Test 2: BlueSky command demonstration
        if not demonstrate_bluesky_commands():
            logger.error("BlueSky commands demonstration failed")
            return 1
        
        # Test 3: Full ASAS vs LLM comparison
        if not run_asas_llm_comparison():
            logger.error("ASAS vs LLM comparison failed")
            return 1
        
        logger.info("All demonstrations completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Demonstration failed with error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
