"""
Test script to debug BlueSky aircraft state issues.
This script will create aircraft and test various scenarios without complex stepping.
"""
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import src modules
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import logging
import time
from src.cdr.schemas import ConfigurationSettings
from src.cdr.bluesky_io import BlueSkyClient

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bluesky_test")

def test_basic_aircraft_creation():
    """Test basic aircraft creation and state retrieval."""
    log.info("Starting BlueSky test...")
    
    # Configuration
    cfg = ConfigurationSettings(
        polling_interval_min=1.0,
        lookahead_time_min=10.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        llm_model_name="llama3.1:8b",
        llm_temperature=0.1,
        llm_max_tokens=512,
        safety_buffer_factor=1.1,
        max_resolution_angle_deg=30.0,
        max_altitude_change_ft=2000.0,
        bluesky_host="127.0.0.1",
        bluesky_port=5555,
        bluesky_timeout_sec=5.0,
        fast_time=True,
        sim_accel_factor=1.0
    )
    
    # Initialize BlueSky
    bs_client = BlueSkyClient(cfg)
    if not bs_client.connect():
        log.error("Failed to connect to BlueSky")
        return False
    
    log.info("BlueSky connected successfully")
    
    # Reset simulation
    bs_client.sim_reset()
    log.info("Simulation reset")
    
    # Check initial state (should be empty)
    initial_states = bs_client.get_aircraft_states()
    log.info(f"Initial aircraft count: {len(initial_states)}")
    
    # Create a test aircraft
    log.info("Creating test aircraft...")
    success = bs_client.create_aircraft("TEST1", "A320", 52.0, 4.0, 90.0, 30000.0, 400.0)
    if not success:
        log.error("Failed to create TEST1")
        return False
    
    log.info("Aircraft creation command sent")
    
    # Check traffic directly
    try:
        n_aircraft = getattr(bs_client.traf, "ntraf", 0)
        log.info(f"BlueSky traffic count: {n_aircraft}")
        
        if hasattr(bs_client.traf, 'id') and n_aircraft > 0:
            aircraft_ids = getattr(bs_client.traf, 'id', [])
            log.info(f"Aircraft IDs in traffic: {list(aircraft_ids[:n_aircraft])}")
    except Exception as e:
        log.error(f"Error checking traffic: {e}")
    
    # Wait and check states
    for i in range(10):
        time.sleep(1)
        states = bs_client.get_aircraft_states()
        log.info(f"Attempt {i+1}: Found {len(states)} aircraft")
        if states:
            for callsign, state in states.items():
                log.info(f"  {callsign}: lat={state['lat']:.4f}, lon={state['lon']:.4f}, "
                        f"hdg={state['hdg_deg']:.1f}°, spd={state['spd_kt']:.0f}kt")
            break
    else:
        log.error("Aircraft never appeared in states")
        return False
    
    # Test stepping without time acceleration
    log.info("Testing basic simulation step...")
    step_success = bs_client.step_minutes(0.1)  # Very small step
    log.info(f"Step result: {step_success}")
    
    # Check states after step
    states_after_step = bs_client.get_aircraft_states()
    log.info(f"After step: Found {len(states_after_step)} aircraft")
    
    if not states_after_step:
        log.error("Aircraft disappeared after simulation step!")
        return False
    
    log.info("Test passed: Aircraft survived simulation step")
    return True

def test_stack_commands():
    """Test basic BlueSky stack commands."""
    log.info("Testing BlueSky stack commands...")
    
    cfg = ConfigurationSettings(
        polling_interval_min=1.0,
        lookahead_time_min=10.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        llm_model_name="llama3.1:8b",
        llm_temperature=0.1,
        llm_max_tokens=512,
        safety_buffer_factor=1.1,
        max_resolution_angle_deg=30.0,
        max_altitude_change_ft=2000.0,
        bluesky_host="127.0.0.1",
        bluesky_port=5555,
        bluesky_timeout_sec=5.0,
        fast_time=True,
        sim_accel_factor=1.0
    )
    bs_client = BlueSkyClient(cfg)
    
    if not bs_client.connect():
        log.error("Failed to connect to BlueSky")
        return False
    
    # Test basic commands
    commands = [
        "RESET",
        "CRE TEST1 A320 52.0 4.0 90 30000 400",
        "POS",
        "TEST1 HDG 180"
    ]
    
    for cmd in commands:
        log.info(f"Testing command: {cmd}")
        try:
            result = bs_client.stack(cmd)
            log.info(f"Command result: {result}")
        except Exception as e:
            log.error(f"Command failed: {e}")
    
    return True

if __name__ == "__main__":
    log.info("Starting BlueSky debug tests...")
    
    # Test 1: Basic aircraft creation
    log.info("\\n=== Test 1: Basic Aircraft Creation ===")
    test1_result = test_basic_aircraft_creation()
    log.info(f"Test 1 result: {'PASS' if test1_result else 'FAIL'}")
    
    # Test 2: Stack commands
    log.info("\\n=== Test 2: Stack Commands ===")
    test2_result = test_stack_commands()
    log.info(f"Test 2 result: {'PASS' if test2_result else 'FAIL'}")
    
    log.info("\\nDebug tests completed")
