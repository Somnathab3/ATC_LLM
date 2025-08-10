#!/usr/bin/env python3
"""
Test script to debug LLM responses with detailed logging
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.cdr.scat_adapter import SCATAdapter
from src.cdr.schemas import AircraftState, ConfigurationSettings
from src.cdr.bluesky_io import BlueSkyClient, BSConfig
from src.cdr.llm_client import LlamaClient
from src.cdr.detect import predict_conflicts
from datetime import datetime

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_llm_debug():
    """Test LLM with debug logging to see raw responses"""
    
    # Create configuration with all required parameters
    config = ConfigurationSettings(
        # Timing settings  
        polling_interval_min=1.0,
        lookahead_time_min=10.0,
        snapshot_interval_min=1.0,
        
        # PromptBuilderV2 settings
        max_intruders_in_prompt=5,
        intruder_proximity_nm=100.0,
        intruder_altitude_diff_ft=5000.0,
        trend_analysis_window_min=2.0,
        
        # Separation standards
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        
        # LLM settings
        llm_enabled=True,
        llm_model_name="llama3.1:8b",
        llm_temperature=0.1,
        llm_max_tokens=512,
        
        # Safety settings
        safety_buffer_factor=1.2,
        max_resolution_angle_deg=30.0,
        max_altitude_change_ft=2000.0,
        
        # Enhanced validation settings
        enforce_ownship_only=True,
        max_climb_rate_fpm=3000.0,
        max_descent_rate_fpm=3000.0,
        min_flight_level=100,
        max_flight_level=600,
        max_heading_change_deg=90.0,
        
        # Dual LLM engine settings
        enable_dual_llm=True,
        horizontal_retry_count=2,
        vertical_retry_count=2,
        
        # BlueSky integration
        bluesky_host="127.0.0.1",
        bluesky_port=5555,
        bluesky_timeout_sec=5.0,
        
        # Fast-time simulation
        fast_time=True,
        sim_accel_factor=1.0
    )
    
    # Create LLM client
    llm_client = LlamaClient(config)
    
    # Create mock aircraft states for testing
    ownship = AircraftState(
        aircraft_id="TEST001",
        timestamp=datetime.now(),
        latitude=54.300,
        longitude=14.720,
        altitude_ft=35000,
        ground_speed_kt=420,
        heading_deg=45,
        vertical_speed_fpm=0
    )
    
    intruder = AircraftState(
        aircraft_id="TEST002", 
        timestamp=datetime.now(),
        latitude=54.310,
        longitude=14.730,
        altitude_ft=35000,
        ground_speed_kt=430,
        heading_deg=225,
        vertical_speed_fpm=0
    )
    
    # Create conflict
    conflicts = predict_conflicts(ownship, [intruder])
    
    if conflicts:
        print(f"Created test conflict: {conflicts[0].ownship_id} vs {conflicts[0].intruder_id}")
        print(f"Distance at CPA: {conflicts[0].distance_at_cpa_nm:.2f} NM")
        print(f"Time to CPA: {conflicts[0].time_to_cpa_min:.2f} min")
        
        # Call LLM with debug logging
        print("\n=== CALLING LLM WITH DEBUG LOGGING ===")
        resolution = llm_client.resolve_conflict(conflicts[0], [ownship, intruder], config)
        
        print(f"\n=== FINAL RESOLUTION ===")
        print(f"Action: {resolution.action}")
        print(f"Params: {resolution.params}")
        print(f"Target: {resolution.target_aircraft}")
        print(f"Rationale: {getattr(resolution, 'rationale', 'N/A')}")
    else:
        print("No conflicts detected - cannot test LLM resolution")

if __name__ == "__main__":
    test_llm_debug()
