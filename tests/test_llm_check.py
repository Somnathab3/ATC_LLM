#!/usr/bin/env python3
"""
Quick test to check if LLM is being called or using mock data
"""

import sys
from pathlib import Path
import logging

# Add the parent directory to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.cdr.schemas import ConfigurationSettings
from src.cdr.llm_client import LlamaClient

# Configure logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_llm_vs_mock():
    """Test if LLM is actually being called or using mock responses"""
    
    # Create configuration
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
        max_resolution_angle_deg=45.0,
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
    
    # Check if mock mode is enabled
    print(f"LLM Client Mock Mode: {llm_client.use_mock}")
    print(f"LLM Host: {llm_client.host}")
    print(f"LLM Model: {llm_client.model_name}")
    print(f"LLM Timeout: {llm_client.timeout}")
    
    # Create a simple test conflict scenario
    detect_out = {
        "ownship": {
            "aircraft_id": "TEST001",
            "lat": 54.3,
            "lon": 14.7,
            "alt_ft": 35000,
            "hdg_deg": 90,
            "spd_kt": 420
        },
        "conflicts": [{
            "intruder_id": "TEST002",
            "time_to_cpa_min": 5.0,
            "distance_at_cpa_nm": 3.0,
            "altitude_diff_ft": 500,
            "severity": 0.8,
            "conflict_type": "both"
        }]
    }
    
    cfg = {
        "max_resolution_angle_deg": 30,
        "max_altitude_change_ft": 2000,
        "min_horizontal_separation_nm": 5.0,
        "min_vertical_separation_ft": 1000.0
    }
    
    print("\n--- Testing LLM Resolution Generation ---")
    try:
        # Call the resolution generation method
        result = llm_client.generate_resolution(detect_out, cfg, use_enhanced=True)
        
        print(f"Resolution Result Type: {type(result)}")
        if hasattr(result, 'action'):
            print(f"Action: {result.action}")
            print(f"Params: {getattr(result, 'params', 'None')}")
            print(f"Rationale: {getattr(result, 'rationale', 'None')}")
        elif isinstance(result, dict):
            print(f"Result Dict: {result}")
        else:
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"Error during resolution generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_vs_mock()
