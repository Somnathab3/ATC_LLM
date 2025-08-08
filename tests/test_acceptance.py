#!/usr/bin/env python3
"""Test acceptance criteria: pipeline.run(..., ownship_id=...) can fetch states without raising."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cdr.pipeline import CDRPipeline
from src.cdr.schemas import ConfigurationSettings


def test_acceptance_criteria():
    """Test that pipeline.run(..., ownship_id=...) can fetch states without raising."""
    
    # Create configuration
    config = ConfigurationSettings(
        polling_interval_min=5.0,
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
        bluesky_timeout_sec=5.0
    )
    
    # Create pipeline
    pipeline = CDRPipeline(config)
    
    # Mock the BlueSky client to avoid connection issues
    def mock_get_states():
        return [
            {"id": "OWNSHIP", "lat": 59.3, "lon": 18.1, "alt_ft": 35000, "hdg_deg": 90, "spd_kt": 450, "roc_fpm": 0},
            {"id": "TRAFFIC1", "lat": 59.4, "lon": 18.2, "alt_ft": 36000, "hdg_deg": 180, "spd_kt": 400, "roc_fpm": 0}
        ]
    
    pipeline.bs.get_aircraft_states = mock_get_states
    
    try:
        # Test the acceptance criteria: run with ownship_id and max_cycles=1 to test fetching
        pipeline.run(max_cycles=1, ownship_id="OWNSHIP")
        print("✓ Acceptance criteria met: pipeline.run(..., ownship_id=...) fetched states without raising")
        return True
        
    except Exception as e:
        print(f"✗ Acceptance criteria failed: {e}")
        return False


if __name__ == "__main__":
    success = test_acceptance_criteria()
    sys.exit(0 if success else 1)
