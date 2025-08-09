#!/usr/bin/env python3
"""Quick test to verify _fetch_aircraft_states implementation."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cdr.pipeline import CDRPipeline
from src.cdr.schemas import ConfigurationSettings


def test_fetch_aircraft_states():
    """Test that _fetch_aircraft_states method works as specified."""
    
    # Create configuration with default values
    config = ConfigurationSettings(
        polling_interval_min=5.0,
        lookahead_time_min=10.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        llm_model_name="llama3.1:8b",
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
    
    # Verify that _fetch_aircraft_states exists and is callable
    assert hasattr(pipeline, '_fetch_aircraft_states'), "_fetch_aircraft_states method missing"
    assert callable(getattr(pipeline, '_fetch_aircraft_states')), "_fetch_aircraft_states is not callable"
    
    # Verify bs alias exists
    assert hasattr(pipeline, 'bs'), "bs alias missing"
    assert pipeline.bs is pipeline.bluesky_client, "bs alias not pointing to bluesky_client"
    
    # Verify log alias exists
    assert hasattr(pipeline, 'log'), "log alias missing"
    
    print("[OK] All structural tests passed")
    
    # Test method signature - should accept ownship_id parameter
    try:
        # Mock the get_aircraft_states method to avoid BlueSky connection
        original_get_states = pipeline.bs.get_aircraft_states
        
        def mock_get_states():
            return [
                {"id": "OWNSHIP", "lat": 59.3, "lon": 18.1, "alt_ft": 35000, "hdg_deg": 90, "spd_kt": 450, "roc_fpm": 0},
                {"id": "TRAFFIC1", "lat": 59.4, "lon": 18.2, "alt_ft": 36000, "hdg_deg": 180, "spd_kt": 400, "roc_fpm": 0},
                {"id": "TRAFFIC2", "lat": 59.5, "lon": 18.3, "alt_ft": 34000, "hdg_deg": 270, "spd_kt": 480, "roc_fpm": 0}
            ]
        
        pipeline.bs.get_aircraft_states = mock_get_states
        
        # Test the method
        ownship, traffic = pipeline._fetch_aircraft_states("OWNSHIP")
        
        # Verify return format
        assert ownship is not None, "Ownship should be found"
        assert ownship["id"] == "OWNSHIP", "Ownship ID should match"
        assert len(traffic) == 2, "Should have 2 traffic aircraft"
        assert all(t["id"] != "OWNSHIP" for t in traffic), "Traffic should not include ownship"
        
        print("[OK] Method functionality test passed")
        
        # Test with missing ownship
        ownship_missing, traffic_all = pipeline._fetch_aircraft_states("MISSING")
        assert ownship_missing is None, "Missing ownship should return None"
        assert len(traffic_all) == 3, "All aircraft should be in traffic when ownship not found"
        
        print("[OK] Missing ownship test passed")
        
        # Restore original method
        pipeline.bs.get_aircraft_states = original_get_states
        
        print("[OK] All tests passed! _fetch_aircraft_states implementation is correct.")
        return True
        
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_fetch_aircraft_states()
    sys.exit(0 if success else 1)
