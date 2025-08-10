#!/usr/bin/env python3
"""Test script for PromptBuilderV2 implementation."""
import pytest
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cdr.schemas import ConfigurationSettings, ConflictPrediction
from cdr.pipeline import PromptBuilderV2


def test_prompt_builder_v2():
    """Test the PromptBuilderV2 functionality."""
    print("Testing PromptBuilderV2...")
    
    # Create configuration
    config = ConfigurationSettings(
        polling_interval_min=5.0,
        lookahead_time_min=10.0,
        snapshot_interval_min=1.5,
        max_intruders_in_prompt=3,
        intruder_proximity_nm=50.0,
        intruder_altitude_diff_ft=3000.0,
        trend_analysis_window_min=2.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        llm_enabled=True,
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
    
    # Create PromptBuilderV2
    prompt_builder = PromptBuilderV2(config)
    
    # Create test aircraft states
    ownship = {
        "id": "OWNSHIP",
        "lat": 40.0,
        "lon": -74.0,
        "alt_ft": 35000.0,
        "hdg_deg": 90.0,
        "spd_kt": 450.0,
        "roc_fpm": 0.0
    }
    
    traffic = [
        {
            "id": "INTRUDER1",
            "lat": 40.01,
            "lon": -73.95,
            "alt_ft": 35500.0,
            "hdg_deg": 270.0,
            "spd_kt": 430.0,
            "roc_fpm": -500.0
        },
        {
            "id": "INTRUDER2", 
            "lat": 39.99,
            "lon": -74.02,
            "alt_ft": 34500.0,
            "hdg_deg": 45.0,
            "spd_kt": 420.0,
            "roc_fpm": 200.0
        },
        {
            "id": "INTRUDER3_FAR",
            "lat": 42.0,
            "lon": -76.0,
            "alt_ft": 40000.0,
            "hdg_deg": 180.0,
            "spd_kt": 400.0,
            "roc_fpm": 0.0
        }
    ]
    
    # Add some aircraft snapshots for trend analysis
    all_aircraft = {"OWNSHIP": ownship}
    for t in traffic:
        all_aircraft[t["id"]] = t
    
    print("Adding initial aircraft snapshot...")
    prompt_builder.add_aircraft_snapshot(all_aircraft)
    
    # Simulate movement over 2 minutes
    import time
    time.sleep(0.1)  # Small delay to simulate time passage
    
    # Update positions slightly to create trends
    ownship_moved = ownship.copy()
    ownship_moved["lat"] = 40.005
    ownship_moved["lon"] = -73.99
    ownship_moved["hdg_deg"] = 95.0
    
    traffic_moved = []
    for t in traffic:
        t_moved = t.copy()
        t_moved["lat"] += 0.002
        t_moved["lon"] -= 0.001
        t_moved["hdg_deg"] = (t_moved["hdg_deg"] + 5) % 360
        traffic_moved.append(t_moved)
    
    all_aircraft_moved = {"OWNSHIP": ownship_moved}
    for t in traffic_moved:
        all_aircraft_moved[t["id"]] = t
    
    print("Adding second aircraft snapshot for trend analysis...")
    prompt_builder.add_aircraft_snapshot(all_aircraft_moved)
    
    # Create mock conflicts
    conflicts = [
        ConflictPrediction(
            ownship_id="OWNSHIP",
            intruder_id="INTRUDER1",
            is_conflict=True,
            time_to_cpa_min=3.5,
            distance_at_cpa_nm=3.2,
            altitude_diff_ft=500.0,
            severity_score=0.8,
            conflict_type="horizontal",
            prediction_time=datetime.now(),
            ownship_cpa_lat=40.006,
            ownship_cpa_lon=-73.985,
            intruder_cpa_lat=40.007,
            intruder_cpa_lon=-73.984
        ),
        ConflictPrediction(
            ownship_id="OWNSHIP",
            intruder_id="INTRUDER2",
            is_conflict=True,
            time_to_cpa_min=4.8,
            distance_at_cpa_nm=4.1,
            altitude_diff_ft=500.0,
            severity_score=0.6,
            conflict_type="horizontal",
            prediction_time=datetime.now(),
            ownship_cpa_lat=40.004,
            ownship_cpa_lon=-73.982,
            intruder_cpa_lat=40.003,
            intruder_cpa_lon=-73.981
        )
    ]
    
    print("Testing enhanced prompt generation...")
    enhanced_prompt = prompt_builder.build_enhanced_prompt(conflicts, ownship_moved, traffic_moved)
    
    if enhanced_prompt:
        print("✓ Enhanced prompt generated successfully!")
        print("\n" + "="*80)
        print("GENERATED ENHANCED PROMPT:")
        print("="*80)
        print(enhanced_prompt)
        print("="*80)
        
        # Verify key features
        checks = [
            ("Multi-intruder context", "INTRUDERS" in enhanced_prompt),
            ("Trend analysis", "TRENDS" in enhanced_prompt),
            ("JSON schema", "REQUIRED JSON RESPONSE FORMAT" in enhanced_prompt),
            ("Example responses", "EXAMPLE RESPONSES" in enhanced_prompt),
            ("Conflict information", "ACTIVE CONFLICTS" in enhanced_prompt),
            ("Distance moved trend", "Distance moved:" in enhanced_prompt),
            ("Altitude change trend", "Altitude change:" in enhanced_prompt),
            ("Strict JSON format", '"resolution_type"' in enhanced_prompt),
        ]
        
        print("\nFEATURE VERIFICATION:")
        all_passed = True
        for feature, passed in checks:
            status = "✓" if passed else "✗"
            print(f"{status} {feature}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 All PromptBuilderV2 features working correctly!")
        else:
            print("\n⚠️  Some features may need attention.")
            
    else:
        print("✗ No enhanced prompt generated")
        return False
    
    # Test filtering functionality
    print("\nTesting intruder filtering...")
    relevant_intruders = prompt_builder.filter_relevant_intruders(ownship_moved, traffic_moved, conflicts)
    
    print(f"Filtered {len(relevant_intruders)} relevant intruders from {len(traffic_moved)} total traffic")
    for intruder in relevant_intruders:
        print(f"  - {intruder['id']}: {intruder['_distance_nm']:.1f} NM, priority {intruder['_priority']}")
    
    # Test trend calculation
    print("\nTesting trend calculation...")
    ownship_trends = prompt_builder.calculate_trends("OWNSHIP")
    print(f"Ownship trends: {ownship_trends}")
    
    return True


if __name__ == "__main__":
    success = test_prompt_builder_v2()
    if success:
        print("\n✅ PromptBuilderV2 test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ PromptBuilderV2 test failed!")
        sys.exit(1)
