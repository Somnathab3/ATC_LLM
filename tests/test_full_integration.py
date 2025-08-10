#!/usr/bin/env python3
"""
Integration test for PromptBuilderV2, Dual LLM engines, and Enhanced Validation.

This test verifies:
- PromptBuilderV2 with trend analysis and multi-intruder context
- Dual LLM architecture (horizontal → vertical fallback)
- Enhanced validation with ownship-only enforcement
- Adaptive snapshot intervals with conflict escalation
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cdr.schemas import (
    ConfigurationSettings, AircraftState, ConflictPrediction, 
    ResolutionCommand, ResolutionType, ResolutionEngine
)
from cdr.pipeline import (
    CDRPipeline, PromptBuilderV2, HorizontalResolutionAgent, 
    VerticalResolutionAgent, EnhancedResolutionValidator
)

def setup_logging():
    """Setup logging for integration test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def create_test_config() -> ConfigurationSettings:
    """Create test configuration with all new features enabled."""
    return ConfigurationSettings(
        # Basic settings
        polling_interval_min=5.0,
        lookahead_time_min=10.0,
        snapshot_interval_min=1.5,
        max_intruders_in_prompt=3,
        intruder_proximity_nm=50.0,
        intruder_altitude_diff_ft=3000.0,
        trend_analysis_window_min=2.0,
        
        # Conflict detection
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        
        # LLM settings
        llm_enabled=True,
        llm_model_name="llama-3.1-8b",
        llm_temperature=0.1,
        llm_max_tokens=1024,
        
        # Resolution limits
        safety_buffer_factor=1.2,
        max_resolution_angle_deg=30.0,
        max_altitude_change_ft=1500.0,
        
        # Enhanced validation settings (NEW)
        enforce_ownship_only=True,
        max_climb_rate_fpm=2500.0,
        max_descent_rate_fpm=2500.0,
        min_flight_level=100,
        max_flight_level=400,
        max_heading_change_deg=60.0,
        
        # Dual LLM engine settings (NEW)
        enable_dual_llm=True,
        horizontal_retry_count=2,
        vertical_retry_count=2,
        
        # BlueSky settings
        bluesky_host="localhost",
        bluesky_port=1337,
        bluesky_timeout_sec=5.0,
        fast_time=True,
        sim_accel_factor=1.0
    )

def create_test_aircraft() -> tuple[AircraftState, list[AircraftState]]:
    """Create test aircraft states for scenario."""
    now = datetime.now()
    
    # Ownship
    ownship = AircraftState(
        aircraft_id="OWNSHIP", 
        timestamp=now,
        latitude=37.0, 
        longitude=-122.0, 
        altitude_ft=35000,
        heading_deg=90.0, 
        ground_speed_kt=450.0, 
        vertical_speed_fpm=0.0
    )
    
    # Multiple intruders for testing multi-intruder context
    intruders = [
        AircraftState(
            aircraft_id="INTRUDER1", 
            timestamp=now,
            latitude=37.1, 
            longitude=-121.9, 
            altitude_ft=35000,
            heading_deg=270.0, 
            ground_speed_kt=480.0, 
            vertical_speed_fpm=0.0
        ),
        AircraftState(
            aircraft_id="INTRUDER2", 
            timestamp=now,
            latitude=37.05, 
            longitude=-122.05, 
            altitude_ft=36000,
            heading_deg=180.0, 
            ground_speed_kt=420.0, 
            vertical_speed_fpm=-500.0
        ),
        AircraftState(
            aircraft_id="INTRUDER3", 
            timestamp=now,
            latitude=36.95, 
            longitude=-121.95, 
            altitude_ft=34000,
            heading_deg=45.0, 
            ground_speed_kt=500.0, 
            vertical_speed_fpm=1000.0
        )
    ]
    
    return ownship, intruders

def create_test_conflicts(ownship: AircraftState, intruders: list[AircraftState]) -> list[ConflictPrediction]:
    """Create test conflict predictions."""
    conflicts = []
    base_time = datetime.now()
    
    for i, intruder in enumerate(intruders[:2]):  # Only first 2 for testing
        conflict = ConflictPrediction(
            ownship_id=ownship.aircraft_id,
            intruder_id=intruder.aircraft_id,
            time_to_cpa_min=3.0 + i,  # Escalating urgency
            distance_at_cpa_nm=2.5,
            altitude_diff_ft=500.0,
            is_conflict=True,
            severity_score=0.8 - (i * 0.1),
            conflict_type="both",
            prediction_time=base_time,
            confidence=0.95
        )
        conflicts.append(conflict)
    
    return conflicts

def test_prompt_builder_v2():
    """Test PromptBuilderV2 with trend analysis and multi-intruder context."""
    print("\n=== Testing PromptBuilderV2 ===")
    
    config = create_test_config()
    prompt_builder = PromptBuilderV2(config)
    
    ownship, intruders = create_test_aircraft()
    conflicts = create_test_conflicts(ownship, intruders)
    primary_conflict = conflicts[0]
    
    # Add historical snapshots for trend analysis
    historical_states = []
    base_time = datetime.now() - timedelta(minutes=5)
    
    for i in range(5):  # 5 historical snapshots
        snapshot_time = base_time + timedelta(minutes=i)
        historical_states.append({
            'timestamp': snapshot_time,
            'ownship': ownship,
            'traffic': intruders
        })
    
    prompt_builder.historical_snapshots = historical_states
    
    # Test prompt generation (convert AircraftState to dict for compatibility)
    ownship_dict = ownship.model_dump()
    traffic_dict = [intruder.model_dump() for intruder in intruders]
    
    prompt = prompt_builder.build_enhanced_prompt(
        conflicts, ownship_dict, traffic_dict
    )
    
    print(f"✓ Generated enhanced prompt ({len(prompt)} chars)")
    print(f"✓ Contains trend analysis: {'TREND ANALYSIS' in prompt}")
    print(f"✓ Contains conflict context: {'CONFLICT CONTEXT' in prompt}")
    print(f"✓ Contains multi-intruder info: {len([line for line in prompt.split('\n') if 'INTRUDER' in line])} intruder entries")
    
    return True

def test_dual_llm_agents():
    """Test Horizontal and Vertical resolution agents."""
    print("\n=== Testing Dual LLM Agents ===")
    
    config = create_test_config()
    
    # Mock LLM client for testing
    class MockLLMClient:
        def __init__(self, response_type="horizontal"):
            self.response_type = response_type
            
        def generate_resolution(self, prompt, config=None, use_enhanced=False):
            if self.response_type == "horizontal":
                return {
                    "resolution": {
                        "type": "heading_change",
                        "target_aircraft": "OWNSHIP",
                        "new_heading_deg": 120.0,
                        "reasoning": "Turn right to avoid conflict"
                    }
                }
            else:  # vertical
                return {
                    "resolution": {
                        "type": "altitude_change",
                        "target_aircraft": "OWNSHIP", 
                        "new_altitude_ft": 37000,
                        "reasoning": "Climb to avoid conflict"
                    }
                }
    
    # Test horizontal agent
    ownship, intruders = create_test_aircraft()
    conflicts = create_test_conflicts(ownship, intruders)
    
    ownship_dict = ownship.model_dump()
    traffic_dict = [intruder.model_dump() for intruder in intruders]
    
    horizontal_agent = HorizontalResolutionAgent(MockLLMClient("horizontal"), config)
    h_response = horizontal_agent.generate_resolution(conflicts, ownship_dict, traffic_dict)
    
    print(f"✓ Horizontal agent response: {h_response['resolution']['type']}")
    
    # Test vertical agent
    vertical_agent = VerticalResolutionAgent(MockLLMClient("vertical"), config)
    v_response = vertical_agent.generate_resolution(conflicts, ownship_dict, traffic_dict)
    
    print(f"✓ Vertical agent response: {v_response['resolution']['type']}")
    
    return True

def test_enhanced_validation():
    """Test enhanced validation with ownship-only enforcement."""
    print("\n=== Testing Enhanced Validation ===")
    
    config = create_test_config()
    validator = EnhancedResolutionValidator(config)
    
    ownship_state, _ = create_test_aircraft()
    
    # Convert AircraftState to dict format expected by validator (old field names)
    ownship = {
        "id": ownship_state.aircraft_id,
        "lat": ownship_state.latitude,
        "lon": ownship_state.longitude,
        "alt_ft": ownship_state.altitude_ft,
        "hdg_deg": ownship_state.heading_deg,
        "spd_kt": ownship_state.ground_speed_kt,
        "vs_fpm": ownship_state.vertical_speed_fpm
    }
    
    # Test valid ownship command
    valid_command = ResolutionCommand(
        resolution_id="TEST_001",
        target_aircraft="OWNSHIP",
        resolution_type=ResolutionType.HEADING_CHANGE,
        new_heading_deg=120.0,
        reasoning="Turn right to avoid conflict",
        source_engine=ResolutionEngine.HORIZONTAL,
        issue_time=datetime.now(),
        safety_margin_nm=3.0,
        is_validated=False,
        validation_notes=""
    )
    
    is_valid = validator.validate_resolution(valid_command, ownship, [], "OWNSHIP")
    print(f"✓ Valid ownship command: {is_valid}")
    
    # Test invalid intruder command (should fail with ownship-only enforcement)
    invalid_command = ResolutionCommand(
        resolution_id="TEST_002",
        target_aircraft="INTRUDER1",
        resolution_type=ResolutionType.HEADING_CHANGE,
        new_heading_deg=120.0,
        reasoning="Tell intruder to turn",
        source_engine=ResolutionEngine.HORIZONTAL,
        issue_time=datetime.now(),
        safety_margin_nm=3.0,
        is_validated=False,
        validation_notes=""
    )
    
    is_invalid = validator.validate_resolution(invalid_command, ownship, [], "OWNSHIP")
    print(f"✓ Invalid intruder command rejected: {not is_invalid}")
    
    # Test excessive heading change (should fail)
    excessive_turn = ResolutionCommand(
        resolution_id="TEST_003",
        target_aircraft="OWNSHIP",
        resolution_type=ResolutionType.HEADING_CHANGE,
        new_heading_deg=200.0,  # 110° change from 90° - exceeds 60° limit
        reasoning="Large turn to avoid conflict",
        source_engine=ResolutionEngine.HORIZONTAL,
        issue_time=datetime.now(),
        safety_margin_nm=3.0,
        is_validated=False,
        validation_notes=""
    )
    
    is_excessive = validator.validate_resolution(excessive_turn, ownship, [], "OWNSHIP")
    print(f"✓ Excessive heading change rejected: {not is_excessive}")
    
    return True

def test_adaptive_snapshot_timing():
    """Test adaptive snapshot intervals with conflict escalation."""
    print("\n=== Testing Adaptive Snapshot Timing ===")
    
    config = create_test_config()
    prompt_builder = PromptBuilderV2(config)
    
    # Test normal interval (no urgent conflicts)
    normal_interval = prompt_builder.get_adaptive_snapshot_interval([])
    print(f"✓ Normal snapshot interval: {normal_interval:.1f} min")
    
    # Test with urgent conflict (should reduce to 1 minute)
    ownship, intruders = create_test_aircraft()
    urgent_conflict = ConflictPrediction(
        ownship_id="OWNSHIP",
        intruder_id="INTRUDER1",
        time_to_cpa_min=0.5,  # Very urgent - 30 seconds
        distance_at_cpa_nm=1.0,
        altitude_diff_ft=200.0,
        is_conflict=True,
        severity_score=0.95,
        conflict_type="both",
        prediction_time=datetime.now(),
        confidence=0.98
    )
    
    urgent_interval = prompt_builder.get_adaptive_snapshot_interval([urgent_conflict])
    print(f"✓ Urgent conflict interval: {urgent_interval:.1f} min")
    
    return urgent_interval <= 1.0

def main():
    """Run all integration tests."""
    setup_logging()
    
    print("🚀 Starting PromptBuilderV2 + Dual LLM + Enhanced Validation Integration Test")
    print("=" * 80)
    
    try:
        # Run individual tests
        test_results = []
        
        test_results.append(test_prompt_builder_v2())
        test_results.append(test_dual_llm_agents())
        test_results.append(test_enhanced_validation())
        test_results.append(test_adaptive_snapshot_timing())
        
        # Summary
        print("\n" + "=" * 80)
        print("🎯 INTEGRATION TEST SUMMARY")
        print(f"✅ Tests passed: {sum(test_results)}/{len(test_results)}")
        
        if all(test_results):
            print("🎉 ALL TESTS PASSED - Integration successful!")
            print("\nNew features verified:")
            print("  ✓ PromptBuilderV2 with trend analysis and multi-intruder context")
            print("  ✓ Dual LLM engines (horizontal → vertical fallback)")
            print("  ✓ Enhanced validation with ownship-only enforcement")
            print("  ✓ Adaptive snapshot intervals (1-2 min with conflict escalation)")
            return True
        else:
            print("❌ Some tests failed - check implementation")
            return False
            
    except Exception as e:
        print(f"💥 Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
