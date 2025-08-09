"""Show example of enhanced LLM prompts and BlueSky command parsing."""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.cdr.schemas import ConfigurationSettings, AircraftState, ConflictPrediction
from src.cdr.enhanced_llm_client import EnhancedLLMClient

def show_enhanced_prompts():
    """Show examples of the enhanced prompts and parsing."""
    
    print("=" * 80)
    print("🤖 ENHANCED LLM PROMPTS DEMONSTRATION")
    print("=" * 80)
    
    # Configuration
    config = ConfigurationSettings(
        llm_model_name="llama3.1:8b",
        llm_temperature=0.1,
        polling_interval_min=1.0,
        lookahead_time_min=10.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        llm_max_tokens=2048,
        safety_buffer_factor=1.1,
        max_resolution_angle_deg=30.0,
        max_altitude_change_ft=2000.0,
        bluesky_host="127.0.0.1",
        bluesky_port=5555,
        bluesky_timeout_sec=5.0,
        fast_time=True,
        sim_accel_factor=1.0
    )
    
    llm_client = EnhancedLLMClient(config)
    
    # Create realistic conflict scenario
    ownship = AircraftState(
        aircraft_id="UAL123",
        latitude=52.3736,  # London area
        longitude=4.8896,  # Amsterdam area  
        altitude_ft=35000,
        heading_deg=90,    # Eastbound
        ground_speed_kt=450,
        vertical_speed_fpm=0,
        aircraft_type="B777",
        timestamp=datetime.fromisoformat("2025-08-08T12:00:00+00:00")
    )
    
    traffic = [
        AircraftState(
            aircraft_id="KLM456",
            latitude=52.3736,  # Same latitude - conflict!
            longitude=5.0896,  # 0.2 degrees east
            altitude_ft=35000,  # Same altitude
            heading_deg=270,   # Westbound - head-on
            ground_speed_kt=420,
            vertical_speed_fpm=0,
            aircraft_type="A330",
            timestamp=datetime.fromisoformat("2025-08-08T12:00:00+00:00")
        )
    ]
    
    # 1. ENHANCED DETECTION PROMPT
    print("\n📡 ENHANCED CONFLICT DETECTION PROMPT:")
    print("-" * 50)
    
    detection_prompt = llm_client.build_enhanced_detect_prompt(ownship, traffic, config)
    
    # Show key sections of the prompt
    lines = detection_prompt.split('\n')
    for i, line in enumerate(lines):
        if i < 10 or 'AIRCRAFT STATES:' in line or 'OUTPUT FORMAT' in line:
            print(f"  {line}")
        elif i == 10:
            print("  ... [aviation standards and requirements] ...")
    
    print(f"\n✅ Total prompt length: {len(detection_prompt)} characters")
    print("🎯 Key improvements:")
    print("  • Expert ATC identity and ICAO certification context")
    print("  • Precise separation standards (5NM horizontal, 1000ft vertical)")
    print("  • Clear CPA calculation requirements")
    print("  • Structured JSON output with conflict details")
    print("  • Professional aviation terminology")
    
    # 2. ENHANCED RESOLUTION PROMPT
    print("\n🛠️  ENHANCED CONFLICT RESOLUTION PROMPT:")
    print("-" * 50)
    
    # Mock conflict for resolution
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
    
    # Show key sections
    lines = resolution_prompt.split('\n')
    for i, line in enumerate(lines):
        if i < 8 or 'CONFLICTS DETECTED:' in line or 'BLUESKY COMMAND FORMAT:' in line or 'OUTPUT FORMAT' in line:
            print(f"  {line}")
        elif i == 8:
            print("  ... [detailed conflict information] ...")
    
    print(f"\n✅ Total prompt length: {len(resolution_prompt)} characters")
    print("🎯 Key improvements:")
    print("  • Aviation resolution constraints (max 30° turn, altitude limits)")
    print("  • Direct BlueSky command format specification")
    print("  • Multiple resolution types (HDG/ALT/SPD/DCT)")
    print("  • Rationale and confidence requirements")
    print("  • Backup action planning")
    
    # 3. COMMAND PARSING EXAMPLES
    print("\n🔧 BLUESKY COMMAND PARSING & SANITIZATION:")
    print("-" * 50)
    
    # Test various command formats that LLM might return
    test_responses = [
        '{"action": "HEADING_CHANGE", "bluesky_command": "UAL123 HDG 120"}',
        '{"action": "ALTITUDE_CHANGE", "bluesky_command": "UAL123 ALTITUDE 37000"}',
        '{"action": "SPEED_CHANGE", "bluesky_command": "UAL123 SPEED 480"}',
        'UAL123 HEADING 095',  # Raw command
        'UAL123 HDG 450',      # Invalid heading
        'UAL123 ALT 99999',    # Invalid altitude
    ]
    
    print("Example command sanitization:")
    for response in test_responses:
        if response.startswith('{'):
            # Parse JSON response
            try:
                import json
                data = json.loads(response)
                command = data.get('bluesky_command', '')
                if command:
                    sanitized = llm_client._sanitize_bluesky_command(command)
                    print(f"  JSON: {data['action']} → '{sanitized}'")
            except:
                print(f"  JSON: Failed to parse")
        else:
            # Direct command
            sanitized = llm_client._sanitize_bluesky_command(response)
            print(f"  Direct: '{response}' → '{sanitized}'")
    
    # 4. COMPARISON WITH STANDARD PROMPTS
    print("\n📊 COMPARISON WITH STANDARD PROMPTS:")
    print("-" * 50)
    
    print("BEFORE (Basic Prompt):")
    print("  'Detect conflicts between aircraft and return JSON'")
    print("  ❌ Vague requirements")
    print("  ❌ No aviation standards")
    print("  ❌ Unclear output format")
    print("  ❌ No command format specification")
    
    print("\nAFTER (Enhanced Prompt):")
    print("  ✅ Expert ATC context with ICAO certification")
    print("  ✅ Precise ICAO separation standards (5NM/1000ft)")
    print("  ✅ Structured aircraft state formatting")
    print("  ✅ Detailed conflict analysis requirements")
    print("  ✅ Exact JSON schema specification")
    print("  ✅ Direct BlueSky command format")
    print("  ✅ Aviation terminology and constraints")
    print("  ✅ Confidence scoring and rationale")
    
    print("\n" + "=" * 80)
    print("✅ ENHANCED PROMPTS PROVIDE:")
    print("  🎯 Industry-standard aviation context")
    print("  📐 Precise technical specifications")
    print("  🛠️  Direct system integration")
    print("  🔍 Robust error handling")
    print("  📊 Performance validation")
    print("=" * 80)

if __name__ == "__main__":
    show_enhanced_prompts()
