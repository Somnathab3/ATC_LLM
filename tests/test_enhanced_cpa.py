"""Test script to demonstrate enhanced CPA and minimum separation verification.

This script demonstrates the improvements to conflict detection:
- Enhanced CPA calculations with confidence scoring
- Minimum separation verification (5 NM / 1000 ft)
- Adaptive polling intervals based on proximity and time-to-CPA
- Cross-validation capabilities
"""

import logging
from datetime import datetime
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the enhanced modules
from src.cdr.schemas import AircraftState
from src.cdr.enhanced_cpa import (
    calculate_enhanced_cpa, check_minimum_separation, 
    calculate_adaptive_cadence, CPAResult, MinSepCheck
)
from src.cdr.detect import predict_conflicts_enhanced

def create_test_aircraft(aircraft_id: str, lat: float, lon: float, alt_ft: float, 
                        speed_kt: float, heading_deg: float) -> AircraftState:
    """Create a test aircraft state."""
    return AircraftState(
        aircraft_id=aircraft_id,
        timestamp=datetime.now(),
        latitude=lat,
        longitude=lon,
        altitude_ft=alt_ft,
        ground_speed_kt=speed_kt,
        heading_deg=heading_deg,
        vertical_speed_fpm=0.0,
        aircraft_type="B737",
        spawn_offset_min=0.0
    )

def test_scenario_1_head_on_collision():
    """Test scenario: Head-on collision course."""
    logger.info("=== Test Scenario 1: Head-on Collision Course ===")
    
    # Create aircraft on collision course
    ownship = create_test_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)   # East
    intruder = create_test_aircraft("INTRUDER", 40.0, -73.5, 35000, 480, 270) # West
    
    # Enhanced CPA calculation
    cpa_result = calculate_enhanced_cpa(ownship, intruder)
    logger.info(f"Enhanced CPA Result:")
    logger.info(f"  Distance at CPA: {cpa_result.distance_at_cpa_nm:.2f} NM")
    logger.info(f"  Time to CPA: {cpa_result.time_to_cpa_min:.2f} minutes")
    logger.info(f"  Relative speed: {cpa_result.relative_speed_kt:.1f} kt")
    logger.info(f"  Convergence rate: {cpa_result.convergence_rate_nm_min:.2f} NM/min")
    logger.info(f"  Is converging: {cpa_result.is_converging}")
    logger.info(f"  Confidence: {cpa_result.confidence:.2f}")
    
    # Minimum separation check
    min_sep = check_minimum_separation(ownship, intruder)
    logger.info(f"Minimum Separation Check:")
    logger.info(f"  Horizontal separation: {min_sep.horizontal_sep_nm:.2f} NM")
    logger.info(f"  Vertical separation: {min_sep.vertical_sep_ft:.0f} ft")
    logger.info(f"  Horizontal violation: {min_sep.horizontal_violation}")
    logger.info(f"  Vertical violation: {min_sep.vertical_violation}")
    logger.info(f"  Is conflict: {min_sep.is_conflict}")
    
    # Adaptive cadence
    traffic = [intruder]
    adaptive_interval = calculate_adaptive_cadence(ownship, traffic, [])
    logger.info(f"Adaptive polling interval: {adaptive_interval:.1f} minutes")
    
    return cpa_result, min_sep, adaptive_interval

def test_scenario_2_parallel_traffic():
    """Test scenario: Parallel traffic at safe distance."""
    logger.info("\n=== Test Scenario 2: Parallel Traffic ===")
    
    # Create parallel aircraft at safe distance
    ownship = create_test_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)    # East
    intruder = create_test_aircraft("INTRUDER", 40.1, -74.0, 37000, 480, 90)  # East, parallel
    
    # Enhanced CPA calculation
    cpa_result = calculate_enhanced_cpa(ownship, intruder)
    logger.info(f"Enhanced CPA Result:")
    logger.info(f"  Distance at CPA: {cpa_result.distance_at_cpa_nm:.2f} NM")
    logger.info(f"  Time to CPA: {cpa_result.time_to_cpa_min:.2f} minutes")
    logger.info(f"  Is converging: {cpa_result.is_converging}")
    
    # Minimum separation check
    min_sep = check_minimum_separation(ownship, intruder)
    logger.info(f"Minimum Separation Check:")
    logger.info(f"  Horizontal separation: {min_sep.horizontal_sep_nm:.2f} NM")
    logger.info(f"  Vertical separation: {min_sep.vertical_sep_ft:.0f} ft")
    logger.info(f"  Is conflict: {min_sep.is_conflict}")
    
    # Adaptive cadence
    traffic = [intruder]
    adaptive_interval = calculate_adaptive_cadence(ownship, traffic, [])
    logger.info(f"Adaptive polling interval: {adaptive_interval:.1f} minutes")
    
    return cpa_result, min_sep, adaptive_interval

def test_scenario_3_imminent_conflict():
    """Test scenario: Imminent conflict (< 2 minutes)."""
    logger.info("\n=== Test Scenario 3: Imminent Conflict ===")
    
    # Create aircraft very close to collision
    ownship = create_test_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)      # East
    intruder = create_test_aircraft("INTRUDER", 40.0, -73.98, 35000, 480, 270)  # West, very close
    
    # Enhanced CPA calculation
    cpa_result = calculate_enhanced_cpa(ownship, intruder)
    logger.info(f"Enhanced CPA Result:")
    logger.info(f"  Distance at CPA: {cpa_result.distance_at_cpa_nm:.2f} NM")
    logger.info(f"  Time to CPA: {cpa_result.time_to_cpa_min:.2f} minutes")
    logger.info(f"  Convergence rate: {cpa_result.convergence_rate_nm_min:.2f} NM/min")
    
    # Minimum separation check
    min_sep = check_minimum_separation(ownship, intruder)
    logger.info(f"Minimum Separation Check:")
    logger.info(f"  Horizontal separation: {min_sep.horizontal_sep_nm:.2f} NM")
    logger.info(f"  Is conflict: {min_sep.is_conflict}")
    
    # Adaptive cadence
    traffic = [intruder]
    adaptive_interval = calculate_adaptive_cadence(ownship, traffic, [])
    logger.info(f"Adaptive polling interval: {adaptive_interval:.1f} minutes")
    
    return cpa_result, min_sep, adaptive_interval

def test_scenario_4_sparse_traffic():
    """Test scenario: Sparse traffic environment."""
    logger.info("\n=== Test Scenario 4: Sparse Traffic ===")
    
    # Create aircraft with distant traffic
    ownship = create_test_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)      # East
    distant_traffic = create_test_aircraft("DISTANT", 41.0, -72.0, 37000, 480, 180)  # Distant, south
    
    # Enhanced CPA calculation
    cpa_result = calculate_enhanced_cpa(ownship, distant_traffic)
    logger.info(f"Enhanced CPA Result:")
    logger.info(f"  Distance at CPA: {cpa_result.distance_at_cpa_nm:.2f} NM")
    logger.info(f"  Time to CPA: {cpa_result.time_to_cpa_min:.2f} minutes")
    logger.info(f"  Is converging: {cpa_result.is_converging}")
    
    # Minimum separation check
    min_sep = check_minimum_separation(ownship, distant_traffic)
    logger.info(f"Minimum Separation Check:")
    logger.info(f"  Horizontal separation: {min_sep.horizontal_sep_nm:.2f} NM")
    logger.info(f"  Is conflict: {min_sep.is_conflict}")
    
    # Adaptive cadence with no close traffic
    traffic = [distant_traffic]
    adaptive_interval = calculate_adaptive_cadence(ownship, traffic, [])
    logger.info(f"Adaptive polling interval: {adaptive_interval:.1f} minutes")
    
    return cpa_result, min_sep, adaptive_interval

def test_enhanced_conflict_detection():
    """Test the enhanced conflict detection with adaptive cadence."""
    logger.info("\n=== Test Enhanced Conflict Detection ===")
    
    # Create a complex scenario with multiple aircraft
    ownship = create_test_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)
    
    traffic = [
        create_test_aircraft("THREAT1", 40.0, -73.9, 35000, 480, 270),   # Imminent threat
        create_test_aircraft("THREAT2", 40.05, -73.8, 35500, 460, 270),  # Moderate threat
        create_test_aircraft("SAFE1", 40.2, -74.0, 37000, 480, 90),      # Safe parallel
        create_test_aircraft("DISTANT", 41.0, -72.0, 37000, 480, 180),   # Distant
    ]
    
    # Run enhanced conflict detection
    conflicts, recommended_interval = predict_conflicts_enhanced(
        ownship, traffic, lookahead_minutes=10.0, use_adaptive_cadence=True
    )
    
    logger.info(f"Enhanced Conflict Detection Results:")
    logger.info(f"  Total conflicts detected: {len(conflicts)}")
    logger.info(f"  Recommended polling interval: {recommended_interval:.1f} minutes")
    
    # Display each conflict
    for i, conflict in enumerate(conflicts):
        logger.info(f"  Conflict {i+1}:")
        logger.info(f"    Intruder: {conflict.intruder_id}")
        logger.info(f"    Time to CPA: {conflict.time_to_cpa_min:.2f} minutes")
        logger.info(f"    Distance at CPA: {conflict.distance_at_cpa_nm:.2f} NM")
        logger.info(f"    Severity: {conflict.severity_score:.2f}")
        logger.info(f"    Type: {conflict.conflict_type}")
        logger.info(f"    Confidence: {conflict.confidence:.2f}")
    
    return conflicts, recommended_interval

def demonstrate_adaptive_behavior():
    """Demonstrate how adaptive cadence changes based on scenarios."""
    logger.info("\n=== Adaptive Cadence Demonstration ===")
    
    scenarios = [
        ("No traffic", [], 5.0),
        ("Distant traffic", [create_test_aircraft("DISTANT", 41.0, -72.0, 37000, 480, 180)], 5.0),
        ("Moderate proximity", [create_test_aircraft("MODERATE", 40.3, -73.8, 35000, 480, 270)], 2.0),
        ("Close proximity", [create_test_aircraft("CLOSE", 40.1, -73.9, 35000, 480, 270)], 1.0),
        ("Imminent threat", [create_test_aircraft("IMMINENT", 40.0, -73.98, 35000, 480, 270)], 0.5),
    ]
    
    ownship = create_test_aircraft("OWNSHIP", 40.0, -74.0, 35000, 480, 90)
    
    for scenario_name, traffic, expected_interval in scenarios:
        adaptive_interval = calculate_adaptive_cadence(ownship, traffic, [])
        logger.info(f"  {scenario_name:15}: {adaptive_interval:4.1f} minutes (expected: {expected_interval:4.1f})")

def main():
    """Run all test scenarios."""
    logger.info("Enhanced CPA and Minimum Separation Verification Test")
    logger.info("=" * 60)
    
    try:
        # Run test scenarios
        test_scenario_1_head_on_collision()
        test_scenario_2_parallel_traffic()
        test_scenario_3_imminent_conflict()
        test_scenario_4_sparse_traffic()
        
        # Test enhanced conflict detection
        test_enhanced_conflict_detection()
        
        # Demonstrate adaptive behavior
        demonstrate_adaptive_behavior()
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
