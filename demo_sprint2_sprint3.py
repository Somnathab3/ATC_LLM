#!/usr/bin/env python3
"""
Demo script for Sprint 2 & 3 implementations.

This script demonstrates:
- Sprint 2: Deterministic conflict detection with CPA and Wolfgang metrics
- Sprint 3: LLM-based detection/resolution with safety validation

Usage:
    python demo_sprint2_sprint3.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from src.cdr.detect import predict_conflicts, is_conflict
from src.cdr.llm_client import LlamaClient
from src.cdr.resolve import execute_resolution
from src.cdr.metrics import MetricsCollector
from src.cdr.schemas import (
    AircraftState, ConfigurationSettings, ResolveOut, ConflictPrediction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_scenario():
    """Create test scenario with ownship and conflicting traffic."""
    
    # Ownship flying east at FL350
    ownship = AircraftState(
        aircraft_id="OWNSHIP",
        timestamp=datetime.now(),
        latitude=59.3,        # Stockholm area
        longitude=18.1,
        altitude_ft=35000,
        ground_speed_kt=450,
        heading_deg=90,       # East
        vertical_speed_fpm=0
    )
    
    # Traffic aircraft on potential collision course
    traffic = [
        # Converging from northeast
        AircraftState(
            aircraft_id="TRF001",
            timestamp=datetime.now(),
            latitude=59.4,
            longitude=18.7,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=270,  # West (head-on)
            vertical_speed_fpm=0
        ),
        
        # Crossing from south
        AircraftState(
            aircraft_id="TRF002", 
            timestamp=datetime.now(),
            latitude=59.0,
            longitude=18.3,
            altitude_ft=35000,
            ground_speed_kt=480,
            heading_deg=0,    # North
            vertical_speed_fpm=0
        ),
        
        # High altitude, no conflict
        AircraftState(
            aircraft_id="TRF003",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.2,
            altitude_ft=41000,  # FL410, safe altitude
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
    ]
    
    return ownship, traffic


def demo_sprint2_detection():
    """Demonstrate Sprint 2 conflict detection."""
    logger.info("=== SPRINT 2 DEMO: Deterministic Conflict Detection ===")
    
    # Create test scenario
    ownship, traffic = create_test_scenario()
    
    logger.info(f"Ownship: {ownship.aircraft_id} at {ownship.latitude:.3f}, {ownship.longitude:.3f}, FL{ownship.altitude_ft//100}")
    logger.info(f"Traffic count: {len(traffic)}")
    
    # Run conflict detection
    conflicts = predict_conflicts(ownship, traffic, lookahead_minutes=10.0)
    
    logger.info(f"Conflicts detected: {len(conflicts)}")
    
    for i, conflict in enumerate(conflicts):
        logger.info(f"  Conflict {i+1}: {conflict.intruder_id}")
        logger.info(f"    Time to CPA: {conflict.time_to_cpa_min:.2f} min")
        logger.info(f"    Distance at CPA: {conflict.distance_at_cpa_nm:.2f} NM")
        logger.info(f"    Altitude diff: {conflict.altitude_diff_ft:.0f} ft")
        logger.info(f"    Is conflict: {conflict.is_conflict}")
        logger.info(f"    Severity: {conflict.severity_score:.3f}")
        logger.info(f"    Type: {conflict.conflict_type}")
    
    return conflicts


def demo_sprint2_metrics(conflicts):
    """Demonstrate Sprint 2 Wolfgang metrics."""
    logger.info("\n=== SPRINT 2 DEMO: Wolfgang KPI Metrics ===")
    
    # Initialize metrics collector
    collector = MetricsCollector()
    
    # Record some sample data
    collector.record_cycle_time(2.3)
    collector.record_cycle_time(1.8)
    collector.record_cycle_time(2.1)
    
    # Record conflicts
    collector.record_conflict_detection(conflicts, datetime.now())
    
    # Calculate Wolfgang KPIs
    kpis = collector.calculate_wolfgang_kpis()
    
    logger.info("Wolfgang (2011) KPIs:")
    for kpi_name, value in kpis.items():
        logger.info(f"  {kpi_name.upper()}: {value:.3f}")
    
    # Generate full summary
    summary = collector.generate_summary()
    
    logger.info(f"\nSummary:")
    logger.info(f"  Total cycles: {summary.total_cycles}")
    logger.info(f"  Avg cycle time: {summary.avg_cycle_time_sec:.2f} sec")
    logger.info(f"  Conflicts detected: {summary.total_conflicts_detected}")
    logger.info(f"  Detection accuracy: {summary.detection_accuracy:.3f}")
    
    return summary


def demo_sprint3_llm():
    """Demonstrate Sprint 3 LLM integration."""
    logger.info("\n=== SPRINT 3 DEMO: LLM Detection & Resolution ===")
    
    # Create configuration
    config = ConfigurationSettings(
        llm_model_name="llama-3.1-8b",
        llm_temperature=0.1,
        llm_max_tokens=2048
    )
    
    # Initialize LLM client
    client = LlamaClient(config)
    logger.info(f"Initialized LLM client: {config.llm_model_name}")
    
    # Test LLM detection
    ownship, traffic = create_test_scenario()
    
    state_data = {
        "ownship": ownship.dict(),
        "traffic": [t.dict() for t in traffic[:1]],  # Just first intruder
        "current_time": datetime.now().isoformat()
    }
    state_json = json.dumps(state_data, indent=2)
    
    logger.info("Calling LLM for conflict detection...")
    detect_result = client.ask_detect(state_json)
    
    if detect_result:
        logger.info(f"LLM Detection Result:")
        logger.info(f"  Conflict detected: {detect_result.conflict}")
        logger.info(f"  Intruders: {len(detect_result.intruders)}")
        for intruder in detect_result.intruders:
            logger.info(f"    {intruder}")
    
    # Test LLM resolution
    if detect_result and detect_result.conflict:
        logger.info("\nCalling LLM for conflict resolution...")
        
        conflict_info = {
            "intruder_id": "TRF001",
            "time_to_cpa_min": 4.5,
            "distance_at_cpa_nm": 3.2
        }
        
        resolve_result = client.ask_resolve(state_json, conflict_info)
        
        if resolve_result:
            logger.info(f"LLM Resolution Result:")
            logger.info(f"  Action: {resolve_result.action}")
            logger.info(f"  Parameters: {resolve_result.params}")
            logger.info(f"  Rationale: {resolve_result.rationale}")
            
            return resolve_result
    
    return None


def demo_sprint3_safety_validation(ownship, traffic, llm_resolution):
    """Demonstrate Sprint 3 safety validation."""
    logger.info("\n=== SPRINT 3 DEMO: Safety Validation ===")
    
    if not llm_resolution:
        logger.warning("No LLM resolution to validate")
        return
    
    # Create a conflict prediction for validation
    conflict = ConflictPrediction(
        ownship_id=ownship.aircraft_id,
        intruder_id="TRF001",
        time_to_cpa_min=4.5,
        distance_at_cpa_nm=3.2,
        altitude_diff_ft=500.0,
        is_conflict=True,
        severity_score=0.8,
        conflict_type="both",
        prediction_time=datetime.now()
    )
    
    # Execute resolution with safety validation
    logger.info("Executing resolution with safety validation...")
    
    intruder = traffic[0]  # First traffic aircraft
    validated_cmd = execute_resolution(llm_resolution, ownship, intruder, conflict)
    
    if validated_cmd:
        logger.info(f"Resolution validated successfully:")
        logger.info(f"  Command ID: {validated_cmd.resolution_id}")
        logger.info(f"  Type: {validated_cmd.resolution_type}")
        logger.info(f"  Target: {validated_cmd.target_aircraft}")
        logger.info(f"  Safety margin: {validated_cmd.safety_margin_nm:.2f} NM")
        logger.info(f"  Validated: {validated_cmd.is_validated}")
        
        if validated_cmd.new_heading_deg:
            logger.info(f"  New heading: {validated_cmd.new_heading_deg:.0f}°")
        if validated_cmd.new_altitude_ft:
            logger.info(f"  New altitude: {validated_cmd.new_altitude_ft:.0f} ft")
    else:
        logger.warning("Resolution failed safety validation - no safe resolution found")
    
    return validated_cmd


def save_demo_results(conflicts, metrics_summary, resolution_cmd):
    """Save demo results for sprint reports."""
    logger.info("\n=== Saving Demo Results ===")
    
    # Create reports directory
    reports_dir = Path("reports/sprint_02")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save conflict detection results
    conflicts_data = [
        {
            "ownship_id": c.ownship_id,
            "intruder_id": c.intruder_id,
            "time_to_cpa_min": c.time_to_cpa_min,
            "distance_at_cpa_nm": c.distance_at_cpa_nm,
            "altitude_diff_ft": c.altitude_diff_ft,
            "is_conflict": c.is_conflict,
            "severity_score": c.severity_score,
            "conflict_type": c.conflict_type
        }
        for c in conflicts
    ]
    
    with open(reports_dir / "conflicts_detected.json", "w") as f:
        json.dump(conflicts_data, f, indent=2)
    
    # Save metrics summary
    with open(reports_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary.__dict__, f, indent=2, default=str)
    
    # Save resolution command if available
    if resolution_cmd:
        resolution_data = {
            "resolution_id": resolution_cmd.resolution_id,
            "target_aircraft": resolution_cmd.target_aircraft,
            "resolution_type": resolution_cmd.resolution_type.value,
            "new_heading_deg": resolution_cmd.new_heading_deg,
            "new_altitude_ft": resolution_cmd.new_altitude_ft,
            "safety_margin_nm": resolution_cmd.safety_margin_nm,
            "is_validated": resolution_cmd.is_validated,
            "issue_time": resolution_cmd.issue_time.isoformat()
        }
        
        with open(reports_dir / "resolution_command.json", "w") as f:
            json.dump(resolution_data, f, indent=2)
    
    logger.info(f"Results saved to {reports_dir}/")


def main():
    """Run complete Sprint 2 & 3 demonstration."""
    logger.info("Starting Sprint 2 & 3 Demonstration")
    logger.info("=" * 50)
    
    try:
        # Sprint 2: Conflict Detection
        conflicts = demo_sprint2_detection()
        
        # Sprint 2: Wolfgang Metrics
        metrics_summary = demo_sprint2_metrics(conflicts)
        
        # Sprint 3: LLM Integration
        llm_resolution = demo_sprint3_llm()
        
        # Sprint 3: Safety Validation
        ownship, traffic = create_test_scenario()
        resolution_cmd = demo_sprint3_safety_validation(ownship, traffic, llm_resolution)
        
        # Save results
        save_demo_results(conflicts, metrics_summary, resolution_cmd)
        
        logger.info("\n" + "=" * 50)
        logger.info("Sprint 2 & 3 demonstration completed successfully!")
        logger.info("Check reports/sprint_02/ for detailed results.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
