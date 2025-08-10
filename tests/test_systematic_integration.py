#!/usr/bin/env python3
"""Integration test for systematic intruder generation and enhanced metrics.

This test verifies that:
1. Systematic intruder generator creates reproducible scenarios
2. Enhanced metrics collector records Wolfgang (2011) KPIs correctly
3. Integration between components works properly
"""

import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Simple dataclass for testing without pydantic dependency
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AircraftState:
    """Simple aircraft state for testing."""
    aircraft_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    altitude_ft: float
    ground_speed_kt: float
    heading_deg: float
    vertical_speed_fpm: float
    aircraft_type: str
    spawn_offset_min: float

def test_systematic_intruder_generation():
    """Test systematic intruder generation."""
    print("Testing systematic intruder generation...")
    
    generator = SystematicIntruderGenerator(base_seed=12345)
    
    # Create ownship
    ownship = AircraftState(
        aircraft_id="TEST_OWNSHIP",
        timestamp=datetime.now(),
        latitude=59.3293,
        longitude=18.0686,
        altitude_ft=35000,
        ground_speed_kt=450,
        heading_deg=90,
        vertical_speed_fpm=0,
        aircraft_type="A320",
        spawn_offset_min=0
    )
    
    # Test scenario creation
    scenario_params = ScenarioParameters(
        pattern=ConflictPattern.CROSSING,
        severity=ConflictSeverity.MEDIUM,
        cpa_time_min=6.0,
        cpa_distance_nm=2.0,
        seed=12345
    )
    
    # Generate intruder
    intruder = generator.generate_intruder_for_scenario(ownship, scenario_params)
    
    # Verify intruder properties
    assert intruder.aircraft_id == "INTRUDER_CROSSING"
    assert intruder.altitude_ft == ownship.altitude_ft
    assert intruder.ground_speed_kt == 450.0
    
    print("✓ Systematic intruder generation works")
    
    # Test reproducibility
    intruder2 = generator.generate_intruder_for_scenario(ownship, scenario_params)
    assert intruder.latitude == intruder2.latitude
    assert intruder.longitude == intruder2.longitude
    assert intruder.heading_deg == intruder2.heading_deg
    
    print("✓ Reproducible generation works")
    
    # Test scenario set generation
    scenario_set = generator.generate_scenario_set("test")
    assert len(scenario_set.scenarios) > 0
    assert scenario_set.name == "test"
    
    print("✓ Scenario set generation works")
    
    return True

def test_enhanced_metrics_collection():
    """Test enhanced metrics collection."""
    print("\\nTesting enhanced metrics collection...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        collector = EnhancedMetricsCollector(Path(temp_dir))
        
        # Start a run
        run_id = collector.start_run("TEST_AIRCRAFT", "TEST_INTRUDER")
        assert run_id in collector.active_runs
        
        print("✓ Run creation works")
        
        # Record detection metrics
        collector.record_detection(
            run_id=run_id,
            detection_time=2.5,
            cpa_time=6.0,
            cpa_distance=2.0,
            confidence=0.85,
            conflict_type=ConflictType.TRUE_POSITIVE
        )
        
        # Verify detection metrics
        run_metrics = collector.active_runs[run_id]
        assert run_metrics.detection.lat == 6.0
        assert run_metrics.detection.confidence_score == 0.85
        assert run_metrics.detection.tbas > 0.0
        
        print("✓ Detection metrics recording works")
        
        # Record resolution metrics
        collector.record_resolution(
            run_id=run_id,
            action_time=1.5,
            maneuver_type="HDG",
            deviation_amount=20.0,
            success=True,
            intrusion=0.5
        )
        
        # Verify resolution metrics
        assert run_metrics.resolution.rat == 1.5
        assert run_metrics.resolution.maneuver_type == "HDG"
        assert run_metrics.resolution.re > 0.0
        
        print("✓ Resolution metrics recording works")
        
        # Record final outcome
        collector.record_final_outcome(
            run_id=run_id,
            min_separation=5.2,
            conflict_resolved=True,
            safety_maintained=True
        )
        
        # Complete run
        completed_run = collector.complete_run(run_id)
        assert completed_run is not None
        assert completed_run.overall_success
        assert len(collector.completed_runs) == 1
        
        print("✓ Run completion works")
        
        # Test session summary
        summary = collector.get_session_summary()
        assert summary["total_runs"] == 1
        assert summary["successful_runs"] == 1
        assert "wolfgang_2011_kpis" in summary
        
        print("✓ Session summary generation works")
        
        # Test metrics saving
        metrics_file = collector.save_session_metrics()
        assert metrics_file.exists()
        
        # Verify saved content
        with open(metrics_file) as f:
            saved_data = json.load(f)
        assert saved_data["total_runs"] == 1
        assert len(saved_data["runs"]) == 1
        
        print("✓ Metrics saving works")
    
    return True

def test_integration():
    """Test integration between components."""
    print("\\nTesting component integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize components
        generator = SystematicIntruderGenerator(base_seed=54321)
        collector = EnhancedMetricsCollector(Path(temp_dir))
        
        # Create ownship
        ownship = AircraftState(
            aircraft_id="INTEGRATION_TEST",
            timestamp=datetime.now(),
            latitude=40.7128,
            longitude=-74.0060,
            altitude_ft=37000,
            ground_speed_kt=480,
            heading_deg=270,
            vertical_speed_fpm=0,
            aircraft_type="B777",
            spawn_offset_min=0
        )
        
        # Generate scenarios and run them
        scenario_set = generator.generate_scenario_set("integration_test")
        
        # Run first 3 scenarios
        for i, scenario_params in enumerate(scenario_set.scenarios[:3]):
            # Generate intruder
            intruder = generator.generate_intruder_for_scenario(ownship, scenario_params)
            
            # Start metrics collection
            run_id = collector.start_run(
                ownship.aircraft_id,
                intruder.aircraft_id,
                {
                    "scenario_id": f"integration_test_{i}",
                    "pattern": scenario_params.pattern.value,
                    "seed": scenario_params.seed
                }
            )
            
            # Simulate detection and resolution
            collector.record_detection(
                run_id=run_id,
                detection_time=1.0 + i * 0.5,
                cpa_time=scenario_params.cpa_time_min,
                cpa_distance=scenario_params.cpa_distance_nm,
                confidence=0.9,
                conflict_type=ConflictType.TRUE_POSITIVE
            )
            
            collector.record_resolution(
                run_id=run_id,
                action_time=2.0,
                maneuver_type="HDG",
                deviation_amount=15.0,
                success=True,
                intrusion=0.0
            )
            
            collector.record_final_outcome(
                run_id=run_id,
                min_separation=5.5,
                conflict_resolved=True,
                safety_maintained=True
            )
            
            collector.complete_run(run_id)
        
        # Verify integration results
        summary = collector.get_session_summary()
        assert summary["total_runs"] == 3
        assert summary["successful_runs"] == 3
        assert summary["success_rate"] == 1.0
        
        # Verify Wolfgang metrics are calculated
        kpis = summary["wolfgang_2011_kpis"]
        assert kpis["tbas_avg"] > 0.0
        assert kpis["lat_avg_min"] > 0.0
        assert kpis["re_avg"] > 0.0
        
        print("✓ Integration test successful")
        print(f"  - Processed {summary['total_runs']} scenarios")
        print(f"  - Success rate: {summary['success_rate']*100:.0f}%")
        print(f"  - Average TBAS: {kpis['tbas_avg']:.3f}")
        print(f"  - Average LAT: {kpis['lat_avg_min']:.2f} min")
    
    return True

def main():
    """Run all tests."""
    print("Running integration tests for systematic scenarios and enhanced metrics...")
    print("=" * 70)
    
    try:
        # Run individual component tests
        test_systematic_intruder_generation()
        test_enhanced_metrics_collection()
        test_integration()
        
        print("\\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("\\nSystematic intruder generation and enhanced metrics are working correctly!")
        print("Ready for production use with:")
        print("  - Reproducible scenario generation")
        print("  - Wolfgang (2011) KPI collection")
        print("  - Comprehensive metrics logging")
        
    except Exception as e:
        print(f"\\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
