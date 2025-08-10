#!/usr/bin/env python3
"""Test Enhanced Reporting Functionality

Quick test to verify the enhanced reporting system works correctly.
This test verifies that the enhanced reporting schemas and systems are properly integrated.
"""

import sys
import unittest
import tempfile
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from cdr.schemas import (
        ConflictResolutionMetrics, ScenarioMetrics, PathComparisonMetrics, 
        EnhancedReportingSystem
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestEnhancedReporting(unittest.TestCase):
    """Test enhanced reporting functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.reporting_system = EnhancedReportingSystem(self.temp_dir)
    
    def test_conflict_resolution_metrics_creation(self):
        """Test creating conflict resolution metrics."""
        metrics = ConflictResolutionMetrics(
            conflict_id="TEST_CONFLICT_001",
            ownship_id="TEST_OWNSHIP",
            intruder_id="TEST_INTRUDER",
            resolved=True,
            engine_used="horizontal",
            resolution_type="heading_change",
            waypoint_vs_heading="heading",
            time_to_action_sec=2.5,
            conflict_detection_time=datetime.now(),
            resolution_command_time=datetime.now(),
            initial_distance_nm=8.0,
            min_sep_nm=5.2,
            final_distance_nm=12.0,
            separation_violation=False,
            ownship_cross_track_error_nm=0.5,
            ownship_along_track_error_nm=1.2,
            path_deviation_total_nm=3.0,
            resolution_effectiveness=0.85,
            operational_impact=0.15
        )
        
        self.assertEqual(metrics.conflict_id, "TEST_CONFLICT_001")
        self.assertEqual(metrics.engine_used, "horizontal")
        self.assertTrue(metrics.resolved)
        self.assertFalse(metrics.separation_violation)
        self.assertEqual(metrics.min_sep_nm, 5.2)
    
    def test_scenario_metrics_creation(self):
        """Test creating scenario metrics."""
        metrics = ScenarioMetrics(
            scenario_id="TEST_SCENARIO_001",
            flight_id="TEST_FLIGHT",
            total_conflicts=5,
            conflicts_resolved=4,
            resolution_success_rate=80.0,
            scenario_duration_min=25.0,
            avg_time_to_action_sec=2.1,
            min_separation_achieved_nm=4.8,
            safety_violations=1,
            separation_standards_maintained=False,
            horizontal_engine_usage=3,
            vertical_engine_usage=1,
            deterministic_engine_usage=0,
            fallback_engine_usage=1,
            ownship_path_similarity=0.92,
            total_path_deviation_nm=8.5,
            max_cross_track_error_nm=2.1
        )
        
        self.assertEqual(metrics.scenario_id, "TEST_SCENARIO_001")
        self.assertEqual(metrics.total_conflicts, 5)
        self.assertEqual(metrics.resolution_success_rate, 80.0)
        self.assertEqual(metrics.horizontal_engine_usage, 3)
        self.assertFalse(metrics.separation_standards_maintained)
    
    def test_enhanced_reporting_system(self):
        """Test enhanced reporting system functionality."""
        # Create test conflict metrics
        conflict_metrics = ConflictResolutionMetrics(
            conflict_id="RPT_TEST_CONFLICT_001",
            ownship_id="RPT_OWNSHIP",
            intruder_id="RPT_INTRUDER",
            resolved=True,
            engine_used="vertical",
            resolution_type="altitude_change",
            waypoint_vs_heading="waypoint",
            time_to_action_sec=1.8,
            conflict_detection_time=datetime.now(),
            resolution_command_time=datetime.now(),
            initial_distance_nm=7.5,
            min_sep_nm=5.8,
            final_distance_nm=15.2,
            separation_violation=False,
            ownship_cross_track_error_nm=0.2,
            ownship_along_track_error_nm=0.8,
            path_deviation_total_nm=1.5,
            resolution_effectiveness=0.92,
            operational_impact=0.08
        )
        
        # Add to reporting system
        self.reporting_system.add_conflict_resolution(conflict_metrics)
        
        # Verify it was added
        self.assertEqual(len(self.reporting_system.conflict_metrics), 1)
        self.assertEqual(self.reporting_system.conflict_metrics[0].conflict_id, "RPT_TEST_CONFLICT_001")
        
        # Create test scenario metrics
        scenario_metrics = ScenarioMetrics(
            scenario_id="RPT_TEST_SCENARIO_001",
            flight_id="RPT_FLIGHT",
            total_conflicts=1,
            conflicts_resolved=1,
            resolution_success_rate=100.0,
            scenario_duration_min=15.0,
            avg_time_to_action_sec=1.8,
            min_separation_achieved_nm=5.8,
            safety_violations=0,
            separation_standards_maintained=True,
            vertical_engine_usage=1,
            ownship_path_similarity=0.95,
            total_path_deviation_nm=1.5,
            max_cross_track_error_nm=0.2,
            conflict_resolutions=[conflict_metrics]
        )
        
        # Add to reporting system
        self.reporting_system.add_scenario_completion(scenario_metrics)
        
        # Verify it was added
        self.assertEqual(len(self.reporting_system.scenario_metrics), 1)
        self.assertEqual(self.reporting_system.scenario_metrics[0].scenario_id, "RPT_TEST_SCENARIO_001")
    
    def test_csv_report_generation(self):
        """Test CSV report generation."""
        # Add some test data
        conflict_metrics = ConflictResolutionMetrics(
            conflict_id="CSV_TEST_CONFLICT_001",
            ownship_id="CSV_OWNSHIP",
            intruder_id="CSV_INTRUDER",
            resolved=True,
            engine_used="horizontal",
            resolution_type="heading_change",
            waypoint_vs_heading="heading",
            time_to_action_sec=3.2,
            conflict_detection_time=datetime.now(),
            resolution_command_time=datetime.now(),
            initial_distance_nm=6.5,
            min_sep_nm=5.1,
            final_distance_nm=11.8,
            separation_violation=False,
            resolution_effectiveness=0.78,
            operational_impact=0.22
        )
        
        self.reporting_system.add_conflict_resolution(conflict_metrics)
        
        # Generate CSV report
        csv_path = self.reporting_system.generate_csv_report("test_report.csv")
        
        # Verify file was created
        self.assertTrue(Path(csv_path).exists())
        
        # Read and verify content
        with open(csv_path, 'r') as f:
            content = f.read()
            self.assertIn("CSV_TEST_CONFLICT_001", content)
            self.assertIn("horizontal", content)
            self.assertIn("heading_change", content)
            self.assertIn("5.1", content)  # min_sep_nm
    
    def test_json_report_generation(self):
        """Test JSON report generation."""
        # Add some test data
        conflict_metrics = ConflictResolutionMetrics(
            conflict_id="JSON_TEST_CONFLICT_001",
            ownship_id="JSON_OWNSHIP",
            intruder_id="JSON_INTRUDER",
            resolved=False,
            engine_used="fallback",
            resolution_type="error",
            waypoint_vs_heading="error",
            time_to_action_sec=5.0,
            conflict_detection_time=datetime.now(),
            resolution_command_time=datetime.now(),
            initial_distance_nm=4.2,
            min_sep_nm=4.2,
            final_distance_nm=4.2,
            separation_violation=True,
            resolution_effectiveness=0.0,
            operational_impact=1.0
        )
        
        self.reporting_system.add_conflict_resolution(conflict_metrics)
        
        # Generate JSON report
        json_path = self.reporting_system.generate_json_report("test_report.json")
        
        # Verify file was created
        self.assertTrue(Path(json_path).exists())
        
        # Read and verify content
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.assertIn("metadata", data)
            self.assertIn("conflict_details", data)
            self.assertEqual(len(data["conflict_details"]), 1)
            self.assertEqual(data["conflict_details"][0]["conflict_id"], "JSON_TEST_CONFLICT_001")
            self.assertFalse(data["conflict_details"][0]["resolved"])
            self.assertTrue(data["conflict_details"][0]["separation_violation"])


def main():
    """Run enhanced reporting tests."""
    print("=" * 60)
    print("ENHANCED REPORTING FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Set up test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedReporting)
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL ENHANCED REPORTING TESTS PASSED!")
        print("Enhanced reporting functionality is working correctly.")
        print("\nFeatures verified:")
        print("  ✓ ConflictResolutionMetrics schema")
        print("  ✓ ScenarioMetrics schema")
        print("  ✓ EnhancedReportingSystem class")
        print("  ✓ CSV report generation")
        print("  ✓ JSON report generation")
        print("  ✓ Metrics collection and storage")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print("=" * 60)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
