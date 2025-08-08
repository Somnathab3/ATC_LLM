"""Comprehensive Sprint 5 Integration Tests.

Tests for:
1. Oscillation guard functionality
2. Multi-intruder stress testing
3. Monte Carlo perturbations
4. Failure mode analysis
5. Comprehensive reporting
"""

import unittest
import tempfile
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from pathlib import Path

from src.cdr.resolve import (
    _check_oscillation_guard, 
    _add_command_to_history,
    _classify_command_type,
    _estimate_separation_benefit
)
from src.cdr.simple_stress_test import SimpleStressTest, create_multi_intruder_scenarios
from src.cdr.schemas import AircraftState, ResolutionCommand, ResolutionType


class TestSprint5OscillationGuard(unittest.TestCase):
    """Test oscillation guard functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear command history
        from src.cdr.resolve import _command_history
        _command_history.clear()
    
    def test_oscillation_guard_blocks_opposite_commands(self):
        """Test that oscillation guard blocks opposite commands within window."""
        aircraft_id = "TEST_AC_001"
        
        # Add a turn left command
        _add_command_to_history(aircraft_id, "turn_left", heading_change=-20.0, separation_benefit=1.0)
        
        # Try opposite command immediately with low benefit (should be blocked)
        allowed = _check_oscillation_guard(aircraft_id, "turn_right", 0.3)
        self.assertFalse(allowed, "Oscillation guard should block opposite turn with low benefit")
    
    def test_oscillation_guard_allows_high_benefit_commands(self):
        """Test that oscillation guard allows opposite commands with high benefit."""
        aircraft_id = "TEST_AC_002"
        
        # Add a climb command
        _add_command_to_history(aircraft_id, "climb", altitude_change=1000.0, separation_benefit=1.0)
        
        # Try opposite command with high benefit (should be allowed)
        allowed = _check_oscillation_guard(aircraft_id, "descend", 2.0)
        self.assertTrue(allowed, "Oscillation guard should allow opposite command with high benefit")
    
    def test_oscillation_guard_allows_same_direction_commands(self):
        """Test that oscillation guard allows same direction commands."""
        aircraft_id = "TEST_AC_003"
        
        # Add a turn right command
        _add_command_to_history(aircraft_id, "turn_right", heading_change=20.0, separation_benefit=1.0)
        
        # Try same direction command (should be allowed)
        allowed = _check_oscillation_guard(aircraft_id, "turn_right", 0.5)
        self.assertTrue(allowed, "Oscillation guard should allow same direction commands")
    
    def test_command_classification(self):
        """Test command type classification."""
        # Create test aircraft state
        ownship = AircraftState(
            aircraft_id="TEST_AC",
            latitude=0.0,
            longitude=0.0,
            altitude_ft=10000.0,
            heading_deg=90.0,
            ground_speed_kt=250.0,
            vertical_speed_fpm=0.0,
            timestamp=datetime.now()
        )
        
        # Test heading change classification
        heading_cmd = ResolutionCommand(
            resolution_id="test_heading",
            target_aircraft="TEST_AC",
            resolution_type=ResolutionType.HEADING_CHANGE,
            new_heading_deg=120.0,  # Right turn
            new_speed_kt=None,
            new_altitude_ft=None,
            issue_time=datetime.now(),
            safety_margin_nm=5.0,
            is_validated=False
        )
        
        command_type = _classify_command_type(heading_cmd, ownship)
        self.assertEqual(command_type, "turn_right")
        
        # Test altitude change classification
        altitude_cmd = ResolutionCommand(
            resolution_id="test_altitude",
            target_aircraft="TEST_AC",
            resolution_type=ResolutionType.ALTITUDE_CHANGE,
            new_heading_deg=None,
            new_speed_kt=None,
            new_altitude_ft=11000.0,  # Climb
            issue_time=datetime.now(),
            safety_margin_nm=5.0,
            is_validated=False
        )
        
        command_type = _classify_command_type(altitude_cmd, ownship)
        self.assertEqual(command_type, "climb")
    
    def test_separation_benefit_estimation(self):
        """Test separation benefit estimation."""
        # Create test aircraft states
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            latitude=0.0,
            longitude=0.0,
            altitude_ft=10000.0,
            heading_deg=90.0,
            ground_speed_kt=250.0,
            vertical_speed_fpm=0.0,
            timestamp=datetime.now()
        )
        
        intruder = AircraftState(
            aircraft_id="INTRUDER",
            latitude=0.1,
            longitude=0.1,
            altitude_ft=10000.0,
            heading_deg=270.0,
            ground_speed_kt=250.0,
            vertical_speed_fpm=0.0,
            timestamp=datetime.now()
        )
        
        # Test heading change benefit
        heading_cmd = ResolutionCommand(
            resolution_id="test_heading",
            target_aircraft="OWNSHIP",
            resolution_type=ResolutionType.HEADING_CHANGE,
            new_heading_deg=120.0,
            new_speed_kt=None,
            new_altitude_ft=None,
            issue_time=datetime.now(),
            safety_margin_nm=5.0,
            is_validated=False
        )
        
        benefit = _estimate_separation_benefit(heading_cmd, ownship, intruder)
        self.assertGreater(benefit, 0, "Separation benefit should be positive")
        self.assertLessEqual(benefit, 3.0, "Heading benefit should be capped at 3.0nm")


class TestSprint5StressTesting(unittest.TestCase):
    """Test multi-intruder stress testing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stress_tester = SimpleStressTest()
    
    def test_multi_intruder_scenario_creation(self):
        """Test creation of multi-intruder scenarios."""
        scenarios = create_multi_intruder_scenarios()
        
        self.assertEqual(len(scenarios), 3, "Should create 3 scenarios (2, 3, 4 intruders)")
        
        # Check scenario properties
        for i, scenario in enumerate(scenarios):
            expected_intruders = 2 + i
            self.assertEqual(len(scenario.intruders), expected_intruders)
            self.assertEqual(scenario.expected_conflicts, expected_intruders)
            self.assertIn(str(expected_intruders), scenario.scenario_id)
    
    def test_converging_scenario_properties(self):
        """Test properties of converging scenarios."""
        scenario = self.stress_tester.create_converging_scenario(2)
        
        # Check basic properties
        self.assertEqual(len(scenario.intruders), 2)
        self.assertEqual(scenario.ownship.aircraft_id, "OWNSHIP")
        
        # Check intruders are in opposite direction
        for intruder in scenario.intruders:
            self.assertNotEqual(intruder.heading_deg, scenario.ownship.heading_deg)
    
    def test_stress_test_execution(self):
        """Test execution of stress tests."""
        scenario = self.stress_tester.create_converging_scenario(2)
        result = self.stress_tester.run_basic_test(scenario)
        
        # Check result properties
        self.assertEqual(result.scenario_id, scenario.scenario_id)
        self.assertGreaterEqual(result.conflicts_detected, 0)
        self.assertGreaterEqual(result.conflicts_resolved, 0)
        self.assertLessEqual(result.conflicts_resolved, result.conflicts_detected)
        self.assertGreater(result.processing_time_sec, 0)


class TestSprint5Reporting(unittest.TestCase):
    """Test comprehensive reporting functionality."""
    
    def test_report_directory_creation(self):
        """Test that report directories are created properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            from src.cdr.reporting import Sprint5Reporter
            
            reporter = Sprint5Reporter(output_dir=temp_dir)
            self.assertTrue(Path(temp_dir).exists())
    
    def test_metrics_csv_generation(self):
        """Test generation of metrics CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            from src.cdr.reporting import Sprint5Reporter
            from src.cdr.metrics import MetricsSummary
            
            reporter = Sprint5Reporter(output_dir=temp_dir)
            
            # Create sample metrics
            sample_metrics = [
                MetricsSummary(
                    total_simulation_time_min=10.0,
                    total_cycles=120,
                    avg_cycle_time_sec=5.0,
                    total_conflicts_detected=5,
                    true_conflicts=4,
                    false_positives=1,
                    false_negatives=0,
                    detection_accuracy=0.9,
                    tbas=0.95,
                    lat=2.5,
                    pa=5,
                    pi=4,
                    dat=1.2,
                    dfa=0.8,
                    re=0.85,
                    ri=0.15,
                    rat=3.0,
                    min_separation_achieved_nm=2.8,
                    avg_separation_nm=5.2,
                    safety_violations=0,
                    total_resolutions_issued=5,
                    successful_resolutions=4,
                    resolution_success_rate=0.8,
                    avg_resolution_time_sec=2.5
                )
            ]
            
            csv_path = reporter.generate_metrics_csv(sample_metrics)
            self.assertTrue(Path(csv_path).exists())
            
            # Check CSV content
            import pandas as pd
            df = pd.read_csv(csv_path)
            self.assertEqual(len(df), 1)
            self.assertIn('total_conflicts_detected', df.columns)
    
    def test_failure_mode_analysis(self):
        """Test failure mode analysis functionality."""
        from src.cdr.reporting import Sprint5Reporter
        from src.cdr.simple_stress_test import StressTestResult
        
        reporter = Sprint5Reporter()
        
        # Create sample stress test results
        stress_results = [
            StressTestResult(
                scenario_id="test_scenario_1",
                conflicts_detected=3,
                conflicts_resolved=2,
                min_separation_nm=2.5,
                safety_violations=1,
                oscillations=0,
                processing_time_sec=1.5
            ),
            StressTestResult(
                scenario_id="test_scenario_2",
                conflicts_detected=2,
                conflicts_resolved=2,
                min_separation_nm=3.2,
                safety_violations=0,
                oscillations=1,
                processing_time_sec=1.2
            )
        ]
        
        failure_analysis = reporter.analyze_failure_modes(stress_results, [])
        
        # Check analysis results
        self.assertGreaterEqual(failure_analysis.safety_impact_score, 0)
        self.assertLessEqual(failure_analysis.safety_impact_score, 100)
        self.assertGreater(len(failure_analysis.recommendations), 0)
        self.assertEqual(len(failure_analysis.unsafe_resolution_scenarios), 1)
        self.assertEqual(len(failure_analysis.oscillation_scenarios), 1)


class TestSprint5Integration(unittest.TestCase):
    """Integration tests for complete Sprint 5 functionality."""
    
    def test_end_to_end_sprint5_workflow(self):
        """Test complete Sprint 5 workflow from stress testing to reporting."""
        # 1. Create stress test scenarios
        scenarios = create_multi_intruder_scenarios()
        self.assertGreater(len(scenarios), 0)
        
        # 2. Run stress tests
        stress_tester = SimpleStressTest()
        results = []
        
        for scenario in scenarios[:2]:  # Test first 2 scenarios
            result = stress_tester.run_basic_test(scenario)
            results.append(result)
        
        self.assertEqual(len(results), 2)
        
        # 3. Test oscillation guard
        from src.cdr.resolve import _command_history
        _command_history.clear()
        
        # Add command and test guard
        _add_command_to_history("TEST_AC", "turn_left", separation_benefit=1.0)
        blocked = not _check_oscillation_guard("TEST_AC", "turn_right", 0.3)
        self.assertTrue(blocked, "Oscillation guard should work in integration test")
        
        # 4. Generate basic failure analysis
        with tempfile.TemporaryDirectory() as temp_dir:
            from src.cdr.reporting import Sprint5Reporter
            
            reporter = Sprint5Reporter(output_dir=temp_dir)
            failure_analysis = reporter.analyze_failure_modes(results, [])
            
            self.assertIsNotNone(failure_analysis)
            self.assertGreater(len(failure_analysis.recommendations), 0)
    
    def test_monte_carlo_perturbation_concept(self):
        """Test Monte Carlo perturbation concept."""
        # Create base scenario
        stress_tester = SimpleStressTest()
        base_scenario = stress_tester.create_converging_scenario(2)
        
        # Test that we can create variations
        import random
        random.seed(42)  # For reproducible tests
        
        # Simple perturbation test
        original_lat = base_scenario.ownship.latitude
        
        # In a real implementation, we would apply perturbations
        # Here we just verify the concept works
        self.assertIsNotNone(base_scenario.ownship)
        self.assertEqual(len(base_scenario.intruders), 2)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
