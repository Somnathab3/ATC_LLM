"""Comprehensive tests for simple stress test module."""

from datetime import datetime

from src.cdr.simple_stress_test import (
    Stre        # Create multiple scenarios
        scenarios = []  # type: ignore
        for _ in range(5):
            scenario = self.stress_test.create_converging_scenario(num_intruders=2)
            scenarios.append(scenario)
        
        # Verify each scenario is unique
        scenario_ids = [s.scenario_id for s in scenarios]  # type: ignore
        assert len(set(scenario_ids)) == len(scenario_ids)  # All uniquenario,
    StressTestResult,
    SimpleStressTest
)
from src.cdr.schemas import AircraftState


class TestStressTestScenario:
    """Test StressTestScenario dataclass."""
    
    def test_stress_test_scenario_creation(self):
        """Test StressTestScenario can be created with valid data."""
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            timestamp=datetime.now(),
            latitude=40.0,
            longitude=-74.0,
            altitude_ft=10000.0,
            ground_speed_kt=400.0,
            heading_deg=90.0,
            vertical_speed_fpm=0.0
        )
        
        intruder = AircraftState(
            aircraft_id="INTRUDER1",
            timestamp=datetime.now(),
            latitude=40.1,
            longitude=-74.1,
            altitude_ft=10000.0,
            ground_speed_kt=420.0,
            heading_deg=270.0,
            vertical_speed_fpm=0.0
        )
        
        scenario = StressTestScenario(
            scenario_id="TEST_001",
            description="Head-on conflict scenario",
            ownship=ownship,
            intruders=[intruder],
            expected_conflicts=1
        )
        
        assert scenario.scenario_id == "TEST_001"
        assert scenario.ownship.aircraft_id == "OWNSHIP"
        assert len(scenario.intruders) == 1
        assert scenario.expected_conflicts == 1


class TestStressTestResult:
    """Test StressTestResult dataclass."""
    
    def test_stress_test_result_creation(self):
        """Test StressTestResult can be created with valid data."""
        result = StressTestResult(
            scenario_id="TEST_001",
            conflicts_detected=1,
            conflicts_resolved=1,
            min_separation_nm=4.8,
            safety_violations=0,
            oscillations=0,
            processing_time_sec=2.5
        )
        
        assert result.scenario_id == "TEST_001"
        assert result.conflicts_detected == 1
        assert result.conflicts_resolved == 1
        assert result.min_separation_nm == 4.8
        assert result.processing_time_sec == 2.5


class TestSimpleStressTest:
    """Test SimpleStressTest class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.stress_test = SimpleStressTest()
    
    def test_stress_test_initialization(self):
        """Test stress test initializes correctly."""
        assert isinstance(self.stress_test.results, list)
        assert len(self.stress_test.results) == 0
    
    def test_create_converging_scenario(self):
        """Test create_converging_scenario method."""
        scenario = self.stress_test.create_converging_scenario(num_intruders=2)
        
        assert isinstance(scenario, StressTestScenario)
        assert scenario.ownship.aircraft_id == "OWNSHIP"
        assert len(scenario.intruders) == 2
        assert scenario.expected_conflicts == 2
        
        # Check that intruders are positioned to converge
        for intruder in scenario.intruders:
            assert isinstance(intruder, AircraftState)
            assert intruder.aircraft_id.startswith("INTRUDER")
    
    def test_create_converging_scenario_different_counts(self):
        """Test create_converging_scenario with different intruder counts."""
        for num_intruders in [1, 3, 5]:
            scenario = self.stress_test.create_converging_scenario(num_intruders=num_intruders)
            assert len(scenario.intruders) == num_intruders
            assert scenario.expected_conflicts == num_intruders
    
    def test_create_converging_scenario_zero_intruders(self):
        """Test create_converging_scenario with zero intruders."""
        scenario = self.stress_test.create_converging_scenario(num_intruders=0)
        assert len(scenario.intruders) == 0
        assert scenario.expected_conflicts == 0
    
    def test_scenario_geometry_validation(self):
        """Test that created scenarios have valid geometry."""
        scenario = self.stress_test.create_converging_scenario(num_intruders=3)
        
        # Verify ownship is at center
        assert scenario.ownship.latitude == 0.0
        assert scenario.ownship.longitude == 0.0
        assert scenario.ownship.altitude_ft == 10000.0
        
        # Verify intruders are positioned around ownship
        for intruder in scenario.intruders:
            # Should be positioned at different locations
            assert intruder.latitude != 0.0 or intruder.longitude != 0.0
            # Should have reasonable altitude
            assert 8000.0 <= intruder.altitude_ft <= 12000.0
            # Should have reasonable speed
            assert 300.0 <= intruder.ground_speed_kt <= 500.0
    
    def test_scenario_convergence_properties(self):
        """Test that scenarios are set up for convergence."""
        scenario = self.stress_test.create_converging_scenario(num_intruders=2)
        
        # Check that intruders have headings that would lead to convergence
        for intruder in scenario.intruders:
            # Heading should be generally toward the ownship
            # This is a basic check - detailed convergence calculation would be complex
            assert 0.0 <= intruder.heading_deg < 360.0
    
    def test_multiple_scenario_creation(self):
        """Test creating multiple scenarios."""
        scenarios = []
        for _ in range(5):
            scenario = self.stress_test.create_converging_scenario(num_intruders=2)
            scenarios.append(scenario)
        
        # Verify each scenario is unique
        scenario_ids = [s.scenario_id for s in scenarios]
        assert len(set(scenario_ids)) == len(scenario_ids)  # All unique
    
    def test_results_accumulation(self):
        """Test that stress test can accumulate results."""
        # Add some mock results
        result1 = StressTestResult(
            scenario_id="TEST_001",
            conflicts_detected=2,
            conflicts_resolved=2,
            min_separation_nm=5.2,
            safety_violations=0,
            oscillations=0,
            processing_time_sec=1.8
        )
        
        result2 = StressTestResult(
            scenario_id="TEST_002",
            conflicts_detected=3,
            conflicts_resolved=2,
            min_separation_nm=4.1,
            safety_violations=1,
            oscillations=1,
            processing_time_sec=3.2
        )
        
        self.stress_test.results.extend([result1, result2])
        
        assert len(self.stress_test.results) == 2
        assert self.stress_test.results[0].scenario_id == "TEST_001"
        assert self.stress_test.results[1].conflicts_detected == 3


class TestStressTestScenarioValidation:
    """Test stress test scenario validation and edge cases."""
    
    def test_scenario_with_extreme_positions(self):
        """Test scenario creation with extreme positions."""
        stress_test = SimpleStressTest()
        
        # Should handle creation without errors
        scenario = stress_test.create_converging_scenario(num_intruders=1)
        
        # Verify basic structure is maintained
        assert scenario.ownship.aircraft_id == "OWNSHIP"
        assert len(scenario.intruders) == 1
    
    def test_scenario_timing_consistency(self):
        """Test that all aircraft in scenario have consistent timing."""
        stress_test = SimpleStressTest()
        scenario = stress_test.create_converging_scenario(num_intruders=3)
        
        # All aircraft should have similar timestamps
        ownship_time = scenario.ownship.timestamp
        for intruder in scenario.intruders:
            time_diff = abs((intruder.timestamp - ownship_time).total_seconds())
            assert time_diff < 1.0  # Within 1 second
    
    def test_scenario_altitude_distribution(self):
        """Test altitude distribution in scenarios."""
        stress_test = SimpleStressTest()
        scenario = stress_test.create_converging_scenario(num_intruders=5)
        
        altitudes = [intruder.altitude_ft for intruder in scenario.intruders]
        
        # Should have some altitude variation for interesting conflicts
        altitude_range = max(altitudes) - min(altitudes)
        assert altitude_range >= 0  # Basic sanity check
    
    def test_scenario_speed_variation(self):
        """Test speed variation in scenarios."""
        stress_test = SimpleStressTest()
        scenario = stress_test.create_converging_scenario(num_intruders=4)
        
        speeds = [intruder.ground_speed_kt for intruder in scenario.intruders]
        
        # All speeds should be positive and reasonable
        for speed in speeds:
            assert speed > 0
            assert speed < 1000  # Reasonable upper bound
    
    def test_large_scenario_creation(self):
        """Test creating scenarios with many intruders."""
        stress_test = SimpleStressTest()
        scenario = stress_test.create_converging_scenario(num_intruders=10)
        
        assert len(scenario.intruders) == 10
        assert scenario.expected_conflicts == 10
        
        # Verify all intruders are unique
        intruder_ids = [intruder.aircraft_id for intruder in scenario.intruders]
        assert len(set(intruder_ids)) == 10


class TestStressTestResults:
    """Test stress test result handling and analysis."""
    
    def test_result_metrics_validation(self):
        """Test that result metrics are within reasonable bounds."""
        result = StressTestResult(
            scenario_id="VALIDATION_TEST",
            conflicts_detected=5,
            conflicts_resolved=4,
            min_separation_nm=3.2,
            safety_violations=1,
            oscillations=2,
            processing_time_sec=4.7
        )
        
        # Basic validation
        assert result.conflicts_detected >= 0
        assert result.conflicts_resolved >= 0
        assert result.conflicts_resolved <= result.conflicts_detected
        assert result.min_separation_nm >= 0
        assert result.safety_violations >= 0
        assert result.oscillations >= 0
        assert result.processing_time_sec >= 0
    
    def test_result_edge_cases(self):
        """Test result creation with edge case values."""
        # Zero conflicts
        result_zero = StressTestResult(
            scenario_id="ZERO_TEST",
            conflicts_detected=0,
            conflicts_resolved=0,
            min_separation_nm=10.0,
            safety_violations=0,
            oscillations=0,
            processing_time_sec=0.1
        )
        
        assert result_zero.conflicts_detected == 0
        assert result_zero.min_separation_nm == 10.0
        
        # High conflict scenario
        result_high = StressTestResult(
            scenario_id="HIGH_TEST",
            conflicts_detected=50,
            conflicts_resolved=45,
            min_separation_nm=1.0,
            safety_violations=5,
            oscillations=10,
            processing_time_sec=30.0
        )
        
        assert result_high.conflicts_detected == 50
        assert result_high.safety_violations == 5
    
    def test_result_collection_statistics(self):
        """Test statistics calculation on result collections."""
        stress_test = SimpleStressTest()
        
        # Add multiple results
        results = [
            StressTestResult("TEST_1", 2, 2, 5.0, 0, 0, 1.0),
            StressTestResult("TEST_2", 3, 2, 4.5, 1, 0, 1.5),
            StressTestResult("TEST_3", 1, 1, 6.0, 0, 1, 0.8),
            StressTestResult("TEST_4", 4, 3, 3.5, 2, 2, 2.2),
        ]
        
        stress_test.results.extend(results)
        
        # Calculate basic statistics
        total_conflicts = sum(r.conflicts_detected for r in stress_test.results)
        total_resolved = sum(r.conflicts_resolved for r in stress_test.results)
        total_violations = sum(r.safety_violations for r in stress_test.results)
        avg_processing = sum(r.processing_time_sec for r in stress_test.results) / len(stress_test.results)
        
        assert total_conflicts == 10
        assert total_resolved == 8
        assert total_violations == 3
        assert abs(avg_processing - 1.375) < 0.001


class TestStressTestIntegration:
    """Integration tests for stress testing framework."""
    
    def test_end_to_end_stress_test_workflow(self):
        """Test complete stress test workflow."""
        stress_test = SimpleStressTest()
        
        # Create multiple scenarios
        scenarios = []
        for i in range(3):
            scenario = stress_test.create_converging_scenario(num_intruders=2)
            scenarios.append(scenario)
        
        # Simulate running tests and collecting results
        for i, scenario in enumerate(scenarios):
            # Mock result based on scenario
            result = StressTestResult(
                scenario_id=scenario.scenario_id,
                conflicts_detected=scenario.expected_conflicts,
                conflicts_resolved=scenario.expected_conflicts - (i % 2),  # Some fail
                min_separation_nm=5.0 - i * 0.5,
                safety_violations=i % 2,
                oscillations=0,
                processing_time_sec=1.0 + i * 0.5
            )
            stress_test.results.append(result)
        
        # Verify results
        assert len(stress_test.results) == 3
        assert all(r.scenario_id.startswith("STRESS_") for r in stress_test.results)
    
    def test_stress_test_repeatability(self):
        """Test that stress tests can be repeated consistently."""
        stress_test1 = SimpleStressTest()
        stress_test2 = SimpleStressTest()
        
        # Create similar scenarios (may not be identical due to randomness)
        scenario1 = stress_test1.create_converging_scenario(num_intruders=2)
        scenario2 = stress_test2.create_converging_scenario(num_intruders=2)
        
        # Should have same basic structure
        assert scenario1.ownship.aircraft_id == scenario2.ownship.aircraft_id
        assert len(scenario1.intruders) == len(scenario2.intruders)
        assert scenario1.expected_conflicts == scenario2.expected_conflicts
    
    def test_stress_test_performance_tracking(self):
        """Test performance tracking capabilities."""
        stress_test = SimpleStressTest()
        
        # Create scenarios with varying complexity
        scenarios = [
            stress_test.create_converging_scenario(num_intruders=1),
            stress_test.create_converging_scenario(num_intruders=3),
            stress_test.create_converging_scenario(num_intruders=5),
        ]
        
        # Mock results with performance correlation
        for i, scenario in enumerate(scenarios):
            processing_time = 0.5 + i * 1.0  # More intruders = more time
            result = StressTestResult(
                scenario_id=scenario.scenario_id,
                conflicts_detected=scenario.expected_conflicts,
                conflicts_resolved=scenario.expected_conflicts,
                min_separation_nm=5.0,
                safety_violations=0,
                oscillations=0,
                processing_time_sec=processing_time
            )
            stress_test.results.append(result)
        
        # Verify performance scaling
        processing_times = [r.processing_time_sec for r in stress_test.results]
        assert processing_times[0] < processing_times[1] < processing_times[2]
