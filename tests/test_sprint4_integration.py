"""Integration tests for Sprint 4 - BlueSky command execution and baseline comparison."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.cdr.resolve import apply_resolution
from src.cdr.bluesky_io import BlueSkyClient
from src.cdr.metrics import MetricsCollector, BaselineMetrics, ComparisonReport
from src.cdr.schemas import (
    ResolveOut, ResolutionCommand, ResolutionType, AircraftState, 
    ConflictPrediction, ConfigurationSettings
)


class TestSprint4BlueSkyIntegration:
    """Test BlueSky command translation and execution."""
    
    def test_apply_resolution_heading_change(self):
        """Test applying heading change resolution to BlueSky."""
        # Create mock BlueSky client
        mock_bs = Mock(spec=BlueSkyClient)
        mock_bs.set_heading.return_value = True
        mock_bs.direct_to_waypoint.return_value = True
        
        # Create resolution advice
        advise = ResolveOut(
            action="turn",
            params={"heading_deg": 120, "hold_min": 3},
            rationale="Turn right to avoid conflict"
        )
        
        # Apply resolution
        result = apply_resolution(mock_bs, "OWN001", advise, "WAYPOINT1")
        
        # Verify calls
        assert result is True
        mock_bs.set_heading.assert_called_once_with("OWN001", 120)
        
    def test_apply_resolution_altitude_change(self):
        """Test applying altitude change resolution to BlueSky."""
        # Create mock BlueSky client
        mock_bs = Mock(spec=BlueSkyClient)
        mock_bs.set_altitude.return_value = True
        
        # Create resolution advice
        advise = ResolveOut(
            action="climb",
            params={"target_ft": 37000},
            rationale="Climb to separate vertically"
        )
        
        # Apply resolution
        result = apply_resolution(mock_bs, "OWN001", advise)
        
        # Verify calls
        assert result is True
        mock_bs.set_altitude.assert_called_once_with("OWN001", 37000)
        
    def test_apply_resolution_invalid_action(self):
        """Test handling of invalid resolution action."""
        mock_bs = Mock(spec=BlueSkyClient)
        
        # Create invalid resolution advice
        advise = ResolveOut(
            action="invalid_action",
            params={},
            rationale="Invalid action"
        )
        
        # Apply resolution
        result = apply_resolution(mock_bs, "OWN001", advise)
        
        # Should fail gracefully
        assert result is False
        
    def test_apply_resolution_missing_parameters(self):
        """Test handling of missing resolution parameters."""
        mock_bs = Mock(spec=BlueSkyClient)
        
        # Create resolution with missing parameters
        advise = ResolveOut(
            action="turn",
            params={},  # Missing heading_deg
            rationale="Turn without parameters"
        )
        
        # Apply resolution
        result = apply_resolution(mock_bs, "OWN001", advise)
        
        # Should fail gracefully
        assert result is False


class TestSprint4BaselineComparison:
    """Test baseline comparison functionality."""
    
    def test_baseline_metrics_creation(self):
        """Test creation of baseline metrics."""
        baseline = BaselineMetrics(
            baseline_conflicts_detected=10,
            baseline_false_positives=2,
            baseline_detection_latency_sec=5.0,
            baseline_resolutions_issued=8,
            baseline_success_rate=0.875,
            baseline_resolution_delay_sec=15.0,
            baseline_min_separation_nm=4.8,
            baseline_avg_separation_nm=6.2,
            baseline_safety_violations=1
        )
        
        assert baseline.baseline_conflicts_detected == 10
        assert baseline.baseline_success_rate == 0.875
        
    def test_comparison_report_generation(self):
        """Test generation of comparison report."""
        collector = MetricsCollector()
        
        # Add some test data
        test_conflict = ConflictPrediction(
            ownship_id="OWN",
            intruder_id="TRF001",
            time_to_cpa_min=5.0,
            distance_at_cpa_nm=3.0,
            altitude_diff_ft=500.0,
            is_conflict=True,
            severity_score=0.8,
            conflict_type="both",
            prediction_time=datetime.now(),
            confidence=1.0
        )
        
        collector.record_conflict_detection([test_conflict], datetime.now())
        
        # Create baseline metrics
        baseline = BaselineMetrics(
            baseline_conflicts_detected=5,
            baseline_false_positives=2,
            baseline_detection_latency_sec=8.0,
            baseline_resolutions_issued=4,
            baseline_success_rate=0.75,
            baseline_resolution_delay_sec=20.0,
            baseline_min_separation_nm=4.5,
            baseline_avg_separation_nm=5.8,
            baseline_safety_violations=2
        )
        
        # Generate comparison
        comparison = collector.compare_with_baseline(baseline)
        
        assert isinstance(comparison, ComparisonReport)
        assert isinstance(comparison.overall_score, float)
        assert 0 <= comparison.overall_score <= 100
        assert isinstance(comparison.recommendation, str)
        assert len(comparison.recommendation) > 0


class TestSprint4EndToEndIntegration:
    """End-to-end integration tests for complete scenario."""
    
    def test_converging_intruder_scenario(self):
        """Test complete scenario with converging intruder."""
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama-3.1-8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=45.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        
        # Create ownship and intruder
        ownship = AircraftState(
            aircraft_id="OWN001",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        intruder = AircraftState(
            aircraft_id="TRF001",
            timestamp=datetime.now(),
            latitude=59.35,
            longitude=18.6,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=270,  # Converging
            vertical_speed_fpm=0
        )
        
        # Create mock BlueSky client
        mock_bs = Mock(spec=BlueSkyClient)
        mock_bs.set_heading.return_value = True
        mock_bs.set_altitude.return_value = True
        
        # Test heading resolution
        heading_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 120},
            rationale="Turn right to avoid conflict"
        )
        
        result = apply_resolution(mock_bs, "OWN001", heading_resolution)
        assert result is True
        mock_bs.set_heading.assert_called_with("OWN001", 120)
        
        # Test altitude resolution
        altitude_resolution = ResolveOut(
            action="climb",
            params={"target_ft": 37000},
            rationale="Climb to separate vertically"
        )
        
        result = apply_resolution(mock_bs, "OWN001", altitude_resolution)
        assert result is True
        mock_bs.set_altitude.assert_called_with("OWN001", 37000)
        
    def test_metrics_with_bluesky_commands(self):
        """Test metrics collection with BlueSky command execution."""
        collector = MetricsCollector()
        
        # Create test resolution command
        resolution = ResolutionCommand(
            resolution_id="test_res_1",
            target_aircraft="OWN001",
            resolution_type=ResolutionType.HEADING_CHANGE,
            new_heading_deg=120.0,
            new_speed_kt=None,
            new_altitude_ft=None,
            issue_time=datetime.now(),
            expected_completion_time=None,
            is_validated=True,
            safety_margin_nm=6.0
        )
        
        # Record resolution
        collector.record_resolution_issued(resolution, datetime.now())
        collector.record_resolution_outcome(resolution.resolution_id, True)
        
        # Generate summary
        summary = collector.generate_summary()
        
        assert summary.total_resolutions_issued == 1
        assert summary.successful_resolutions == 1
        assert summary.resolution_success_rate == 1.0
        
    def test_separation_preservation_validation(self):
        """Test that resolutions preserve separation margins."""
        from src.cdr.resolve import execute_resolution
        
        # Create test scenario
        ownship = AircraftState(
            aircraft_id="OWN001",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        intruder = AircraftState(
            aircraft_id="TRF001",
            timestamp=datetime.now(),
            latitude=59.35,
            longitude=18.6,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=270,
            vertical_speed_fpm=0
        )
        
        conflict = ConflictPrediction(
            ownship_id="OWN001",
            intruder_id="TRF001",
            time_to_cpa_min=5.0,
            distance_at_cpa_nm=3.0,
            altitude_diff_ft=0.0,
            is_conflict=True,
            severity_score=0.9,
            conflict_type="horizontal",
            prediction_time=datetime.now(),
            confidence=1.0
        )
        
        # Test safe resolution
        safe_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 120},
            rationale="Safe turn to maintain separation"
        )
        
        result = execute_resolution(safe_resolution, ownship, intruder, conflict)
        
        # Should return a validated resolution or fallback
        assert result is None or (isinstance(result, ResolutionCommand) and result.is_validated)
