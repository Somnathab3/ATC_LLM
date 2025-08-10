"""Comprehensive test suite for metrics collection module.

Tests metrics.py functionality including:
- Minimum separation computation
- Violation counters
- Engine usage tracking
- Time to action metrics
- Wolfgang KPI calculations

Coverage target: ≥65%
"""

import pytest
import numpy as np
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from src.cdr.metrics import (
    MetricsCollector,
    MetricsSummary,
    BaselineMetrics,
    ComparisonReport,
    summarize_run
)
from src.cdr.schemas import (
    ConflictPrediction,
    ResolutionCommand,
    ResolutionType,
    ResolutionEngine
)


class TestMetricsImports:
    """Test metrics module imports."""
    
    def test_module_imports(self):
        """Test that key metrics classes can be imported."""
        from src.cdr.metrics import MetricsCollector
        assert MetricsCollector is not None
        
        from src.cdr.metrics import MetricsSummary
        assert MetricsSummary is not None
        
        from src.cdr.metrics import BaselineMetrics
        assert BaselineMetrics is not None
        
        from src.cdr.metrics import ComparisonReport
        assert ComparisonReport is not None


class TestMetricsCollectorInitialization:
    """Test MetricsCollector initialization and basic functionality."""
    
    def test_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        
        assert collector is not None
        assert hasattr(collector, 'start_time')
        assert hasattr(collector, 'cycle_times')
        assert hasattr(collector, 'conflicts_detected')
        assert hasattr(collector, 'resolutions_issued')
        assert isinstance(collector.start_time, datetime)
        assert isinstance(collector.cycle_times, list)
        assert isinstance(collector.conflicts_detected, list)
        assert isinstance(collector.resolutions_issued, list)
        assert collector.run_label == "run"
    
    def test_reset(self):
        """Test metrics reset functionality."""
        collector = MetricsCollector()
        
        # Add some data
        collector.record_cycle_time(2.5)
        initial_start_time = collector.start_time
        
        # Reset
        collector.reset()
        
        # Should be empty again but start_time updated
        assert len(collector.cycle_times) == 0
        assert isinstance(collector.start_time, datetime)
        assert collector.start_time >= initial_start_time
        assert len(collector.conflicts_detected) == 0
        assert len(collector.resolutions_issued) == 0


class TestCycleTimeRecording:
    """Test cycle time recording functionality."""
    
    def setup_method(self):
        """Set up collector for each test."""
        self.collector = MetricsCollector()
    
    def test_record_cycle_time(self):
        """Test recording cycle execution times."""
        # Record some cycle times
        self.collector.record_cycle_time(2.5)
        self.collector.record_cycle_time(3.1)
        self.collector.record_cycle_time(2.8)
        
        assert len(self.collector.cycle_times) == 3
        assert 2.5 in self.collector.cycle_times
        assert 3.1 in self.collector.cycle_times
        assert 2.8 in self.collector.cycle_times
    
    def test_cycle_time_statistics(self):
        """Test cycle time statistics calculation."""
        cycle_times = [1.5, 2.0, 1.8, 2.2, 1.9]
        for time in cycle_times:
            self.collector.record_cycle_time(time)
        
        assert len(self.collector.cycle_times) == 5
        assert min(self.collector.cycle_times) == 1.5
        assert max(self.collector.cycle_times) == 2.2
        
        # Calculate average
        avg_time = sum(self.collector.cycle_times) / len(self.collector.cycle_times)
        assert abs(avg_time - 1.88) < 0.01


class TestConflictDetectionRecording:
    """Test conflict detection recording."""
    
    def setup_method(self):
        """Set up collector for each test."""
        self.collector = MetricsCollector()
    
    def create_conflict_prediction(self, ownship_id: str, intruder_id: str, 
                                   is_conflict: bool = True, time_to_cpa: float = 3.0) -> ConflictPrediction:
        """Create a test conflict prediction."""
        return ConflictPrediction(
            ownship_id=ownship_id,
            intruder_id=intruder_id,
            time_to_cpa_min=time_to_cpa,
            distance_at_cpa_nm=2.5,
            altitude_diff_ft=500,
            is_conflict=is_conflict,
            severity_score=0.8,
            conflict_type="horizontal",
            prediction_time=datetime.now(),
            confidence=0.95
        )
    
    def test_record_conflict_detection(self):
        """Test recording conflict detection results."""
        conflicts = [
            self.create_conflict_prediction("AC001", "AC002", is_conflict=True),
            self.create_conflict_prediction("AC003", "AC004", is_conflict=False)
        ]
        
        detection_time = datetime.now()
        self.collector.record_conflict_detection(conflicts, detection_time)
        
        assert len(self.collector.conflicts_detected) == 2
        assert len(self.collector.alert_times) == 2
        
        # Check alert times recorded
        assert "AC001_AC002" in self.collector.alert_times
        assert "AC003_AC004" in self.collector.alert_times
    
    def test_multiple_detections_same_pair(self):
        """Test multiple detections for same aircraft pair."""
        conflict1 = self.create_conflict_prediction("AC001", "AC002", time_to_cpa=5.0)
        conflict2 = self.create_conflict_prediction("AC001", "AC002", time_to_cpa=3.0)
        
        time1 = datetime.now()
        time2 = time1 + timedelta(seconds=30)
        
        self.collector.record_conflict_detection([conflict1], time1)
        self.collector.record_conflict_detection([conflict2], time2)
        
        # Should have 2 conflicts but only 1 alert time (first one)
        assert len(self.collector.conflicts_detected) == 2
        assert len(self.collector.alert_times) == 1
        assert self.collector.alert_times["AC001_AC002"] == time1


class TestResolutionRecording:
    """Test resolution command recording."""
    
    def setup_method(self):
        """Set up collector for each test."""
        self.collector = MetricsCollector()
    
    def create_resolution_command(self, resolution_id: str, target_aircraft: str,
                                  engine: ResolutionEngine = ResolutionEngine.HORIZONTAL) -> ResolutionCommand:
        """Create a test resolution command."""
        return ResolutionCommand(
            resolution_id=resolution_id,
            target_aircraft=target_aircraft,
            resolution_type=ResolutionType.HEADING_CHANGE,
            source_engine=engine,
            new_heading_deg=90.0,
            new_speed_kt=None,
            new_altitude_ft=None,
            waypoint_name=None,
            waypoint_lat=None,
            waypoint_lon=None,
            diversion_distance_nm=None,
            issue_time=datetime.now(),
            is_validated=False,
            safety_margin_nm=5.0,
            is_ownship_command=True,
            angle_within_limits=True,
            altitude_within_limits=True,
            rate_within_limits=True
        )
    
    def test_record_resolution_issued(self):
        """Test recording resolution issuance."""
        resolution = self.create_resolution_command("RES001", "AC001")
        issue_time = datetime.now()
        
        self.collector.record_resolution_issued(resolution, issue_time)
        
        assert len(self.collector.resolutions_issued) == 1
        assert "RES001" in self.collector.resolution_timings
        
        start_time, end_time = self.collector.resolution_timings["RES001"]
        assert start_time == issue_time
        assert end_time is None
    
    def test_record_resolution_outcome(self):
        """Test recording resolution execution outcome."""
        resolution = self.create_resolution_command("RES001", "AC001")
        issue_time = datetime.now()
        completion_time = issue_time + timedelta(seconds=45)
        
        self.collector.record_resolution_issued(resolution, issue_time)
        self.collector.record_resolution_outcome("RES001", True, completion_time)
        
        assert self.collector.resolution_outcomes["RES001"] is True
        
        start_time, end_time = self.collector.resolution_timings["RES001"]
        assert start_time == issue_time
        assert end_time == completion_time
    
    def test_resolution_outcome_without_timing(self):
        """Test recording outcome without completion time."""
        resolution = self.create_resolution_command("RES001", "AC001")
        issue_time = datetime.now()
        
        self.collector.record_resolution_issued(resolution, issue_time)
        self.collector.record_resolution_outcome("RES001", False)
        
        assert self.collector.resolution_outcomes["RES001"] is False
        
        start_time, end_time = self.collector.resolution_timings["RES001"]
        assert start_time == issue_time
        assert end_time is None


class TestSeparationRecording:
    """Test aircraft separation recording and violation detection."""
    
    def setup_method(self):
        """Set up collector for each test."""
        self.collector = MetricsCollector()
    
    def test_record_separation_safe(self):
        """Test recording safe separation distances."""
        time = datetime.now()
        
        self.collector.record_separation(time, "AC001", "AC002", 8.5)
        self.collector.record_separation(time + timedelta(seconds=30), "AC001", "AC002", 7.2)
        
        assert len(self.collector.separation_history) == 2
        assert len(self.collector.safety_violations) == 0
    
    def test_record_separation_violation(self):
        """Test recording separation violations."""
        time = datetime.now()
        
        # Safe separation first
        self.collector.record_separation(time, "AC001", "AC002", 6.0)
        
        # Then violation
        violation_time = time + timedelta(seconds=30)
        self.collector.record_separation(violation_time, "AC001", "AC002", 3.5)
        
        assert len(self.collector.separation_history) == 2
        assert len(self.collector.safety_violations) == 1
        
        violation = self.collector.safety_violations[0]
        assert violation[0] == violation_time
        assert violation[1] == "AC001"
        assert violation[2] == "AC002"
        assert violation[3] == 3.5
    
    def test_minimum_separation_tracking(self):
        """Test tracking of minimum separation achieved."""
        time = datetime.now()
        separations = [8.0, 6.5, 4.2, 5.8, 7.1]
        
        for i, sep in enumerate(separations):
            self.collector.record_separation(
                time + timedelta(seconds=i*30), 
                "AC001", "AC002", 
                sep
            )
        
        # Extract separations for analysis
        recorded_seps = [sep for _, _, _, sep in self.collector.separation_history]
        min_separation = min(recorded_seps)
        
        assert min_separation == 4.2
        assert len(self.collector.safety_violations) == 1  # Only 4.2 < 5.0


class TestWolfgangKPIsCalculation:
    """Test Wolfgang KPI calculations."""
    
    def setup_method(self):
        """Set up collector for each test."""
        self.collector = MetricsCollector()
    
    def add_sample_conflicts(self):
        """Add sample conflicts for testing."""
        conflicts = [
            ConflictPrediction(
                ownship_id="AC001",
                intruder_id="AC002",
                time_to_cpa_min=3.0,
                distance_at_cpa_nm=2.0,
                altitude_diff_ft=500,
                is_conflict=True,
                severity_score=0.8,
                conflict_type="horizontal",
                prediction_time=datetime.now(),
                confidence=0.95
            ),
            ConflictPrediction(
                ownship_id="AC003",
                intruder_id="AC004",
                time_to_cpa_min=4.5,
                distance_at_cpa_nm=1.5,
                altitude_diff_ft=300,
                is_conflict=True,
                severity_score=0.9,
                conflict_type="horizontal",
                prediction_time=datetime.now(),
                confidence=0.98
            ),
            ConflictPrediction(
                ownship_id="AC005",
                intruder_id="AC006",
                time_to_cpa_min=2.0,
                distance_at_cpa_nm=6.0,
                altitude_diff_ft=1200,
                is_conflict=False,
                severity_score=0.3,
                conflict_type="none",
                prediction_time=datetime.now(),
                confidence=0.85
            )
        ]
        
        self.collector.record_conflict_detection(conflicts, datetime.now())
    
    def test_calculate_wolfgang_kpis_basic(self):
        """Test basic Wolfgang KPI calculation."""
        self.add_sample_conflicts()
        
        kpis = self.collector.calculate_wolfgang_kpis()
        
        assert isinstance(kpis, dict)
        assert 'tbas' in kpis
        assert 'lat' in kpis
        assert 'pa' in kpis
        assert 'pi' in kpis
        assert 'dat' in kpis
        assert 'dfa' in kpis
        assert 're' in kpis
        assert 'ri' in kpis
        assert 'rat' in kpis
        
        # Check specific values
        assert kpis['pa'] == 3  # Total predicted alerts
        assert kpis['pi'] == 2  # Predicted intrusions (conflicts)
    
    def test_tbas_calculation(self):
        """Test TBAS calculation with synthetic data."""
        # Add conflicts with different detection times
        conflicts = [
            ConflictPrediction(
                ownship_id="AC001",
                intruder_id="AC002",
                time_to_cpa_min=6.0,  # Good advance warning
                distance_at_cpa_nm=2.0,
                altitude_diff_ft=500,
                is_conflict=True,
                severity_score=0.8,
                conflict_type="horizontal",
                prediction_time=datetime.now(),
                confidence=0.95
            ),
            ConflictPrediction(
                ownship_id="AC003",
                intruder_id="AC004",
                time_to_cpa_min=2.0,  # Late detection
                distance_at_cpa_nm=1.5,
                altitude_diff_ft=300,
                is_conflict=True,
                severity_score=0.9,
                conflict_type="horizontal",
                prediction_time=datetime.now(),
                confidence=0.98
            )
        ]
        
        self.collector.record_conflict_detection(conflicts, datetime.now())
        
        kpis = self.collector.calculate_wolfgang_kpis()
        
        # TBAS should be between 0 and 1
        assert 0.0 <= kpis['tbas'] <= 1.0
    
    def test_empty_conflicts_kpis(self):
        """Test KPI calculation with no conflicts."""
        kpis = self.collector.calculate_wolfgang_kpis()
        
        # Should return zeros for empty data
        assert kpis['tbas'] == 0.0
        assert kpis['lat'] == 0.0
        assert kpis['pa'] == 0
        assert kpis['pi'] == 0
        assert kpis['dat'] == 0.0
        assert kpis['dfa'] == 0.0
        assert kpis['re'] == 0.0
        assert kpis['ri'] == 0.0
        assert kpis['rat'] == 0.0


class TestGroundTruthRecording:
    """Test ground truth conflict recording."""
    
    def setup_method(self):
        """Set up collector for each test."""
        self.collector = MetricsCollector()
    
    def test_record_ground_truth(self):
        """Test recording ground truth conflicts."""
        true_conflicts = [
            ConflictPrediction(
                ownship_id="AC001",
                intruder_id="AC002",
                time_to_cpa_min=3.0,
                distance_at_cpa_nm=1.8,
                altitude_diff_ft=400,
                is_conflict=True,
                severity_score=0.9,
                conflict_type="horizontal",
                prediction_time=datetime.now(),
                confidence=1.0
            )
        ]
        
        self.collector.record_ground_truth(true_conflicts)
        
        assert len(self.collector.ground_truth_conflicts) == 1
        assert self.collector.ground_truth_conflicts[0].ownship_id == "AC001"


class TestMetricsSummaryGeneration:
    """Test comprehensive metrics summary generation."""
    
    def setup_method(self):
        """Set up collector with sample data."""
        self.collector = MetricsCollector()
        
        # Add sample cycle times
        for cycle_time in [2.1, 2.3, 1.9, 2.5, 2.0]:
            self.collector.record_cycle_time(cycle_time)
        
        # Add sample conflicts
        conflicts = [
            ConflictPrediction(
                ownship_id="AC001",
                intruder_id="AC002",
                time_to_cpa_min=3.0,
                distance_at_cpa_nm=2.0,
                altitude_diff_ft=500,
                is_conflict=True,
                severity_score=0.8,
                conflict_type="horizontal",
                prediction_time=datetime.now(),
                confidence=0.95
            ),
            ConflictPrediction(
                ownship_id="AC003",
                intruder_id="AC004",
                time_to_cpa_min=2.0,
                distance_at_cpa_nm=7.0,
                altitude_diff_ft=1200,
                is_conflict=False,
                severity_score=0.2,
                conflict_type="none",
                prediction_time=datetime.now(),
                confidence=0.88
            )
        ]
        
        self.collector.record_conflict_detection(conflicts, datetime.now())
        
        # Add separation measurements
        base_time = datetime.now()
        separations = [8.0, 6.5, 4.5, 6.2, 7.8]
        for i, sep in enumerate(separations):
            self.collector.record_separation(
                base_time + timedelta(seconds=i*30),
                "AC001", "AC002", sep
            )
        
        # Add resolutions
        resolution = self.create_resolution_command("RES001", "AC001")
        
        issue_time = datetime.now()
        self.collector.record_resolution_issued(resolution, issue_time)
        self.collector.record_resolution_outcome("RES001", True, issue_time + timedelta(seconds=45))
    
    def create_resolution_command(self, resolution_id: str, target_aircraft: str) -> ResolutionCommand:
        """Create a minimal resolution command for testing."""
        return ResolutionCommand(
            resolution_id=resolution_id,
            target_aircraft=target_aircraft,
            resolution_type=ResolutionType.HEADING_CHANGE,
            source_engine=ResolutionEngine.HORIZONTAL,
            new_heading_deg=90.0,
            new_speed_kt=None,
            new_altitude_ft=None,
            waypoint_name=None,
            waypoint_lat=None,
            waypoint_lon=None,
            diversion_distance_nm=None,
            issue_time=datetime.now(),
            is_validated=False,
            safety_margin_nm=5.0,
            is_ownship_command=True,
            angle_within_limits=True,
            altitude_within_limits=True,
            rate_within_limits=True
        )
    
    def test_generate_summary_basic(self):
        """Test basic summary generation."""
        summary = self.collector.generate_summary()
        
        assert isinstance(summary, MetricsSummary)
        assert summary.total_cycles == 5
        assert summary.total_conflicts_detected == 2
        assert summary.true_conflicts == 1
        assert summary.false_positives == 1
        assert summary.total_resolutions_issued == 1
        assert summary.successful_resolutions == 1
        assert summary.resolution_success_rate == 1.0
        assert summary.safety_violations == 1  # One separation < 5.0
    
    def test_summary_cycle_statistics(self):
        """Test cycle time statistics in summary."""
        summary = self.collector.generate_summary()
        
        expected_avg = sum([2.1, 2.3, 1.9, 2.5, 2.0]) / 5
        assert abs(summary.avg_cycle_time_sec - expected_avg) < 0.01
    
    def test_summary_separation_statistics(self):
        """Test separation statistics in summary."""
        summary = self.collector.generate_summary()
        
        assert summary.min_separation_achieved_nm == 4.5
        expected_avg = sum([8.0, 6.5, 4.5, 6.2, 7.8]) / 5
        assert abs(summary.avg_separation_nm - expected_avg) < 0.01
    
    def test_summary_wolfgang_kpis(self):
        """Test Wolfgang KPIs in summary."""
        summary = self.collector.generate_summary()
        
        # Should include all Wolfgang KPIs
        assert hasattr(summary, 'tbas')
        assert hasattr(summary, 'lat')
        assert hasattr(summary, 'pa')
        assert hasattr(summary, 'pi')
        assert hasattr(summary, 'dat')
        assert hasattr(summary, 'dfa')
        assert hasattr(summary, 're')
        assert hasattr(summary, 'ri')
        assert hasattr(summary, 'rat')
        
        assert summary.pa == 2  # Total predicted alerts
        assert summary.pi == 1  # Predicted intrusions


class TestBaselineComparison:
    """Test baseline metrics comparison functionality."""
    
    def setup_method(self):
        """Set up collector and baseline metrics."""
        self.collector = MetricsCollector()
        
        # Add sample data to collector
        conflicts = [
            ConflictPrediction(
                ownship_id="AC001",
                intruder_id="AC002",
                time_to_cpa_min=3.0,
                distance_at_cpa_nm=2.0,
                altitude_diff_ft=500,
                is_conflict=True,
                severity_score=0.8,
                conflict_type="horizontal",
                prediction_time=datetime.now(),
                confidence=0.95
            )
        ]
        
        self.collector.record_conflict_detection(conflicts, datetime.now())
        self.collector.record_separation(datetime.now(), "AC001", "AC002", 6.5)
        
        resolution = self.create_resolution_command("RES001", "AC001")
        
        issue_time = datetime.now()
        self.collector.record_resolution_issued(resolution, issue_time)
        self.collector.record_resolution_outcome("RES001", True, issue_time + timedelta(seconds=30))
        
        # Create baseline metrics
        self.baseline = BaselineMetrics(
            baseline_conflicts_detected=10,
            baseline_false_positives=3,
            baseline_detection_latency_sec=15.0,
            baseline_resolutions_issued=8,
            baseline_success_rate=0.6,
            baseline_resolution_delay_sec=45.0,
            baseline_min_separation_nm=4.5,
            baseline_avg_separation_nm=6.0,
            baseline_safety_violations=2
        )
    
    def create_resolution_command(self, resolution_id: str, target_aircraft: str) -> ResolutionCommand:
        """Create a minimal resolution command for testing."""
        return ResolutionCommand(
            resolution_id=resolution_id,
            target_aircraft=target_aircraft,
            resolution_type=ResolutionType.HEADING_CHANGE,
            source_engine=ResolutionEngine.HORIZONTAL,
            new_heading_deg=90.0,
            new_speed_kt=None,
            new_altitude_ft=None,
            waypoint_name=None,
            waypoint_lat=None,
            waypoint_lon=None,
            diversion_distance_nm=None,
            issue_time=datetime.now(),
            is_validated=False,
            safety_margin_nm=5.0,
            is_ownship_command=True,
            angle_within_limits=True,
            altitude_within_limits=True,
            rate_within_limits=True
        )
    
    def test_compare_with_baseline(self):
        """Test comparison with baseline metrics."""
        comparison = self.collector.compare_with_baseline(self.baseline)
        
        assert isinstance(comparison, ComparisonReport)
        assert hasattr(comparison, 'detection_accuracy_improvement')
        assert hasattr(comparison, 'success_rate_improvement')
        assert hasattr(comparison, 'separation_margin_improvement_nm')
        assert hasattr(comparison, 'overall_score')
        assert hasattr(comparison, 'recommendation')
        
        # Should have improvement in success rate (1.0 vs 0.6)
        assert comparison.success_rate_improvement > 0
        
        # Overall score should be between 0 and 100
        assert 0 <= comparison.overall_score <= 100
        
        # Should have a recommendation
        assert isinstance(comparison.recommendation, str)
        assert len(comparison.recommendation) > 0


class TestReportSaving:
    """Test metrics report saving functionality."""
    
    def setup_method(self):
        """Set up collector with sample data."""
        self.collector = MetricsCollector()
        
        # Add minimal sample data
        conflict = ConflictPrediction(
            ownship_id="AC001",
            intruder_id="AC002",
            time_to_cpa_min=3.0,
            distance_at_cpa_nm=2.0,
            altitude_diff_ft=500,
            is_conflict=True,
            severity_score=0.8,
            conflict_type="horizontal",
            prediction_time=datetime.now(),
            confidence=0.95
        )
        
        self.collector.record_conflict_detection([conflict], datetime.now())
    
    def test_save_report(self):
        """Test saving metrics report to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            filepath = temp_file.name
        
        try:
            self.collector.save_report(filepath)
            
            # Verify file exists
            assert Path(filepath).exists()
            
            # Verify content
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check required fields
            assert 'conflicts_detected' in data
            assert 'conflicts_resolved' in data
            assert 'loss_of_separation_events' in data
            assert 'detection_accuracy' in data
            assert 'tbas' in data
            assert 'lat' in data
            assert 'summary' in data
            
            # Check types
            assert isinstance(data['conflicts_detected'], int)
            assert isinstance(data['detection_accuracy'], float)
            assert isinstance(data['tbas'], float)
            
        finally:
            # Clean up
            Path(filepath).unlink(missing_ok=True)


class TestSummarizeRunFunction:
    """Test the summarize_run utility function."""
    
    def test_summarize_run_complete_data(self):
        """Test summarize_run with complete input data."""
        input_stats: Dict[str, Any] = {
            "conflicts_detected": 5,
            "conflicts_resolved": 3,
            "los": 1,
            "TBAS": 0.8,
            "LAT": 2.5,
            "PA": 5.0,
            "PI": 3.0,
            "DAT": 4.2,
            "DFA": 0.6,
            "RE": 0.75,
            "RI": 0.15,
            "RAT": 1.8,
            "extra_field": "should_be_preserved"
        }
        
        result = summarize_run(input_stats)
        
        # Check required fields
        assert result["conflicts_detected"] == 5
        assert result["conflicts_resolved"] == 3
        assert result["loss_of_separation_events"] == 1
        assert result["TBAS"] == 0.8
        assert result["LAT"] == 2.5
        assert result["PA"] == 5.0
        assert result["PI"] == 3.0
        assert result["DAT"] == 4.2
        assert result["DFA"] == 0.6
        assert result["RE"] == 0.75
        assert result["RI"] == 0.15
        assert result["RAT"] == 1.8
        
        # Check extra field preserved
        assert result["extra_field"] == "should_be_preserved"
        
        # Check types
        assert isinstance(result["conflicts_detected"], int)
        assert isinstance(result["TBAS"], float)
    
    def test_summarize_run_missing_data(self):
        """Test summarize_run with missing input data."""
        input_stats: Dict[str, Any] = {
            "conflicts_detected": 2,
            # Missing most fields
            "extra_data": "preserved"
        }
        
        result = summarize_run(input_stats)
        
        # Should have defaults for missing required fields
        assert result["conflicts_detected"] == 2
        assert result["conflicts_resolved"] == 0
        assert result["loss_of_separation_events"] == 0
        assert result["TBAS"] == 0.0
        assert result["LAT"] == 0.0
        assert result["PA"] == 0.0
        assert result["PI"] == 0.0
        assert result["DAT"] == 0.0
        assert result["DFA"] == 0.0
        assert result["RE"] == 0.0
        assert result["RI"] == 0.0
        assert result["RAT"] == 0.0
        
        # Extra data preserved
        assert result["extra_data"] == "preserved"
    
    def test_summarize_run_empty_input(self):
        """Test summarize_run with empty input."""
        result = summarize_run({})
        
        # Should have all required fields with defaults
        required_fields = [
            "conflicts_detected", "conflicts_resolved", "loss_of_separation_events",
            "TBAS", "LAT", "PA", "PI", "DAT", "DFA", "RE", "RI", "RAT"
        ]
        
        for field in required_fields:
            assert field in result
            
        # Numeric fields should be 0
        assert result["conflicts_detected"] == 0
        assert result["TBAS"] == 0.0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_metrics_with_no_data(self):
        """Test metrics calculation with no data."""
        collector = MetricsCollector()
        
        # Should not crash with empty data
        summary = collector.generate_summary()
        
        assert summary.total_conflicts_detected == 0
        assert summary.total_cycles == 0
        assert summary.avg_cycle_time_sec == 0.0
        assert summary.detection_accuracy == 0.0
        assert summary.min_separation_achieved_nm == float('inf')
    
    def test_division_by_zero_protection(self):
        """Test protection against division by zero."""
        collector = MetricsCollector()
        
        # Add conflicts but no ground truth
        conflict = ConflictPrediction(
            ownship_id="AC001",
            intruder_id="AC002",
            time_to_cpa_min=3.0,
            distance_at_cpa_nm=2.0,
            altitude_diff_ft=500,
            is_conflict=True,
            severity_score=0.8,
            conflict_type="horizontal",
            prediction_time=datetime.now(),
            confidence=0.95
        )
        
        collector.record_conflict_detection([conflict], datetime.now())
        
        # Should handle division by zero gracefully
        summary = collector.generate_summary()
        assert summary.detection_accuracy >= 0.0
    
    def test_negative_separation(self):
        """Test handling of negative separation values."""
        collector = MetricsCollector()
        
        # This shouldn't happen in practice but test robustness
        collector.record_separation(datetime.now(), "AC001", "AC002", -1.0)
        
        # Should record but treat as violation
        assert len(collector.separation_history) == 1
        assert len(collector.safety_violations) == 1
    
    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        collector = MetricsCollector()
        
        collector.record_cycle_time(1e6)  # Very large cycle time
        collector.record_separation(datetime.now(), "AC001", "AC002", 1e6)  # Very large separation
        
        summary = collector.generate_summary()
        
        # Should handle without crashing
        assert summary.avg_cycle_time_sec == 1e6
        assert summary.min_separation_achieved_nm == 1e6


class TestEngineUsageTracking:
    """Test resolution engine usage tracking."""
    
    def test_engine_used_tracking(self):
        """Test tracking which engines are used for resolutions."""
        collector = MetricsCollector()
        
        # Create resolutions from different engines
        resolutions = [
            ResolutionCommand(
                resolution_id="RES001",
                target_aircraft="AC001",
                resolution_type=ResolutionType.HEADING_CHANGE,
                source_engine=ResolutionEngine.HORIZONTAL,
                new_heading_deg=90.0,
                new_speed_kt=None,
                new_altitude_ft=None,
                waypoint_name=None,
                waypoint_lat=None,
                waypoint_lon=None,
                diversion_distance_nm=None,
                issue_time=datetime.now(),
                is_validated=False,
                safety_margin_nm=5.0,
                is_ownship_command=True,
                angle_within_limits=True,
                altitude_within_limits=True,
                rate_within_limits=True
            ),
            ResolutionCommand(
                resolution_id="RES002",
                target_aircraft="AC002",
                resolution_type=ResolutionType.ALTITUDE_CHANGE,
                source_engine=ResolutionEngine.VERTICAL,
                new_heading_deg=None,
                new_speed_kt=None,
                new_altitude_ft=36000,
                waypoint_name=None,
                waypoint_lat=None,
                waypoint_lon=None,
                diversion_distance_nm=None,
                issue_time=datetime.now(),
                is_validated=False,
                safety_margin_nm=5.0,
                is_ownship_command=True,
                angle_within_limits=True,
                altitude_within_limits=True,
                rate_within_limits=True
            ),
            ResolutionCommand(
                resolution_id="RES003",
                target_aircraft="AC003",
                resolution_type=ResolutionType.SPEED_CHANGE,
                source_engine=ResolutionEngine.DETERMINISTIC,
                new_heading_deg=None,
                new_speed_kt=420,
                new_altitude_ft=None,
                waypoint_name=None,
                waypoint_lat=None,
                waypoint_lon=None,
                diversion_distance_nm=None,
                issue_time=datetime.now(),
                is_validated=False,
                safety_margin_nm=5.0,
                is_ownship_command=True,
                angle_within_limits=True,
                altitude_within_limits=True,
                rate_within_limits=True
            )
        ]
        
        # Record resolutions
        base_time = datetime.now()
        for i, resolution in enumerate(resolutions):
            collector.record_resolution_issued(resolution, base_time + timedelta(seconds=i*30))
        
        # Check engine distribution
        engine_counts: Dict[ResolutionEngine, int] = {}
        for resolution in collector.resolutions_issued:
            engine = resolution.source_engine
            engine_counts[engine] = engine_counts.get(engine, 0) + 1
        
        assert engine_counts[ResolutionEngine.HORIZONTAL] == 1
        assert engine_counts[ResolutionEngine.VERTICAL] == 1
        assert engine_counts[ResolutionEngine.DETERMINISTIC] == 1


class TestTimeToActionMetrics:
    """Test time-to-action metrics calculation."""
    
    def test_resolution_time_calculation(self):
        """Test calculation of time from detection to resolution."""
        collector = MetricsCollector()
        
        # Record conflict detection
        conflict = ConflictPrediction(
            ownship_id="AC001",
            intruder_id="AC002",
            time_to_cpa_min=3.0,
            distance_at_cpa_nm=2.0,
            altitude_diff_ft=500,
            is_conflict=True,
            severity_score=0.8,
            conflict_type="horizontal",
            prediction_time=datetime.now(),
            confidence=0.95
        )
        
        detection_time = datetime.now()
        collector.record_conflict_detection([conflict], detection_time)
        
        # Record resolution issued later
        resolution = ResolutionCommand(
            resolution_id="RES001",
            target_aircraft="AC001",
            resolution_type=ResolutionType.HEADING_CHANGE,
            source_engine=ResolutionEngine.HORIZONTAL,
            new_heading_deg=90.0,
            new_speed_kt=None,
            new_altitude_ft=None,
            waypoint_name=None,
            waypoint_lat=None,
            waypoint_lon=None,
            diversion_distance_nm=None,
            issue_time=datetime.now(),
            is_validated=False,
            safety_margin_nm=5.0,
            is_ownship_command=True,
            angle_within_limits=True,
            altitude_within_limits=True,
            rate_within_limits=True
        )
        
        resolution_time = detection_time + timedelta(seconds=45)
        collector.record_resolution_issued(resolution, resolution_time)
        
        # Record successful completion
        completion_time = resolution_time + timedelta(seconds=30)
        collector.record_resolution_outcome("RES001", True, completion_time)
        
        summary = collector.generate_summary()
        
        # Check resolution timing
        assert summary.avg_resolution_time_sec == 30.0  # Time from issue to completion
        assert summary.resolution_success_rate == 1.0


class TestAccuracyCalculation:
    """Test detection accuracy calculation."""
    
    def test_compute_accuracy_safe(self):
        """Test safe accuracy computation."""
        collector = MetricsCollector()
        
        # Test the _compute_accuracy_safe method
        accuracy = collector._compute_accuracy_safe()
        
        # With no data, should return 0.0
        assert accuracy == 0.0
        
        # Add some conflicts and test again
        conflicts = [
            ConflictPrediction(
                ownship_id="AC001",
                intruder_id="AC002",
                time_to_cpa_min=3.0,
                distance_at_cpa_nm=2.0,
                altitude_diff_ft=500,
                is_conflict=True,
                severity_score=0.8,
                conflict_type="horizontal",
                prediction_time=datetime.now(),
                confidence=0.95
            )
        ]
        
        collector.record_conflict_detection(conflicts, datetime.now())
        
        # Should compute some accuracy
        accuracy = collector._compute_accuracy_safe()
        assert 0.0 <= accuracy <= 1.0


class TestPathDeltaMetrics:
    """Test path delta calculations placeholder."""
    
    def test_path_delta_placeholder(self):
        """Placeholder test for future path delta implementation."""
        # This would test Hausdorff distance and cross-track error calculations
        # between planned and actual flight paths
        
        collector = MetricsCollector()
        
        # Future implementation would include:
        # - collector.record_planned_path(aircraft_id, waypoints)
        # - collector.record_actual_path(aircraft_id, waypoints)  
        # - collector.calculate_path_deltas()
        
        # Placeholder assertion
        assert hasattr(collector, 'generate_summary')
        
        # This test serves as a marker for future path delta implementation
        # For now, verify basic functionality exists
        summary = collector.generate_summary()
        assert isinstance(summary, MetricsSummary)
