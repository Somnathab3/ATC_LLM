"""Comprehensive tests for reporting module."""

import pytest
import tempfile
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from src.cdr.reporting import (
    FailureModeAnalysis,
    ReportPackage,
    Sprint5Reporter
)
from src.cdr.metrics import MetricsSummary
from src.cdr.simple_stress_test import StressTestResult


class TestFailureModeAnalysis:
    """Test FailureModeAnalysis dataclass."""
    
    def test_failure_mode_analysis_creation(self):
        """Test FailureModeAnalysis can be created with valid data."""
        analysis = FailureModeAnalysis(
            late_detection_rate=0.15,
            missed_conflict_rate=0.05,
            unsafe_resolution_rate=0.02,
            oscillation_rate=0.08,
            late_detection_scenarios=["scenario_1", "scenario_2"],
            missed_conflict_scenarios=["scenario_3"],
            unsafe_resolution_scenarios=["scenario_4"],
            oscillation_scenarios=["scenario_5", "scenario_6"],
            safety_impact_score=85.0,
            performance_impact_score=92.0,
            recommendations=["Improve detection timing", "Add safety buffer"]
        )
        
        assert analysis.late_detection_rate == 0.15
        assert analysis.missed_conflict_rate == 0.05
        assert analysis.safety_impact_score == 85.0
        assert len(analysis.recommendations) == 2
        assert "scenario_1" in analysis.late_detection_scenarios


class TestReportPackage:
    """Test ReportPackage dataclass."""
    
    def test_report_package_creation(self):
        """Test ReportPackage can be created with required fields."""
        failure_analysis = FailureModeAnalysis(
            late_detection_rate=0.1,
            missed_conflict_rate=0.05,
            unsafe_resolution_rate=0.02,
            oscillation_rate=0.03,
            late_detection_scenarios=[],
            missed_conflict_scenarios=[],
            unsafe_resolution_scenarios=[],
            oscillation_scenarios=[],
            safety_impact_score=90.0,
            performance_impact_score=88.0,
            recommendations=[]
        )
        
        package = ReportPackage(
            generation_timestamp=datetime.now(),
            summary_metrics={"total_conflicts": 50, "resolution_rate": 0.95},
            failure_analysis=failure_analysis,
            stress_test_results=[],
            baseline_comparison=None,
            metrics_csv_path="/path/to/metrics.csv",
            charts_directory="/path/to/charts",
            narrative_report_path="/path/to/report.txt"
        )
        
        assert package.summary_metrics["total_conflicts"] == 50
        assert package.failure_analysis.safety_impact_score == 90.0
        assert package.baseline_comparison is None


class TestSprint5Reporter:
    """Test Sprint5Reporter functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = Sprint5Reporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')
    
    def test_reporter_initialization(self):
        """Test reporter initializes correctly."""
        assert self.reporter.output_dir.exists()
        assert str(self.reporter.output_dir) == self.temp_dir
    
    def test_generate_metrics_csv_with_data(self):
        """Test CSV generation with metrics data."""
        # Create mock metrics data using correct MetricsSummary structure
        metrics_data = [
            MetricsSummary(
                total_conflicts_detected=10,
                total_resolutions_issued=8,
                successful_resolutions=7,
                safety_violations=1,
                avg_resolution_time_sec=5.0,
                min_separation_achieved_nm=4.8,
                detection_accuracy=0.9,
                false_positives=1,
                false_negatives=2,
                run_label="test_run_1"
            ),
            MetricsSummary(
                total_conflicts_detected=15,
                total_resolutions_issued=12,
                successful_resolutions=11,
                safety_violations=0,
                avg_resolution_time_sec=4.2,
                min_separation_achieved_nm=5.2,
                detection_accuracy=0.95,
                false_positives=0,
                false_negatives=3,
                run_label="test_run_2"
            )
        ]
        
        csv_path = self.reporter.generate_metrics_csv(metrics_data)
        
        # Verify file was created
        assert os.path.exists(csv_path)
        
        # Verify content
        df: Any = pd.read_csv(csv_path)
        assert len(df) == 2
        assert 'conflicts_detected' in df.columns
        assert 'conflict_resolution_efficiency' in df.columns
        assert 'safety_score' in df.columns
        
        # Check computed columns
        assert df['conflict_resolution_efficiency'].iloc[0] == 7/8
        assert df['safety_score'].iloc[1] == 100  # No safety violations
    
    def test_generate_metrics_csv_empty_data(self):
        """Test CSV generation with empty data."""
        csv_path = self.reporter.generate_metrics_csv([])
        
        # Verify file was created
        assert os.path.exists(csv_path)
        
        # Verify it has headers but no data
        df: Any = pd.read_csv(csv_path)
        assert len(df) == 0
        assert 'conflicts_detected' in df.columns
    
    def test_generate_performance_charts_with_time_series(self):
        """Test chart generation with time-series data."""
        # Create mock metrics with timestamps
        class MockMetrics:
            def __init__(self, timestamp: datetime, conflicts_detected: int = 5, conflicts_resolved: int = 4):
                self.timestamp = timestamp
                self.conflicts_detected = conflicts_detected
                self.conflicts_resolved = conflicts_resolved
                self.avg_detection_latency_sec = 2.0
        
        metrics_data = [
            MockMetrics(datetime.now()),
            MockMetrics(datetime.now()),
        ]
        
        charts_dir = self.reporter.generate_performance_charts(metrics_data)
        
        # Verify charts directory was created
        assert os.path.exists(charts_dir)
        assert os.path.isdir(charts_dir)
    
    def test_generate_performance_charts_with_summary_data(self):
        """Test chart generation with summary data (no timestamps)."""
        # Create mock metrics without timestamps
        class MockMetrics:
            def __init__(self, label: str, conflicts_detected: int = 5, total_conflicts_detected: int = 5):
                self.run_label = label
                self.conflicts_detected = conflicts_detected
                self.total_conflicts_detected = total_conflicts_detected
                self.false_negatives = 1
                self.false_positives = 0
                self.total_resolutions_issued = 4
                self.safety_violations = 0
                self.detection_accuracy = 0.9
        
        metrics_data = [
            MockMetrics("Run 1"),
            MockMetrics("Run 2"),
        ]
        
        charts_dir = self.reporter.generate_performance_charts(metrics_data)
        
        # Verify charts directory was created
        assert os.path.exists(charts_dir)
        assert os.path.isdir(charts_dir)
    
    def test_generate_performance_charts_empty_data(self):
        """Test chart generation with empty data."""
        charts_dir = self.reporter.generate_performance_charts([])
        
        # Should still create charts directory
        assert os.path.exists(charts_dir)
        assert os.path.isdir(charts_dir)
    
    def test_plot_summary_bars_functionality(self):
        """Test _plot_summary_bars method directly."""
        # Create mock metrics
        class MockMetrics:
            def __init__(self, label: str):
                self.run_label = label
                self.total_conflicts_detected = 10
                self.false_negatives = 2
                self.false_positives = 1
                self.total_resolutions_issued = 8
                self.safety_violations = 0
                self.detection_accuracy = 0.85
        
        metrics_list = [MockMetrics("Test Run")]
        charts_dir = os.path.join(self.temp_dir, "test_charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Access protected method for testing
        reporter = self.reporter
        getattr(reporter, '_plot_summary_bars')(metrics_list, charts_dir)
        
        # Check that chart files were created
        assert os.path.exists(os.path.join(charts_dir, "summary_counts.png"))
        assert os.path.exists(os.path.join(charts_dir, "detection_accuracy.png"))
    
    def test_plot_summary_bars_empty_data(self):
        """Test _plot_summary_bars with empty data."""
        charts_dir = os.path.join(self.temp_dir, "test_charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Should handle empty data gracefully
        reporter = self.reporter
        getattr(reporter, '_plot_summary_bars')([], charts_dir)
        
        # No charts should be created for empty data
        chart_files = os.listdir(charts_dir)
        assert len(chart_files) == 0
    
    @patch('matplotlib.pyplot.savefig')
    def test_chart_generation_with_matplotlib_mock(self, mock_savefig: Mock):
        """Test chart generation with mocked matplotlib to avoid file I/O."""
        mock_savefig.return_value = None
        
        metrics_data = [
            MetricsSummary(
                total_conflicts_detected=5,
                total_resolutions_issued=4,
                successful_resolutions=3,
                safety_violations=0,
                avg_resolution_time_sec=3.0,
                min_separation_achieved_nm=5.5,
                detection_accuracy=0.95,
                false_positives=0,
                false_negatives=1
            )
        ]
        
        charts_dir = self.reporter.generate_performance_charts(metrics_data)
        
        # Verify matplotlib savefig was called
        assert mock_savefig.called
        assert os.path.exists(charts_dir)


class TestReportingIntegration:
    """Integration tests for reporting module."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')
    
    def test_full_reporting_workflow(self):
        """Test complete reporting workflow."""
        reporter = Sprint5Reporter(output_dir=self.temp_dir)
        
        # Create comprehensive test data
        metrics_data = [
            MetricsSummary(
                total_conflicts_detected=20,
                total_resolutions_issued=18,
                successful_resolutions=16,
                safety_violations=2,
                avg_resolution_time_sec=6.0,
                min_separation_achieved_nm=4.5,
                detection_accuracy=0.88,
                false_positives=2,
                false_negatives=4
            ),
            MetricsSummary(
                total_conflicts_detected=25,
                total_resolutions_issued=22,
                successful_resolutions=20,
                safety_violations=1,
                avg_resolution_time_sec=5.5,
                min_separation_achieved_nm=4.8,
                detection_accuracy=0.91,
                false_positives=1,
                false_negatives=3
            )
        ]
        
        # Generate CSV
        csv_path = reporter.generate_metrics_csv(metrics_data)
        assert os.path.exists(csv_path)
        
        # Generate charts
        charts_dir = reporter.generate_performance_charts(metrics_data)
        assert os.path.exists(charts_dir)
        
        # Verify CSV content
        df: Any = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df['conflicts_detected'].sum() == 45
        assert all(df['safety_score'] >= 0)
        assert all(df['conflict_resolution_efficiency'] >= 0)
    
    def test_error_handling_in_reporting(self):
        """Test error handling in reporting functions."""
        reporter = Sprint5Reporter(output_dir=self.temp_dir)
        
        # Test with malformed data
        invalid_metrics = [None, "invalid", 123]
        
        # Should handle gracefully without crashing
        try:
            csv_path = reporter.generate_metrics_csv(invalid_metrics)
            charts_dir = reporter.generate_performance_charts(invalid_metrics)
            # If we get here, error handling worked
            assert True
        except (TypeError, AttributeError, ValueError) as e:
            # If an exception occurs, it should be a specific, handled one
            assert "dataclass" in str(e).lower() or "expected" in str(e).lower() or "invalid" in str(e).lower()


class TestReportingEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')
    
    def test_reporter_with_readonly_directory(self):
        """Test reporter behavior with read-only output directory."""
        # Create reporter in temp directory first
        reporter = Sprint5Reporter(output_dir=self.temp_dir)
        
        # Try to make directory read-only (may not work on all systems)
        try:
            os.chmod(self.temp_dir, 0o444)
            
            # Attempt operations - should handle permission errors gracefully
            metrics_data = [
                MetricsSummary(
                    total_conflicts_detected=5,
                    total_resolutions_issued=4,
                    successful_resolutions=3,
                    safety_violations=0,
                    avg_resolution_time_sec=2.0,
                    min_separation_achieved_nm=5.0,
                    detection_accuracy=0.9,
                    false_positives=0,
                    false_negatives=1
                )
            ]
            
            # These might fail due to permissions, but shouldn't crash
            try:
                csv_path = reporter.generate_metrics_csv(metrics_data)
            except PermissionError:
                pass  # Expected on some systems
            
        finally:
            # Restore permissions for cleanup
            os.chmod(self.temp_dir, 0o755)
    
    def test_metrics_with_missing_attributes(self):
        """Test handling of metrics objects with missing attributes."""
        reporter = Sprint5Reporter(output_dir=self.temp_dir)
        
        # Create metrics with minimal attributes
        class MinimalMetrics:
            def __init__(self):
                self.conflicts_detected = 5
                # Missing many other attributes
        
        metrics_data = [MinimalMetrics()]
        
        # Should handle missing attributes gracefully  
        try:
            csv_path = reporter.generate_metrics_csv(metrics_data)
            charts_dir = reporter.generate_performance_charts(metrics_data)
            
            assert os.path.exists(csv_path)
            assert os.path.exists(charts_dir)
        except (TypeError, AttributeError) as e:
            # Expected error for non-dataclass objects
            assert "dataclass" in str(e).lower()
    
    def test_large_dataset_handling(self):
        """Test reporter with large datasets."""
        reporter = Sprint5Reporter(output_dir=self.temp_dir)
        
        # Create large dataset
        large_metrics_data = []
        for i in range(100):
            large_metrics_data.append(
                MetricsSummary(
                    total_conflicts_detected=i % 20 + 1,
                    total_resolutions_issued=i % 18 + 1,
                    successful_resolutions=i % 16 + 1,
                    safety_violations=i % 3,
                    avg_resolution_time_sec=2.0 + (i % 15) * 0.1,
                    min_separation_achieved_nm=4.5 + (i % 5) * 0.1,
                    detection_accuracy=0.8 + (i % 20) * 0.01,
                    false_positives=i % 5,
                    false_negatives=i % 7
                )
            )
        
        # Should handle large datasets without issues
        csv_path = reporter.generate_metrics_csv(large_metrics_data)
        charts_dir = reporter.generate_performance_charts(large_metrics_data)
        
        assert os.path.exists(csv_path)
        assert os.path.exists(charts_dir)
        
        # Verify CSV content
        df: Any = pd.read_csv(csv_path)
        assert len(df) == 100
