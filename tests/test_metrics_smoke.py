"""Smoke tests for metrics collection module."""

import pytest
from datetime import datetime
from src.cdr.metrics import MetricsCollector


class TestMetricsSmoke:
    """Smoke tests for metrics.py module."""
    
    def test_metrics_collector_initialization(self):
        """Test that MetricsCollector can be initialized."""
        collector = MetricsCollector()
        
        assert collector is not None
        assert hasattr(collector, 'start_time')
        assert hasattr(collector, 'cycle_times')
        assert hasattr(collector, 'conflicts_detected')
        assert hasattr(collector, 'resolutions_issued')
    
    def test_record_cycle_time(self):
        """Test recording cycle execution times."""
        collector = MetricsCollector()
        
        # Record some cycle times
        collector.record_cycle_time(2.5)
        collector.record_cycle_time(3.1)
        collector.record_cycle_time(2.8)
        
        assert len(collector.cycle_times) == 3
        assert 2.5 in collector.cycle_times
        assert 3.1 in collector.cycle_times
        assert 2.8 in collector.cycle_times
    
    def test_reset_metrics(self):
        """Test resetting metrics collector."""
        collector = MetricsCollector()
        
        # Add some data
        collector.record_cycle_time(2.5)
        
        # Reset
        collector.reset()
        
        # Should be empty again
        assert len(collector.cycle_times) == 0
        assert isinstance(collector.start_time, datetime)
    
    def test_module_imports(self):
        """Test that key metrics classes can be imported."""
        try:
            from src.cdr.metrics import MetricsCollector
            assert MetricsCollector is not None
        except ImportError:
            pytest.fail("Cannot import MetricsCollector")
        
        try:
            from src.cdr.metrics import MetricsSummary
            assert MetricsSummary is not None
        except ImportError:
            pytest.skip("MetricsSummary not available")
    
    def test_basic_functionality(self):
        """Test basic metrics functionality."""
        collector = MetricsCollector()
        
        # Test basic attributes exist
        assert hasattr(collector, 'cycle_times')
        assert hasattr(collector, 'conflicts_detected')
        assert hasattr(collector, 'resolutions_issued')
        assert hasattr(collector, 'start_time')
        
        # Test basic list operations work
        assert len(collector.cycle_times) == 0
        assert len(collector.conflicts_detected) == 0
        assert len(collector.resolutions_issued) == 0
    
    def test_performance_calculation(self):
        """Test basic performance calculation."""
        collector = MetricsCollector()
        
        # Add some sample data
        collector.record_cycle_time(2.0)
        collector.record_cycle_time(3.0)
        collector.record_cycle_time(2.5)
        
        # Calculate basic stats
        assert len(collector.cycle_times) == 3
        assert min(collector.cycle_times) == 2.0
        assert max(collector.cycle_times) == 3.0
        
        # Calculate average
        avg_time = sum(collector.cycle_times) / len(collector.cycle_times)
        assert abs(avg_time - 2.5) < 0.01
    
    def test_methods_exist(self):
        """Test that expected methods exist on MetricsCollector."""
        collector = MetricsCollector()
        
        # Check basic methods
        assert hasattr(collector, 'reset')
        assert hasattr(collector, 'record_cycle_time')
        
        # Test that methods are callable
        assert callable(collector.reset)
        assert callable(collector.record_cycle_time)
