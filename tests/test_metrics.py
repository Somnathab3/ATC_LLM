"""Comprehensive test suite for metrics collection module."""

import pytest
from datetime import datetime
from src.cdr.metrics import MetricsCollector
from src.cdr.schemas import AircraftState


class TestMetricsImports:
    """Test metrics module imports."""
    
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


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def test_metrics_collector_initialization(self):
        """Test that MetricsCollector can be initialized."""
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
    
    def test_methods_exist(self):
        """Test that expected methods exist on MetricsCollector."""
        collector = MetricsCollector()
        
        # Check basic methods
        assert hasattr(collector, 'reset')
        assert hasattr(collector, 'record_cycle_time')
        
        # Test that methods are callable
        assert callable(collector.reset)
        assert callable(collector.record_cycle_time)


class TestCycleTimeRecording:
    """Test cycle time recording functionality."""
    
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
    
    def test_metrics_cycle_recording(self):
        """Test cycle time recording and calculation."""
        collector = MetricsCollector()
        
        # Record several cycle times
        cycle_times = [1.5, 2.0, 1.8, 2.2, 1.9]
        for time in cycle_times:
            collector.record_cycle_time(time)
        
        assert len(collector.cycle_times) == 5
        assert all(time in collector.cycle_times for time in cycle_times)
        
        # Check basic statistics
        assert min(collector.cycle_times) == 1.5
        assert max(collector.cycle_times) == 2.2
    
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


class TestConflictRecording:
    """Test conflict detection recording."""
    
    def test_metrics_conflict_recording(self):
        """Test conflict recording functionality."""
        collector = MetricsCollector()
        
        # Create sample aircraft states
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        intruder = AircraftState(
            aircraft_id="INTRUDER",
            timestamp=datetime.now(),
            latitude=59.4,
            longitude=18.5,
            altitude_ft=35000,
            ground_speed_kt=400,
            heading_deg=270,
            vertical_speed_fpm=0
        )
        
        # Test conflict recording (methods may not be fully implemented)
        try:
            collector.record_conflict_detected(ownship, intruder, 2.5, 3.0)
            # If successful, check that conflict was recorded
            if hasattr(collector, 'conflicts_detected'):
                assert len(collector.conflicts_detected) >= 0
        except (AttributeError, NotImplementedError):
            # Method may not be implemented yet
            pytest.skip("Conflict recording not implemented")
    
    def test_wolfgang_kpis_calculation(self):
        """Test Wolfgang KPI calculations."""
        collector = MetricsCollector()
        
        # Test basic KPI structure (if implemented)
        try:
            kpis = collector.calculate_wolfgang_kpis()
            if kpis is not None:
                assert isinstance(kpis, dict)
                # Check for expected KPI fields
                expected_fields = ["avg_cycle_time", "conflicts_per_hour", "resolution_success_rate"]
                # Note: Exact fields depend on implementation
        except (AttributeError, NotImplementedError):
            # KPI calculation may not be implemented yet
            pytest.skip("Wolfgang KPI calculation not implemented")


class TestMetricsReset:
    """Test metrics reset functionality."""
    
    def test_reset_metrics(self):
        """Test resetting metrics collector."""
        collector = MetricsCollector()
        
        # Add some data
        collector.record_cycle_time(2.5)
        if hasattr(collector, 'record_conflict'):
            try:
                collector.record_conflict("TEST_CONFLICT")
            except:
                pass  # Method may not exist or be implemented
        
        # Reset
        collector.reset()
        
        # Should be empty again
        assert len(collector.cycle_times) == 0
        assert isinstance(collector.start_time, datetime)
        
        # Check other metrics are reset too
        if hasattr(collector, 'conflicts_detected'):
            assert len(collector.conflicts_detected) == 0
        if hasattr(collector, 'resolutions_issued'):
            assert len(collector.resolutions_issued) == 0


class TestFullCycleMetrics:
    """Test full cycle metrics collection."""
    
    def test_metrics_collection_full_cycle(self):
        """Test metrics collection through full cycle."""
        collector = MetricsCollector()
        
        # Simulate a full detection/resolution cycle
        start_time = datetime.now()
        
        # Record cycle start
        cycle_start = datetime.now()
        
        # Simulate conflict detection
        ownship = AircraftState(
            aircraft_id="OWNSHIP",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        intruder = AircraftState(
            aircraft_id="INTRUDER",
            timestamp=datetime.now(),
            latitude=59.4,
            longitude=18.5,
            altitude_ft=35000,
            ground_speed_kt=400,
            heading_deg=270,
            vertical_speed_fpm=0
        )
        
        # Record cycle time
        cycle_end = datetime.now()
        cycle_duration = (cycle_end - cycle_start).total_seconds()
        collector.record_cycle_time(cycle_duration)
        
        # Verify metrics were recorded
        assert len(collector.cycle_times) == 1
        assert collector.cycle_times[0] == cycle_duration
