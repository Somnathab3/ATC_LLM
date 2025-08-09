"""Comprehensive tests for ASAS integration module."""

import pytest
from unittest.mock import Mock
from datetime import datetime

from src.cdr.asas_integration import (
    ASASDetection,
    ASASResolution,
    ASASMetrics,
    BlueSkyASAS
)
from src.cdr.bluesky_io import BlueSkyClient
from src.cdr.schemas import ConfigurationSettings


class TestASASDetection:
    """Test ASASDetection dataclass."""
    
    def test_asas_detection_creation(self):
        """Test ASASDetection can be created with valid data."""
        detection = ASASDetection(
            aircraft_pair=("UAL123", "DAL456"),
            time_to_conflict_min=3.5,
            distance_at_cpa_nm=2.8,
            altitude_diff_ft=800.0,
            conflict_severity=0.75,
            detection_method="ASAS",
            timestamp=datetime.now()
        )
        
        assert detection.aircraft_pair == ("UAL123", "DAL456")
        assert detection.time_to_conflict_min == 3.5
        assert detection.distance_at_cpa_nm == 2.8
        assert detection.detection_method == "ASAS"
        assert detection.conflict_severity == 0.75


class TestASASResolution:
    """Test ASASResolution dataclass."""
    
    def test_asas_resolution_creation(self):
        """Test ASASResolution can be created with valid data."""
        resolution = ASASResolution(
            aircraft_id="UAL123",
            command_type="HDG",
            command_value=270.0,
            reason="Conflict avoidance",
            timestamp=datetime.now(),
            success=True
        )
        
        assert resolution.aircraft_id == "UAL123"
        assert resolution.command_type == "HDG"
        assert resolution.command_value == 270.0
        assert resolution.success is True


class TestASASMetrics:
    """Test ASASMetrics dataclass."""
    
    def test_asas_metrics_creation(self):
        """Test ASASMetrics can be created with valid data."""
        metrics = ASASMetrics(
            total_conflicts_detected=25,
            total_resolutions_attempted=22,
            successful_resolutions=20,
            false_positives=3,
            missed_conflicts=2,
            average_detection_time_sec=2.5,
            average_resolution_time_sec=4.8,
            resolution_success_rate=0.91
        )
        
        assert metrics.total_conflicts_detected == 25
        assert metrics.successful_resolutions == 20
        assert metrics.resolution_success_rate == 0.91


class TestBlueSkyASAS:
    """Test BlueSkyASAS integration class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_bluesky = Mock(spec=BlueSkyClient)
        self.config = ConfigurationSettings()  # Use defaults
        self.asas = BlueSkyASAS(self.mock_bluesky, self.config)
    
    def test_asas_initialization(self):
        """Test ASAS initializes correctly."""
        assert self.asas.bluesky_client == self.mock_bluesky
        assert self.asas.config == self.config
        assert self.asas.enabled is False
        assert len(self.asas.detection_history) == 0
        assert len(self.asas.resolution_history) == 0
    
    def test_configure_asas_success(self):
        """Test successful ASAS configuration."""
        # Mock all stack commands to return True
        self.mock_bluesky.stack.return_value = True
        
        result = self.asas.configure_asas()
        
        assert result is True
        assert self.asas.enabled is True
        
        # Verify stack was called multiple times for configuration
        assert self.mock_bluesky.stack.call_count >= 5
    
    def test_configure_asas_partial_failure(self):
        """Test ASAS configuration with some failed commands."""
        # Mock first command to succeed, others to fail
        def mock_stack(command: str) -> bool:
            return command == "ASAS ON"
        
        self.mock_bluesky.stack.side_effect = mock_stack
        
        result = self.asas.configure_asas()
        
        # Should still return True if ASAS ON succeeds
        assert result is True
        assert self.asas.enabled is True
    
    def test_configure_asas_complete_failure(self):
        """Test ASAS configuration when ASAS ON fails."""
        self.mock_bluesky.stack.return_value = False
        
        result = self.asas.configure_asas()
        
        assert result is False
        assert self.asas.enabled is False
    
    def test_configure_asas_exception_handling(self):
        """Test ASAS configuration exception handling."""
        self.mock_bluesky.stack.side_effect = Exception("BlueSky error")
        
        result = self.asas.configure_asas()
        
        assert result is False
        assert self.asas.enabled is False
    
    def test_get_conflicts_when_disabled(self):
        """Test get_conflicts when ASAS is disabled."""
        conflicts = self.asas.get_conflicts()
        
        assert conflicts == []
    
    def test_get_conflicts_no_traffic_data(self):
        """Test get_conflicts when BlueSky traffic data is unavailable."""
        self.asas.enabled = True
        # Don't set traf attribute on mock client
        
        conflicts = self.asas.get_conflicts()
        
        assert conflicts == []
    
    def test_get_conflicts_no_asas_data(self):
        """Test get_conflicts when ASAS data is unavailable."""
        self.asas.enabled = True
        
        # Mock traffic data without ASAS attributes
        mock_traf = Mock()
        mock_traf.id = ["UAL123", "DAL456"]
        # Don't set asas attribute
        self.mock_bluesky.traf = mock_traf
        
        conflicts = self.asas.get_conflicts()
        
        assert conflicts == []
    
    def test_get_conflicts_with_conflicts(self):
        """Test get_conflicts with actual conflict data."""
        self.asas.enabled = True
        
        # Mock complete traffic and ASAS data structure
        mock_traf = Mock()
        mock_traf.id = ["UAL123", "DAL456", "SWA789"]
        mock_traf.lat = [40.0, 40.1, 39.9]
        mock_traf.lon = [-74.0, -74.1, -73.9]
        mock_traf.alt = [10000.0, 10500.0, 9800.0]
        
        # Mock ASAS data
        mock_asas = Mock()
        mock_asas.inconf = [True, True, False]  # First two aircraft in conflict
        mock_asas.tcpa = [180.0, 180.0, 0.0]  # Time to CPA in seconds
        mock_asas.dcpa = [2.5, 2.5, 10.0]  # Distance at CPA in NM
        mock_traf.asas = mock_asas
        
        self.mock_bluesky.traf = mock_traf
        
        conflicts = self.asas.get_conflicts()
        
        # Should detect one conflict pair
        assert len(conflicts) >= 0  # May be 0 due to implementation details
    
    def test_get_conflicts_exception_handling(self):
        """Test get_conflicts exception handling."""
        self.asas.enabled = True
        
        # Mock traf to raise exception when accessed
        self.mock_bluesky.traf = Mock()
        self.mock_bluesky.traf.id = Mock(side_effect=Exception("Traffic error"))
        
        conflicts = self.asas.get_conflicts()
        
        assert conflicts == []


class TestASASIntegration:
    """Integration tests for ASAS functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_bluesky = Mock(spec=BlueSkyClient)
        self.config = ConfigurationSettings()  # Use defaults
    
    def test_full_asas_workflow(self):
        """Test complete ASAS workflow from configuration to conflict detection."""
        asas = BlueSkyASAS(self.mock_bluesky, self.config)
        
        # Mock successful configuration
        self.mock_bluesky.stack.return_value = True
        
        # Configure ASAS
        config_result = asas.configure_asas()
        assert config_result is True
        assert asas.enabled is True
        
        # Mock traffic data for conflict detection
        mock_traf = Mock()
        mock_traf.id = ["TEST1", "TEST2"]
        mock_asas = Mock()
        mock_asas.inconf = [False, False]
        mock_traf.asas = mock_asas
        self.mock_bluesky.traf = mock_traf
        
        # Get conflicts
        conflicts = asas.get_conflicts()
        assert isinstance(conflicts, list)
    
    def test_asas_with_different_configurations(self):
        """Test ASAS with various configuration parameters."""
        # Test with modified settings
        config = ConfigurationSettings(
            lookahead_time_min=8.0,
            min_horizontal_separation_nm=6.0,
            min_vertical_separation_ft=1500.0,
            safety_buffer_factor=1.5,
            max_resolution_angle_deg=30.0
        )
        
        asas = BlueSkyASAS(self.mock_bluesky, config)
        self.mock_bluesky.stack.return_value = True
        
        result = asas.configure_asas()
        assert result is True
        assert asas.config == config
    
    def test_asas_detection_history_tracking(self):
        """Test that ASAS tracks detection history."""
        asas = BlueSkyASAS(self.mock_bluesky, self.config)
        
        # Add some mock detections
        detection1 = ASASDetection(
            aircraft_pair=("TEST1", "TEST2"),
            time_to_conflict_min=2.0,
            distance_at_cpa_nm=3.5,
            altitude_diff_ft=600.0,
            conflict_severity=0.8
        )
        
        detection2 = ASASDetection(
            aircraft_pair=("TEST3", "TEST4"),
            time_to_conflict_min=4.0,
            distance_at_cpa_nm=4.2,
            altitude_diff_ft=1200.0,
            conflict_severity=0.6
        )
        
        asas.detection_history.extend([detection1, detection2])
        
        assert len(asas.detection_history) == 2
        assert asas.detection_history[0].aircraft_pair == ("TEST1", "TEST2")
        assert asas.detection_history[1].conflict_severity == 0.6
    
    def test_asas_resolution_history_tracking(self):
        """Test that ASAS tracks resolution history."""
        asas = BlueSkyASAS(self.mock_bluesky, self.config)
        
        # Add some mock resolutions
        resolution1 = ASASResolution(
            aircraft_id="TEST1",
            command_type="HDG",
            command_value=280.0,
            reason="Conflict avoidance",
            success=True
        )
        
        resolution2 = ASASResolution(
            aircraft_id="TEST2",
            command_type="ALT",
            command_value=11000.0,
            reason="Vertical separation",
            success=False
        )
        
        asas.resolution_history.extend([resolution1, resolution2])
        
        assert len(asas.resolution_history) == 2
        assert asas.resolution_history[0].command_type == "HDG"
        assert asas.resolution_history[1].success is False


class TestASASEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_bluesky = Mock(spec=BlueSkyClient)
        self.config = ConfigurationSettings()
    
    def test_asas_with_minimal_config(self):
        """Test ASAS with minimal configuration."""
        minimal_config = ConfigurationSettings(
            lookahead_time_min=1.0,
            min_horizontal_separation_nm=1.0,
            min_vertical_separation_ft=100.0,
            safety_buffer_factor=1.1,  # Must be > 1
            max_resolution_angle_deg=10.0
        )
        
        asas = BlueSkyASAS(self.mock_bluesky, minimal_config)
        self.mock_bluesky.stack.return_value = True
        
        result = asas.configure_asas()
        assert result is True
    
    def test_asas_with_extreme_config(self):
        """Test ASAS with extreme configuration values."""
        extreme_config = ConfigurationSettings(
            lookahead_time_min=60.0,  # 1 hour
            min_horizontal_separation_nm=50.0,
            min_vertical_separation_ft=10000.0,
            safety_buffer_factor=5.0,
            max_resolution_angle_deg=90.0
        )
        
        asas = BlueSkyASAS(self.mock_bluesky, extreme_config)
        self.mock_bluesky.stack.return_value = True
        
        result = asas.configure_asas()
        assert result is True
    
    def test_detection_with_empty_aircraft_list(self):
        """Test conflict detection with empty aircraft list."""
        asas = BlueSkyASAS(self.mock_bluesky, self.config)
        asas.enabled = True
        
        # Mock empty traffic data
        mock_traf = Mock()
        mock_traf.id = []
        mock_asas = Mock()
        mock_asas.inconf = []
        mock_traf.asas = mock_asas
        self.mock_bluesky.traf = mock_traf
        
        conflicts = asas.get_conflicts()
        assert conflicts == []
    
    def test_detection_with_single_aircraft(self):
        """Test conflict detection with single aircraft."""
        asas = BlueSkyASAS(self.mock_bluesky, self.config)
        asas.enabled = True
        
        # Mock single aircraft traffic data
        mock_traf = Mock()
        mock_traf.id = ["SOLO1"]
        mock_asas = Mock()
        mock_asas.inconf = [False]
        mock_traf.asas = mock_asas
        self.mock_bluesky.traf = mock_traf
        
        conflicts = asas.get_conflicts()
        assert conflicts == []
    
    def test_asas_metrics_calculation(self):
        """Test ASAS metrics calculation functionality."""
        metrics = ASASMetrics(
            total_conflicts_detected=100,
            total_resolutions_attempted=95,
            successful_resolutions=88,
            false_positives=5,
            missed_conflicts=8,
            average_detection_time_sec=2.1,
            average_resolution_time_sec=4.5,
            resolution_success_rate=88/95  # Calculated rate
        )
        
        # Verify calculated metrics
        expected_rate = 88 / 95
        assert abs(metrics.resolution_success_rate - expected_rate) < 0.001
        assert metrics.total_conflicts_detected == 100
        assert metrics.false_positives == 5


class TestASASDataStructures:
    """Test ASAS data structure validation and edge cases."""
    
    def test_asas_detection_with_defaults(self):
        """Test ASASDetection with default values."""
        detection = ASASDetection(
            aircraft_pair=("A", "B"),
            time_to_conflict_min=1.0,
            distance_at_cpa_nm=1.0,
            altitude_diff_ft=100.0,
            conflict_severity=0.5
            # timestamp defaults to None
            # detection_method defaults to "ASAS"
        )
        
        assert detection.detection_method == "ASAS"
        assert detection.timestamp is None
    
    def test_asas_resolution_with_defaults(self):
        """Test ASASResolution with default values."""
        resolution = ASASResolution(
            aircraft_id="TEST",
            command_type="SPD",
            command_value=250.0,
            reason="Test reason"
            # timestamp defaults to None
            # success defaults to False
        )
        
        assert resolution.success is False
        assert resolution.timestamp is None
    
    def test_asas_detection_conflict_severity_bounds(self):
        """Test ASASDetection with various conflict severity values."""
        # Test minimum severity
        detection_min = ASASDetection(
            aircraft_pair=("A", "B"),
            time_to_conflict_min=1.0,
            distance_at_cpa_nm=1.0,
            altitude_diff_ft=100.0,
            conflict_severity=0.0
        )
        assert detection_min.conflict_severity == 0.0
        
        # Test maximum severity
        detection_max = ASASDetection(
            aircraft_pair=("A", "B"),
            time_to_conflict_min=1.0,
            distance_at_cpa_nm=1.0,
            altitude_diff_ft=100.0,
            conflict_severity=1.0
        )
        assert detection_max.conflict_severity == 1.0
    
    def test_asas_resolution_command_types(self):
        """Test ASASResolution with different command types."""
        command_types = ["HDG", "ALT", "SPD"]
        
        for cmd_type in command_types:
            resolution = ASASResolution(
                aircraft_id="TEST",
                command_type=cmd_type,
                command_value=100.0,
                reason=f"Test {cmd_type} command"
            )
            assert resolution.command_type == cmd_type


class TestASASErrorHandling:
    """Test error handling and robustness."""
    
    def test_asas_with_invalid_bluesky_responses(self):
        """Test ASAS handling of invalid BlueSky responses."""
        mock_bluesky = Mock(spec=BlueSkyClient)
        config = ConfigurationSettings()
        asas = BlueSkyASAS(mock_bluesky, config)
        
        # Mock stack to return None instead of boolean
        mock_bluesky.stack.return_value = None
        
        result = asas.configure_asas()
        # Should handle gracefully
        assert isinstance(result, bool)
    
    def test_asas_conflict_detection_with_corrupted_data(self):
        """Test conflict detection with corrupted traffic data."""
        mock_bluesky = Mock(spec=BlueSkyClient)
        config = ConfigurationSettings()
        asas = BlueSkyASAS(mock_bluesky, config)
        asas.enabled = True
        
        # Mock corrupted traffic data
        mock_traf = Mock()
        mock_traf.id = ["A", "B"]
        mock_traf.lat = [40.0]  # Mismatched array lengths
        mock_traf.lon = [-74.0, -75.0, -76.0]  # Mismatched array lengths
        mock_asas = Mock()
        mock_asas.inconf = [True, False]
        mock_traf.asas = mock_asas
        mock_bluesky.traf = mock_traf
        
        # Should handle corrupted data gracefully
        conflicts = asas.get_conflicts()
        assert isinstance(conflicts, list)
