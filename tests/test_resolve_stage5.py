"""Comprehensive test suite for resolve.py - Stage 5 coverage target ≥75%.

Tests cover:
- Turn: absolute heading, delta changes, clamping over limits
- Altitude: absolute/delta changes with service ceiling/floor clamping  
- Waypoint: valid/invalid/too-far scenarios
- Ownship-only enforcement
- BlueSky command application
- Safety validation logic
- Oscillation guards
- Fallback resolution
"""

import pytest
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.cdr.resolve import (
    execute_resolution,
    apply_resolution,
    format_resolution_command,
    MAX_HEADING_CHANGE_DEG,
    MIN_ALTITUDE_CHANGE_FT,
    MAX_ALTITUDE_CHANGE_FT,
    MIN_SAFE_SEPARATION_NM,
    MIN_SAFE_SEPARATION_FT,
    to_bluesky_command_heading,
    to_bluesky_command_altitude
)
from src.cdr.schemas import (
    ResolveOut, ResolutionCommand, ResolutionType, ResolutionEngine,
    AircraftState, ConflictPrediction
)


@pytest.fixture
def ownship():
    """Standard ownship aircraft state."""
    return AircraftState(
        aircraft_id="OWNSHIP",
        timestamp=datetime.now(),
        latitude=59.3293,
        longitude=18.0686,
        altitude_ft=35000,
        ground_speed_kt=450,
        heading_deg=90,  # East
        vertical_speed_fpm=0,
        aircraft_type="B737",
        spawn_offset_min=0.0
    )


@pytest.fixture
def intruder():
    """Standard intruder aircraft state."""
    return AircraftState(
        aircraft_id="INTRUDER",
        timestamp=datetime.now(),
        latitude=59.3393,  # Slightly north
        longitude=18.0786,  # Slightly east
        altitude_ft=35000,  # Same altitude - horizontal conflict
        ground_speed_kt=420,
        heading_deg=270,  # West - head-on
        vertical_speed_fpm=0,
        aircraft_type="A320",
        spawn_offset_min=0.0
    )


@pytest.fixture
def conflict():
    """Standard conflict prediction."""
    return ConflictPrediction(
        ownship_id="OWNSHIP",
        intruder_id="INTRUDER",
        time_to_cpa_min=3.0,
        distance_at_cpa_nm=2.0,  # Below 5 NM threshold
        altitude_diff_ft=0.0,
        is_conflict=True,
        severity_score=0.8,
        conflict_type="horizontal",
        prediction_time=datetime.now(),
        confidence=0.9
    )


@pytest.fixture
def clear_command_history():
    """Clear global command history before each test."""
    from src.cdr.resolve import _command_history
    _command_history.clear()
    yield
    _command_history.clear()


class TestTurnResolution:
    """Test turn/heading change resolution logic."""
    
    def test_turn_absolute_heading(self, ownship, intruder, conflict, clear_command_history):
        """Test turn with absolute heading specification."""
        llm_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 120},
            rationale="Turn right to avoid conflict"
        )
        
        # Mock the safety validation to return True so we get the LLM resolution
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.resolution_type == ResolutionType.HEADING_CHANGE
        assert result.new_heading_deg == 120
        assert result.target_aircraft == "OWNSHIP"
        assert result.is_ownship_command is True
    
    def test_turn_delta_via_direction_and_degrees(self, ownship, intruder, conflict, clear_command_history):
        """Test turn with direction and degrees parameters."""
        llm_resolution = ResolveOut(
            action="turn",
            params={"direction": "right", "degrees": 25},
            rationale="Turn right 25 degrees"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.resolution_type == ResolutionType.HEADING_CHANGE
        expected_heading = (90 + 25) % 360  # 115
        assert result.new_heading_deg == expected_heading
    
    def test_turn_delta_left(self, ownship, intruder, conflict, clear_command_history):
        """Test left turn with delta degrees."""
        llm_resolution = ResolveOut(
            action="turn",
            params={"direction": "left", "degrees": 20},
            rationale="Turn left 20 degrees"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        expected_heading = (90 - 20) % 360  # 70
        assert result.new_heading_deg == expected_heading
    
    def test_turn_clamp_over_limit_positive(self, ownship, intruder, conflict, clear_command_history):
        """Test clamping when turn exceeds maximum allowed change (positive)."""
        # Request 45-degree turn, should be clamped to MAX_HEADING_CHANGE_DEG (30)
        llm_resolution = ResolveOut(
            action="turn",
            params={"direction": "right", "degrees": 45},
            rationale="Big right turn"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        expected_heading = (90 + MAX_HEADING_CHANGE_DEG) % 360  # 120
        assert result.new_heading_deg == expected_heading
    
    def test_turn_clamp_over_limit_negative(self, ownship, intruder, conflict, clear_command_history):
        """Test clamping when turn exceeds maximum allowed change (negative)."""
        # Request 50-degree left turn, should be clamped to MAX_HEADING_CHANGE_DEG (30)
        llm_resolution = ResolveOut(
            action="turn",
            params={"direction": "left", "degrees": 50},
            rationale="Big left turn"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        expected_heading = (90 - MAX_HEADING_CHANGE_DEG) % 360  # 60
        assert result.new_heading_deg == expected_heading
    
    def test_turn_default_behavior(self, ownship, intruder, conflict, clear_command_history):
        """Test default turn behavior when no specific parameters given."""
        llm_resolution = ResolveOut(
            action="turn",
            params={},  # No specific parameters
            rationale="Generic turn"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        # Should default to right turn of 30 degrees
        expected_heading = (90 + 30) % 360  # 120
        assert result.new_heading_deg == expected_heading


class TestAltitudeResolution:
    """Test altitude change resolution logic."""
    
    def test_climb_absolute_altitude(self, ownship, intruder, conflict, clear_command_history):
        """Test climb with absolute altitude specification."""
        llm_resolution = ResolveOut(
            action="climb",
            params={"delta_ft": 1500},
            rationale="Climb to avoid conflict"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.resolution_type == ResolutionType.ALTITUDE_CHANGE
        assert result.new_altitude_ft == 36500  # 35000 + 1500
        assert result.target_aircraft == "OWNSHIP"
    
    def test_descend_absolute_altitude(self, ownship, intruder, conflict, clear_command_history):
        """Test descend with absolute altitude specification."""
        llm_resolution = ResolveOut(
            action="descend",
            params={"delta_ft": 1200},
            rationale="Descend to avoid conflict"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.resolution_type == ResolutionType.ALTITUDE_CHANGE
        assert result.new_altitude_ft == 33800  # 35000 - 1200
    
    def test_altitude_clamp_below_minimum(self, ownship, intruder, conflict, clear_command_history):
        """Test clamping when altitude change is below minimum."""
        llm_resolution = ResolveOut(
            action="climb",
            params={"delta_ft": 500},  # Below MIN_ALTITUDE_CHANGE_FT (1000)
            rationale="Small climb"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.new_altitude_ft == 36000  # 35000 + MIN_ALTITUDE_CHANGE_FT (1000)
    
    def test_altitude_clamp_above_maximum(self, ownship, intruder, conflict, clear_command_history):
        """Test clamping when altitude change exceeds maximum."""
        llm_resolution = ResolveOut(
            action="climb",
            params={"delta_ft": 3000},  # Above MAX_ALTITUDE_CHANGE_FT (2000)
            rationale="Big climb"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.new_altitude_ft == 37000  # 35000 + MAX_ALTITUDE_CHANGE_FT (2000)
    
    def test_altitude_service_ceiling_clamp(self, ownship, intruder, conflict, clear_command_history):
        """Test clamping to service ceiling."""
        # Set high altitude
        ownship.altitude_ft = 44000
        
        llm_resolution = ResolveOut(
            action="climb",
            params={"delta_ft": 2000},
            rationale="Climb near ceiling"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.new_altitude_ft == 45000  # Clamped to max (45000)
    
    def test_altitude_service_floor_clamp(self, ownship, intruder, conflict, clear_command_history):
        """Test clamping to service floor."""
        # Set low altitude
        ownship.altitude_ft = 2000
        
        llm_resolution = ResolveOut(
            action="descend",
            params={"delta_ft": 1500},
            rationale="Descend near floor"
        )
        
        with patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.new_altitude_ft == 1000  # Clamped to min (1000)


class TestWaypointResolution:
    """Test waypoint direct resolution logic."""
    
    def test_waypoint_success_valid_fix(self, ownship, intruder, conflict, clear_command_history):
        """Test successful waypoint resolution with valid fix."""
        # Mock successful waypoint validation
        with patch('src.cdr.nav_utils.validate_waypoint_diversion', return_value=(59.4, 18.2, 25.0)), \
             patch('src.cdr.resolve._validate_resolution_safety', return_value=True):
            
            llm_resolution = ResolveOut(
                action="waypoint",
                params={"waypoint_name": "KAPPA"},
                rationale="Direct to KAPPA waypoint"
            )
            
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
            
            assert result is not None
            assert result.resolution_type == ResolutionType.WAYPOINT_DIRECT
            assert result.waypoint_name == "KAPPA"
            assert result.waypoint_lat == 59.4
            assert result.waypoint_lon == 18.2
            assert result.diversion_distance_nm == 25.0
    
    def test_waypoint_reject_unknown_fix(self, ownship, intruder, conflict, clear_command_history):
        """Test waypoint rejection for unknown fix."""
        # Mock failed waypoint validation (waypoint not found)
        with patch('src.cdr.nav_utils.validate_waypoint_diversion', return_value=None):
            
            llm_resolution = ResolveOut(
                action="waypoint",
                params={"waypoint_name": "UNKNOWN"},
                rationale="Direct to unknown waypoint"
            )
            
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
            
            # Should return None due to waypoint validation failure
            assert result is None
    
    def test_waypoint_missing_name_parameter(self, ownship, intruder, conflict, clear_command_history):
        """Test waypoint resolution with missing waypoint_name parameter."""
        llm_resolution = ResolveOut(
            action="waypoint",
            params={},  # Missing waypoint_name
            rationale="Direct to waypoint"
        )
        
        result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        # Should return None due to missing parameter
        assert result is None


class TestOwnshipOnlyEnforcement:
    """Test that commands are only applied to ownship aircraft."""
    
    def test_target_aircraft_is_ownship(self, ownship, intruder, conflict, clear_command_history):
        """Test that resolution commands target ownship only."""
        llm_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 120},
            rationale="Turn ownship only"
        )
        
        with patch('src.cdr.geodesy.cpa_nm', return_value=(6.0, 5.0)):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.target_aircraft == ownship.aircraft_id
        assert result.target_aircraft != intruder.aircraft_id
        assert result.is_ownship_command is True


class TestBlueSkyCommandApplication:
    """Test application of commands to BlueSky simulator."""
    
    def test_apply_heading_command(self):
        """Test applying heading command to BlueSky."""
        mock_bs = Mock()
        mock_bs.set_heading.return_value = True
        
        llm_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 120},
            rationale="Turn to heading 120"
        )
        
        result = apply_resolution(mock_bs, "OWNSHIP", llm_resolution)
        
        assert result is True
        mock_bs.set_heading.assert_called_once_with("OWNSHIP", 120)
    
    def test_apply_altitude_command(self):
        """Test applying altitude command to BlueSky."""
        mock_bs = Mock()
        mock_bs.set_altitude.return_value = True
        
        llm_resolution = ResolveOut(
            action="climb",
            params={"target_ft": 36000},
            rationale="Climb to FL360"
        )
        
        result = apply_resolution(mock_bs, "OWNSHIP", llm_resolution)
        
        assert result is True
        mock_bs.set_altitude.assert_called_once_with("OWNSHIP", 36000)
    
    def test_apply_waypoint_command(self):
        """Test applying waypoint direct command to BlueSky."""
        mock_bs = Mock()
        mock_bs.direct_to.return_value = True
        
        llm_resolution = ResolveOut(
            action="waypoint",
            params={"waypoint_name": "ALPHA"},
            rationale="Direct to ALPHA"
        )
        
        result = apply_resolution(mock_bs, "OWNSHIP", llm_resolution)
        
        assert result is True
        mock_bs.direct_to.assert_called_once_with("OWNSHIP", "ALPHA")
    
    def test_bluesky_command_failure(self):
        """Test handling of BlueSky command failures."""
        mock_bs = Mock()
        mock_bs.set_heading.return_value = False  # Simulate command failure
        
        llm_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 120},
            rationale="Turn to heading 120"
        )
        
        result = apply_resolution(mock_bs, "OWNSHIP", llm_resolution)
        
        assert result is False
        mock_bs.set_heading.assert_called_once_with("OWNSHIP", 120)


class TestSafetyValidation:
    """Test safety validation logic."""
    
    def test_safety_validation_safe_horizontal(self, ownship, intruder, conflict, clear_command_history):
        """Test safety validation for horizontally safe resolution."""
        # Create a resolution with moderate heading change (within limits)
        llm_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 110},  # 20 degree turn (within 30deg limit)
            rationale="Safe turn away"
        )
        
        with patch('src.cdr.geodesy.cpa_nm', return_value=(6.0, 5.0)):  # Safe separation
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
            
            assert result is not None
            # Result may be fallback resolution due to complex safety validation
            assert result.safety_margin_nm > 0.0
    
    def test_safety_validation_safe_vertical(self, ownship, intruder, conflict, clear_command_history):
        """Test safety validation for vertically safe resolution."""
        # Create vertical resolution with safe altitude separation
        llm_resolution = ResolveOut(
            action="climb",
            params={"delta_ft": 1500},
            rationale="Climb for vertical separation"
        )
        
        with patch('src.cdr.geodesy.cpa_nm', return_value=(2.0, 3.0)):  # Unsafe horizontal
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
            
            # Should pass because vertical separation (1500 ft) > MIN_SAFE_SEPARATION_FT (1000)
            assert result is not None
            assert result.is_validated is True
    
    def test_safety_validation_unsafe_triggers_fallback(self, ownship, intruder, conflict, clear_command_history):
        """Test safety validation failure triggers fallback."""
        # Create resolution that doesn't provide enough separation
        llm_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 95},  # Small turn, still conflict
            rationale="Small turn"
        )
        
        with patch('src.cdr.geodesy.cpa_nm', return_value=(2.0, 3.0)):  # Unsafe separation
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
            
            # Should get fallback resolution (climb +1000 ft)
            assert result is not None
            assert result.source_engine == ResolutionEngine.FALLBACK
            assert result.resolution_type == ResolutionType.ALTITUDE_CHANGE
            assert result.new_altitude_ft == 36000  # 35000 + 1000


class TestUtilityFunctions:
    """Test utility and helper functions."""
    
    def test_format_resolution_command_heading(self):
        """Test formatting heading resolution command."""
        cmd = ResolutionCommand(
            resolution_id="test_cmd",
            target_aircraft="OWNSHIP",
            resolution_type=ResolutionType.HEADING_CHANGE,
            source_engine=ResolutionEngine.HORIZONTAL,
            new_heading_deg=270,
            new_speed_kt=None,
            new_altitude_ft=None,
            waypoint_name=None,
            waypoint_lat=None,
            waypoint_lon=None,
            diversion_distance_nm=None,
            issue_time=datetime.now(),
            safety_margin_nm=5.0,
            is_validated=True,
            is_ownship_command=True,
            angle_within_limits=True,
            altitude_within_limits=True,
            rate_within_limits=True
        )
        
        result = format_resolution_command(cmd)
        assert result == "HDG OWNSHIP 270"
    
    def test_format_resolution_command_altitude(self):
        """Test formatting altitude resolution command."""
        cmd = ResolutionCommand(
            resolution_id="test_cmd",
            target_aircraft="OWNSHIP",
            resolution_type=ResolutionType.ALTITUDE_CHANGE,
            source_engine=ResolutionEngine.VERTICAL,
            new_heading_deg=None,
            new_speed_kt=None,
            new_altitude_ft=37000,
            waypoint_name=None,
            waypoint_lat=None,
            waypoint_lon=None,
            diversion_distance_nm=None,
            issue_time=datetime.now(),
            safety_margin_nm=5.0,
            is_validated=True,
            is_ownship_command=True,
            angle_within_limits=True,
            altitude_within_limits=True,
            rate_within_limits=True
        )
        
        result = format_resolution_command(cmd)
        assert result == "ALT OWNSHIP 37000"
    
    def test_format_resolution_command_waypoint(self):
        """Test formatting waypoint resolution command."""
        cmd = ResolutionCommand(
            resolution_id="test_cmd",
            target_aircraft="OWNSHIP",
            resolution_type=ResolutionType.WAYPOINT_DIRECT,
            source_engine=ResolutionEngine.HORIZONTAL,
            new_heading_deg=None,
            new_speed_kt=None,
            new_altitude_ft=None,
            waypoint_name="KAPPA",
            waypoint_lat=59.4,
            waypoint_lon=18.2,
            diversion_distance_nm=25.0,
            issue_time=datetime.now(),
            safety_margin_nm=5.0,
            is_validated=True,
            is_ownship_command=True,
            angle_within_limits=True,
            altitude_within_limits=True,
            rate_within_limits=True
        )
        
        result = format_resolution_command(cmd)
        assert result == "DIRECT OWNSHIP KAPPA"
    
    def test_bluesky_command_formatting_edge_cases(self):
        """Test BlueSky command formatting with edge cases."""
        # Test heading command formatting
        result = to_bluesky_command_heading("TEST123", 359.7)
        assert result == "TEST123 HDG 360"
        
        # Test altitude command formatting  
        result = to_bluesky_command_altitude("TEST123", 35999.9)
        assert result == "TEST123 ALT 36000"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_execute_resolution_without_context(self, clear_command_history):
        """Test execute_resolution called without full context."""
        llm_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 120},
            rationale="Turn without context"
        )
        
        result = execute_resolution(llm_resolution)
        assert result is None
    
    def test_apply_resolution_without_context(self, clear_command_history):
        """Test apply_resolution called without full context."""
        llm_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 120},
            rationale="Turn without context"
        )
        
        result = apply_resolution()
        assert result is False
    
    def test_invalid_action_type(self, ownship, intruder, conflict, clear_command_history):
        """Test handling of invalid action types."""
        llm_resolution = ResolveOut(
            action="invalid_action",
            params={"heading_deg": 120},
            rationale="Invalid action"
        )
        
        result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        assert result is None
    
    def test_missing_altitude_parameters(self, ownship, intruder, conflict, clear_command_history):
        """Test altitude action with missing parameters."""
        llm_resolution = ResolveOut(
            action="climb",
            params={},  # No altitude parameters
            rationale="Climb without parameters"
        )
        
        with patch('src.cdr.geodesy.cpa_nm', return_value=(2.0, 3.0)):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
            # Should use default values
            assert result is not None
            assert result.new_altitude_ft == 36000  # 35000 + default 1000
    
    def test_heading_wraparound_cases(self, ownship, intruder, conflict, clear_command_history):
        """Test heading wraparound edge cases."""
        # Test turning past 360 degrees
        ownship.heading_deg = 350
        
        llm_resolution = ResolveOut(
            action="turn",
            params={"direction": "right", "degrees": 20},
            rationale="Turn past 360"
        )
        
        with patch('src.cdr.geodesy.cpa_nm', return_value=(6.0, 5.0)):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        expected_heading = (350 + 20) % 360  # 10
        assert result.new_heading_deg == expected_heading
    
    def test_altitude_boundary_conditions(self, ownship, intruder, conflict, clear_command_history):
        """Test altitude at service ceiling and floor boundaries."""
        # Test at service ceiling
        ownship.altitude_ft = 45000
        
        llm_resolution = ResolveOut(
            action="climb",
            params={"delta_ft": 500},
            rationale="Climb at ceiling"
        )
        
        with patch('src.cdr.geodesy.cpa_nm', return_value=(2.0, 3.0)):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.new_altitude_ft == 45000  # Clamped to ceiling
        
        # Test at service floor  
        ownship.altitude_ft = 1500
        
        llm_resolution = ResolveOut(
            action="descend",
            params={"delta_ft": 1000},
            rationale="Descend near floor"
        )
        
        with patch('src.cdr.geodesy.cpa_nm', return_value=(2.0, 3.0)):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.new_altitude_ft == 1000  # Clamped to floor


class TestOscillationGuards:
    """Test oscillation guard mechanisms."""
    
    def test_oscillation_guard_allows_first_command(self, ownship, intruder, conflict, clear_command_history):
        """Test that oscillation guard allows first command."""
        llm_resolution = ResolveOut(
            action="turn",
            params={"direction": "right", "degrees": 20},
            rationale="First turn command"
        )
        
        with patch('src.cdr.geodesy.cpa_nm', return_value=(6.0, 5.0)):
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        
        assert result is not None
        assert result.is_validated is True
    
    def test_oscillation_guard_blocks_opposite_without_benefit(self, ownship, intruder, conflict, clear_command_history):
        """Test that oscillation guard blocks opposite commands without sufficient benefit."""
        from src.cdr.resolve import _add_command_to_history
        
        # First command - right turn
        _add_command_to_history("OWNSHIP", "turn_right", heading_change=20, separation_benefit=2.0)
        
        # Immediately try opposite command with low benefit
        llm_resolution = ResolveOut(
            action="turn",
            params={"direction": "left", "degrees": 15},
            rationale="Opposite turn"
        )
        
        with patch('src.cdr.resolve._estimate_separation_benefit', return_value=0.3):  # Below threshold
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
            
            # Should be blocked by oscillation guard
            assert result is None
    
    def test_oscillation_guard_allows_opposite_with_sufficient_benefit(self, ownship, intruder, conflict, clear_command_history):
        """Test that oscillation guard allows opposite commands with sufficient benefit."""
        from src.cdr.resolve import _add_command_to_history
        
        # First command - right turn
        _add_command_to_history("OWNSHIP", "turn_right", heading_change=20, separation_benefit=2.0)
        
        # Try opposite command with sufficient benefit
        llm_resolution = ResolveOut(
            action="turn",
            params={"direction": "left", "degrees": 25},
            rationale="Opposite turn with good benefit"
        )
        
        with patch('src.cdr.resolve._estimate_separation_benefit', return_value=1.0), \
             patch('src.cdr.geodesy.cpa_nm', return_value=(6.0, 5.0)):  # Above threshold
            result = execute_resolution(llm_resolution, ownship, intruder, conflict)
            
            # Should be allowed despite being opposite
            assert result is not None
            assert result.is_validated is True


class TestAdditionalEdgeCases:
    """Test additional edge cases and error conditions."""
    
    def test_execute_resolution_missing_context(self):
        """Test execute_resolution with missing context parameters."""
        from src.cdr.resolve import execute_resolution
        
        llm_resolution = ResolveOut(
            action="turn",
            params={"heading_deg": 90},
            rationale="Test resolution"
        )
        
        # Call without context - should return None
        result = execute_resolution(llm_resolution, None, None, None)
        assert result is None
        
        # Call with partial context - should return None
        result = execute_resolution(llm_resolution, Mock(), None, None)
        assert result is None
    
    def test_invalid_action_type(self, ownship, intruder, conflict):
        """Test resolution with invalid action type."""
        from src.cdr.resolve import execute_resolution
        
        llm_resolution = ResolveOut(
            action="invalid_action",
            params={},
            rationale="Invalid action test"
        )
        
        result = execute_resolution(llm_resolution, ownship, intruder, conflict)
        assert result is None
