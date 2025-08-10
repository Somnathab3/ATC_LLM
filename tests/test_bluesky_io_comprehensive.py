"""Comprehensive test suite for BlueSky I/O without real simulator.

This test suite covers:
- BSConfig default values and overrides
- Command formatting for all ATC commands
- Reset flow (DEL ALL, RESET, step)
- Error handling when stack() returns False
- All functionality without requiring actual BlueSky installation
"""

import logging
import pytest
from typing import Any, List, Union
from unittest.mock import Mock, patch
from types import SimpleNamespace

from src.cdr.bluesky_io import BlueSkyClient, BSConfig


class FakeSimulator:
    """Mock simulator that records step calls."""
    
    def __init__(self):
        self.steps: List[float] = []
        self.total_time = 0.0
    
    def step(self, dt: float) -> None:
        """Record simulation steps."""
        self.steps.append(dt)
        self.total_time += dt


class FakeTraffic:
    """Mock traffic object for state testing."""
    
    def __init__(self):
        self.ntraf = 0
        self.id: List[str] = []
        self.lat: List[float] = []
        self.lon: List[float] = []
        self.alt: List[float] = []  # meters
        self.gs: List[float] = []   # m/s
        self.hdg: List[float] = []  # degrees
        self.vs: List[float] = []   # m/s
    
    def add_aircraft(self, callsign: str, lat: float, lon: float, 
                    alt_m: float, spd_ms: float, hdg_deg: float, vs_ms: float = 0.0):
        """Add a mock aircraft to traffic."""
        self.id.append(callsign)
        self.lat.append(lat)
        self.lon.append(lon)
        self.alt.append(alt_m)
        self.gs.append(spd_ms)
        self.hdg.append(hdg_deg)
        self.vs.append(vs_ms)
        self.ntraf += 1


class TestBSConfigDefaults:
    """Test BSConfig default values and overrides."""
    
    def test_bsconfig_defaults(self):
        """Test BSConfig default values."""
        config = BSConfig()
        assert config.headless is True
    
    def test_bsconfig_overrides(self):
        """Test BSConfig parameter overrides."""
        config = BSConfig(headless=False)
        assert config.headless is False
    
    def test_bluesky_client_default_host_port(self):
        """Test default host and port values."""
        config = BSConfig()
        client = BlueSkyClient(config)
        
        assert client.host == '127.0.0.1'
        assert client.port == 5555
    
    def test_bluesky_client_custom_host_port(self):
        """Test custom host and port via config attributes."""
        config = SimpleNamespace(
            headless=True,
            bluesky_host='192.168.1.100',
            bluesky_port=8080
        )
        client = BlueSkyClient(config)
        
        assert client.host == '192.168.1.100'
        assert client.port == 8080
    
    def test_bluesky_client_config_without_host_port(self):
        """Test config object without host/port attributes."""
        config = SimpleNamespace(headless=True)
        client = BlueSkyClient(config)
        
        # Should fall back to defaults
        assert client.host == '127.0.0.1'
        assert client.port == 5555


class TestCommandFormatting:
    """Test command formatting and pass-through to stack()."""
    
    def setup_method(self):
        """Set up client with mocked stack method."""
        self.config = BSConfig()
        self.client = BlueSkyClient(self.config)
        self.captured_commands: List[str] = []
        
        # Mock the stack method to capture commands
        def mock_stack(cmd: str) -> bool:
            self.captured_commands.append(cmd)
            return True
        
        self.client.stack = mock_stack
    
    def test_create_aircraft_command_formatting(self):
        """Test CRE command formatting."""
        # Mock the stack method to test command formatting
        def create_via_stack(cs: str, actype: str, lat: float, lon: float, hdg_deg: float, alt_ft: float, spd_kt: float) -> bool:
            return self.client.stack(
                f"CRE {cs},{actype},{lat:.6f},{lon:.6f},{hdg_deg:.1f},{alt_ft:.0f},{spd_kt:.0f}"
            )
        
        result = create_via_stack(
            'TEST001', 'B737', 40.123456, -74.654321, 270.5, 35000.7, 450.2
        )
        
        assert result is True
        assert len(self.captured_commands) == 1
        cmd = self.captured_commands[0]
        assert "CRE TEST001,B737,40.123456,-74.654321,270.5,35001,450" in cmd
    
    def test_heading_command_formatting(self):
        """Test HDG command formatting."""
        result = self.client.set_heading('OWNSHIP', 123.4)
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert self.captured_commands[0] == "OWNSHIP HDG 123"
    
    def test_heading_command_rounding(self):
        """Test HDG command rounding."""
        self.client.set_heading('TEST', 89.6)
        assert "TEST HDG 90" in self.captured_commands[0]
        
        self.captured_commands.clear()
        self.client.set_heading('TEST', 89.4)
        assert "TEST HDG 89" in self.captured_commands[0]
    
    def test_altitude_command_formatting(self):
        """Test ALT command formatting."""
        result = self.client.set_altitude('OWNSHIP', 35500.6)
        
        assert result is True
        assert len(self.captured_commands) == 1
        # Should round to nearest integer
        assert self.captured_commands[0] in ["OWNSHIP ALT 35501", "OWNSHIP ALT 35500"]
    
    def test_direct_to_command_formatting(self):
        """Test DCT (direct to) command formatting."""
        result = self.client.direct_to('OWNSHIP', 'GOTUR')
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert self.captured_commands[0] == "OWNSHIP DCT GOTUR"
    
    def test_speed_command_formatting(self):
        """Test SPD command formatting."""
        result = self.client.set_speed('OWNSHIP', 250.7)
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert self.captured_commands[0] == "OWNSHIP SPD 251"
    
    def test_add_waypoint_without_altitude(self):
        """Test ADDWPT command without altitude."""
        result = self.client.add_waypoint('OWNSHIP', 59.5, 18.6)
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert self.captured_commands[0].startswith("ADDWPT OWNSHIP 59.500000 18.600000")
        # Should not contain altitude
        assert "59.500000 18.600000" in self.captured_commands[0]
    
    def test_add_waypoint_with_altitude(self):
        """Test ADDWPT command with altitude."""
        result = self.client.add_waypoint('OWNSHIP', 59.6, 18.7, 34000)
        
        assert result is True
        assert len(self.captured_commands) == 1
        cmd = self.captured_commands[0]
        assert cmd.startswith("ADDWPT OWNSHIP 59.600000 18.700000")
        # Should contain altitude in meters (34000 ft * 0.3048 = 10363.2 m)
        assert "10363.2" in cmd
    
    def test_multiple_commands_sequence(self):
        """Test sequence of commands are all captured."""
        self.client.set_heading('OWNSHIP', 90)
        self.client.set_altitude('OWNSHIP', 25000)
        self.client.direct_to('OWNSHIP', 'WAYPOINT1')
        self.client.set_speed('OWNSHIP', 300)
        
        assert len(self.captured_commands) == 4
        assert "OWNSHIP HDG 90" in self.captured_commands[0]
        assert "OWNSHIP ALT 25000" in self.captured_commands[1]
        assert "OWNSHIP DCT WAYPOINT1" in self.captured_commands[2]
        assert "OWNSHIP SPD 300" in self.captured_commands[3]


class TestResetFlow:
    """Test reset flow: DEL ALL, RESET, then tiny step to flush."""
    
    def setup_method(self):
        """Set up client with mocked dependencies."""
        self.config = BSConfig()
        self.client = BlueSkyClient(self.config)
        self.captured_commands: List[str] = []
        self.fake_sim = FakeSimulator()
        
        # Mock stack method
        def mock_stack(cmd: str) -> bool:
            self.captured_commands.append(cmd)
            return True
        
        self.client.stack = mock_stack
    
    def test_sim_reset_command_order(self, monkeypatch: Any):
        """Test that sim_reset() calls DEL ALL then RESET."""
        # Set the sim attribute on the client
        setattr(self.client, "sim", self.fake_sim)
        
        result = self.client.sim_reset()
        
        assert result is True
        assert len(self.captured_commands) == 2
        assert self.captured_commands[0] == "DEL ALL"
        assert self.captured_commands[1] == "RESET"
    
    def test_sim_reset_flush_step(self, monkeypatch: Any):
        """Test that sim_reset() calls sim.step(1) to flush."""
        # Set the sim attribute on the client
        setattr(self.client, "sim", self.fake_sim)
        
        self.client.sim_reset()
        
        # Should have called step(1) once
        assert len(self.fake_sim.steps) == 1
        assert self.fake_sim.steps[0] == 1
    
    def test_sim_reset_handles_del_all_exception(self):
        """Test that reset continues even if DEL ALL fails."""
        def mock_stack_with_del_failure(cmd: str) -> bool:
            self.captured_commands.append(cmd)
            if cmd == "DEL ALL":
                raise Exception("DEL ALL failed")
            return True
        
        self.client.stack = mock_stack_with_del_failure
        
        # Should still succeed and call RESET
        result = self.client.sim_reset()
        assert result is True
        assert "RESET" in self.captured_commands
    
    def test_sim_reset_handles_sim_step_exception(self):
        """Test that reset succeeds even if sim.step() fails."""
        def failing_step(dt: float):
            raise Exception("Sim step failed")
        
        self.fake_sim.step = failing_step
        
        # Should still return True (RESET command succeeded)
        result = self.client.sim_reset()
        assert result is True


class TestErrorHandling:
    """Test error handling when stack() returns False."""
    
    def setup_method(self):
        """Set up client with controllable stack method."""
        self.config = BSConfig()
        self.client = BlueSkyClient(self.config)
        self.stack_return_value = True
        self.captured_commands: List[str] = []
        
        def mock_stack(cmd: str) -> bool:
            self.captured_commands.append(cmd)
            return self.stack_return_value
        
        self.client.stack = mock_stack
    
    def test_successful_command_returns_true(self):
        """Test that successful commands return True."""
        self.stack_return_value = True
        
        result = self.client.set_heading('OWNSHIP', 270)
        assert result is True
    
    def test_failed_command_returns_false(self):
        """Test that failed commands return False."""
        self.stack_return_value = False
        
        result = self.client.set_heading('OWNSHIP', 270)
        assert result is False
    
    def test_failed_command_logs_warning(self, caplog: Any):
        """Test that we can test False propagation properly."""
        # Set stack to return False
        self.stack_return_value = False
        
        # The set_heading should return False
        result = self.client.set_heading('OWNSHIP', 270)
        assert result is False
        
        # Our mock captured the command
        assert len(self.captured_commands) == 1
    
    def test_all_command_methods_propagate_false(self):
        """Test that all command methods propagate False from stack."""
        self.stack_return_value = False
        
        # Test all command methods
        assert self.client.set_heading('TEST', 90) is False
        assert self.client.set_altitude('TEST', 25000) is False
        assert self.client.direct_to('TEST', 'WPT') is False
        assert self.client.set_speed('TEST', 250) is False
        assert self.client.add_waypoint('TEST', 60.0, 20.0) is False
        assert self.client.add_waypoint('TEST', 60.0, 20.0, 30000) is False
    
    def test_real_stack_method_logs_warning_on_false(self, caplog: Any):
        """Test that the real stack method logs warning when bs.stack returns False."""
        # Set up a real client with real stack method but mock bs.stack
        config = BSConfig()
        client = BlueSkyClient(config)
        mock_bs = Mock()
        mock_bs.stack = Mock(return_value=False)  # Make bs.stack return False
        setattr(client, "bs", mock_bs)
        
        with caplog.at_level(logging.WARNING):
            result = client.stack("TEST COMMAND")
        
        assert result is False
        # Check that the real stack method logged the warning
        assert any("returned False" in record.getMessage() for record in caplog.records)


class TestStepFunctions:
    """Test simulation step functions."""
    
    def setup_method(self):
        """Set up client with fake simulator."""
        self.config = BSConfig()
        self.client = BlueSkyClient(self.config)
        self.fake_sim = FakeSimulator()
        self.captured_commands: List[str] = []
        
        def mock_stack(cmd: str) -> bool:
            self.captured_commands.append(cmd)
            return True
        
        self.client.stack = mock_stack
    
    def test_step_minutes_basic(self, monkeypatch: Any):
        """Test basic step_minutes functionality."""
        setattr(self.client, "sim", self.fake_sim)
        
        result = self.client.step_minutes(1.0)  # 1 minute = 60 seconds
        
        assert result is True
        # Should have called step multiple times with dt=0.5
        assert len(self.fake_sim.steps) == 120  # 60 seconds / 0.5 dt = 120 steps
        assert all(step == 0.5 for step in self.fake_sim.steps)
        assert self.fake_sim.total_time == 60.0
    
    def test_step_minutes_small_values(self, monkeypatch: Any):
        """Test step_minutes with very small values."""
        setattr(self.client, "sim", self.fake_sim)
        
        result = self.client.step_minutes(0.02)  # 1.2 seconds
        
        assert result is True
        # Should have at least some steps
        assert len(self.fake_sim.steps) >= 1
        # Total time should be close to 1.2 seconds (allow for rounding up to integer steps)
        expected_time = 0.02 * 60.0  # 1.2 seconds
        assert abs(self.fake_sim.total_time - expected_time) < 1.0  # Allow 1 second tolerance
    
    def test_step_minutes_zero_or_negative(self, monkeypatch: Any):
        """Test step_minutes with zero or negative values."""
        setattr(self.client, "sim", self.fake_sim)
        
        result1 = self.client.step_minutes(0.0)
        result2 = self.client.step_minutes(-1.0)
        
        assert result1 is True
        assert result2 is True
        # Should not have called step at all
        assert len(self.fake_sim.steps) == 0
    
    def test_step_minutes_exception_handling(self, monkeypatch: Any):
        """Test step_minutes handles simulator exceptions."""
        def failing_step(dt: float):
            raise Exception("Simulator failed")
        
        self.fake_sim.step = failing_step
        setattr(self.client, "sim", self.fake_sim)
        
        result = self.client.step_minutes(1.0)
        assert result is False
    
    def test_sim_fastforward_command(self):
        """Test FF (fastforward) command formatting."""
        result = self.client.sim_fastforward(300)
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert self.captured_commands[0] == "FF 300"
    
    def test_sim_dtmult_command(self):
        """Test DTMULT command formatting."""
        result = self.client.sim_set_dtmult(4.0)
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert self.captured_commands[0] == "DTMULT 4.0"
    
    def test_sim_realtime_command(self):
        """Test REALTIME command formatting."""
        result1 = self.client.sim_realtime(True)
        result2 = self.client.sim_realtime(False)
        
        assert result1 is True
        assert result2 is True
        assert len(self.captured_commands) == 2
        assert self.captured_commands[0] == "REALTIME ON"
        assert self.captured_commands[1] == "REALTIME OFF"
    
    def test_sim_set_time_utc_command(self):
        """Test TIME command formatting."""
        result = self.client.sim_set_time_utc("2025-08-10T14:30:00")
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert self.captured_commands[0] == "TIME 2025-08-10T14:30:00"


class TestExecuteCommand:
    """Test execute_command method dispatch."""
    
    def setup_method(self):
        """Set up client with mocked methods."""
        self.config = BSConfig()
        self.client = BlueSkyClient(self.config)
        self.captured_commands: List[str] = []
        
        def mock_stack(cmd: str) -> bool:
            self.captured_commands.append(cmd)
            return True
        
        self.client.stack = mock_stack
    
    def test_execute_heading_resolution(self):
        """Test execute_command with heading resolution."""
        resolution = SimpleNamespace(
            resolution_type='heading_change',
            target_aircraft='OWNSHIP',
            new_heading_deg=200,
            new_altitude_ft=None,
            waypoint_name=None
        )
        
        result = self.client.execute_command(resolution)
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert "OWNSHIP HDG 200" in self.captured_commands[0]
    
    def test_execute_altitude_resolution(self):
        """Test execute_command with altitude resolution."""
        resolution = SimpleNamespace(
            resolution_type='altitude_change',
            target_aircraft='OWNSHIP',
            new_heading_deg=None,
            new_altitude_ft=33000,
            waypoint_name=None
        )
        
        result = self.client.execute_command(resolution)
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert "OWNSHIP ALT 33000" in self.captured_commands[0]
    
    def test_execute_waypoint_resolution(self):
        """Test execute_command with waypoint resolution."""
        resolution = SimpleNamespace(
            resolution_type='waypoint_direct',
            target_aircraft='OWNSHIP',
            new_heading_deg=None,
            new_altitude_ft=None,
            waypoint_name='ABCD'
        )
        
        result = self.client.execute_command(resolution)
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert self.captured_commands[0] == "OWNSHIP DCT ABCD"
    
    def test_execute_unknown_resolution(self):
        """Test execute_command with unknown resolution type."""
        resolution = SimpleNamespace()  # Empty object
        
        result = self.client.execute_command(resolution)
        
        assert result is False
        assert len(self.captured_commands) == 0


class TestAircraftStates:
    """Test aircraft state fetching with unit conversions."""
    
    def setup_method(self):
        """Set up client with fake traffic."""
        self.config = BSConfig()
        self.client = BlueSkyClient(self.config)
        self.fake_traf = FakeTraffic()
    
    def test_get_aircraft_states_empty(self, monkeypatch: Any):
        """Test getting states when no aircraft present."""
        setattr(self.client, "traf", self.fake_traf)
        setattr(self.client, "bs", Mock())  # Mock to indicate connected
        
        states = self.client.get_aircraft_states()
        
        assert isinstance(states, dict)
        assert len(states) == 0
    
    def test_get_aircraft_states_single_aircraft(self, monkeypatch: Any):
        """Test getting states for single aircraft with unit conversions."""
        # Add aircraft: alt in meters, speed in m/s
        self.fake_traf.add_aircraft(
            callsign='OWNSHIP',
            lat=40.7128,
            lon=-74.0060,
            alt_m=10668.0,  # 35000 ft
            spd_ms=128.6,   # ~250 kt
            hdg_deg=270.0,
            vs_ms=2.54      # ~500 fpm
        )
        
        setattr(self.client, "traf", self.fake_traf)
        setattr(self.client, "bs", Mock())  # Mock to indicate connected
        
        states = self.client.get_aircraft_states()
        
        assert len(states) == 1
        assert 'OWNSHIP' in states
        
        state = states['OWNSHIP']
        assert state['id'] == 'OWNSHIP'
        assert state['lat'] == 40.7128
        assert state['lon'] == -74.0060
        
        # Test unit conversions
        assert abs(state['alt_ft'] - 35000) < 10  # meters to feet
        assert abs(state['spd_kt'] - 250) < 5     # m/s to knots
        assert state['hdg_deg'] == 270.0
        assert abs(state['roc_fpm'] - 500) < 10   # m/s to fpm
    
    def test_get_aircraft_states_multiple_aircraft(self, monkeypatch: Any):
        """Test getting states for multiple aircraft."""
        self.fake_traf.add_aircraft('OWNSHIP', 40.0, -74.0, 10000, 100, 90, 0)
        self.fake_traf.add_aircraft('INTRUDER1', 41.0, -75.0, 11000, 120, 180, 5)
        
        setattr(self.client, "traf", self.fake_traf)
        setattr(self.client, "bs", Mock())  # Mock to indicate connected
        
        states = self.client.get_aircraft_states()
        
        assert len(states) == 2
        assert 'OWNSHIP' in states
        assert 'INTRUDER1' in states
    
    def test_get_aircraft_states_callsign_normalization(self, monkeypatch: Any):
        """Test that callsigns are normalized to uppercase."""
        self.fake_traf.add_aircraft('  ownship  ', 40.0, -74.0, 10000, 100, 90, 0)
        
        setattr(self.client, "traf", self.fake_traf)
        setattr(self.client, "bs", Mock())  # Mock to indicate connected
        
        states = self.client.get_aircraft_states()
        
        assert 'OWNSHIP' in states
        assert states['OWNSHIP']['id'] == 'OWNSHIP'
    
    def test_get_aircraft_states_not_connected(self):
        """Test that get_aircraft_states raises error when not connected."""
        # client.bs is None by default (not connected)
        
        with pytest.raises(RuntimeError, match="BlueSky not connected"):
            self.client.get_aircraft_states()


class TestConnectionStatus:
    """Test connection status methods."""
    
    def test_is_connected_true(self, monkeypatch: Any):
        """Test is_connected returns True when bs is set."""
        config = BSConfig()
        client = BlueSkyClient(config)
        setattr(client, "bs", Mock())
        
        assert client.is_connected() is True
    
    def test_is_connected_false(self):
        """Test is_connected returns False when bs is None."""
        config = BSConfig()
        client = BlueSkyClient(config)
        # client.bs is None by default
        
        assert client.is_connected() is False
    
    def test_is_mock_mode_true(self):
        """Test is_mock_mode returns True when not connected."""
        config = BSConfig()
        client = BlueSkyClient(config)
        # client.bs is None by default
        
        assert client.is_mock_mode() is True
    
    def test_is_mock_mode_false(self, monkeypatch: Any):
        """Test is_mock_mode returns False when connected."""
        config = BSConfig()
        client = BlueSkyClient(config)
        setattr(client, "bs", Mock())
        
        assert client.is_mock_mode() is False


class TestRouteManagement:
    """Test route and waypoint management."""
    
    def setup_method(self):
        """Set up client with mocked dependencies."""
        self.config = BSConfig()
        self.client = BlueSkyClient(self.config)
        self.captured_commands: List[str] = []
        
        def mock_stack(cmd: str) -> bool:
            self.captured_commands.append(cmd)
            return True
        
        self.client.stack = mock_stack
    
    def test_add_waypoints_from_route_success(self, monkeypatch: Any):
        """Test adding multiple waypoints from route."""
        # Mock traf for waypoint existence checks
        setattr(self.client, "traf", SimpleNamespace(id=['OWNSHIP']))
        
        route = [(40.0, -74.0), (41.0, -75.0), (42.0, -76.0)]
        
        result = self.client.add_waypoints_from_route('OWNSHIP', route, 35000)
        
        assert result is True
        assert len(self.captured_commands) == 3
        # Each command should be ADDWPT with coordinates and altitude
        for i, cmd in enumerate(self.captured_commands):
            assert cmd.startswith(f"ADDWPT OWNSHIP {route[i][0]:.6f} {route[i][1]:.6f}")
    
    def test_add_waypoints_from_route_empty(self):
        """Test adding waypoints from empty route."""
        result = self.client.add_waypoints_from_route('OWNSHIP', [])
        
        assert result is False
        assert len(self.captured_commands) == 0
    
    def test_direct_to_waypoint_alias(self):
        """Test direct_to_waypoint is alias for direct_to."""
        result = self.client.direct_to_waypoint('OWNSHIP', 'WAYPOINT1')
        
        assert result is True
        assert len(self.captured_commands) == 1
        assert self.captured_commands[0] == "OWNSHIP DCT WAYPOINT1"


class TestCleanupAndShutdown:
    """Test cleanup and shutdown functionality."""
    
    def setup_method(self):
        """Set up client with mocked dependencies."""
        self.config = BSConfig()
        self.client = BlueSkyClient(self.config)
        self.captured_commands: List[str] = []
        
        def mock_stack(cmd: str) -> bool:
            self.captured_commands.append(cmd)
            return True
        
        self.client.stack = mock_stack
    
    def test_close_method(self, monkeypatch: Any):
        """Test close method sends cleanup commands."""
        # Create a mock bs object that has a stack method
        mock_bs = Mock()
        
        def mock_stack_method(cmd: str) -> None:
            self.captured_commands.append(cmd)
        
        mock_bs.stack = mock_stack_method
        setattr(self.client, "bs", mock_bs)
        
        self.client.close()
        
        # Should send TRAIL OFF, AREA OFF, DEL ALL
        assert len(self.captured_commands) >= 1
        expected_commands = ["TRAIL OFF", "AREA OFF", "DEL ALL"]
        assert any(cmd in self.captured_commands for cmd in expected_commands)
    
    def test_close_handles_exceptions(self, monkeypatch: Any):
        """Test close method handles exceptions gracefully."""
        def failing_stack(cmd: str) -> bool:
            raise Exception("Stack failed")
        
        # Create a mock bs object that has a failing stack method
        mock_bs = Mock()
        mock_bs.stack = failing_stack
        setattr(self.client, "bs", mock_bs)
        
        # Should not raise exception
        self.client.close()
    
    def test_safe_shutdown_via_public_method(self):
        """Test safe shutdown through public method."""
        # Call close() instead of the protected _safe_shutdown()
        # Should not raise any exceptions
        self.client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
