"""Test suite for SCAT dataset adapter."""

import pytest
import tempfile
import json
from datetime import datetime, timezone
from pathlib import Path

from src.cdr.scat_adapter import SCATAdapter, SCATFlightRecord, load_scat_scenario
from src.cdr.schemas import AircraftState


class TestSCATAdapter:
    """Test SCAT adapter functionality."""
    
    def create_mock_scat_file(self, callsign="TEST001", track_points=5):
        """Create a mock SCAT JSON file for testing."""
        track = []
        base_time = "2016-10-18T19:19:00.000000"
        
        for i in range(track_points):
            track.append({
                "I062/105": {"lat": 56.0 + i * 0.01, "lon": 18.0 + i * 0.01},
                "I062/136": {"measured_flight_level": 350.0},
                "I062/185": {"vx": 200.0, "vy": 150.0},
                "I062/200": {"adf": False, "long": 0, "trans": 0, "vert": 0},
                "I062/220": {"rocd": 0.0},
                "time_of_track": f"2016-10-18T19:19:{i:02d}.000000"
            })
        
        return {
            "centre_ctrl": [{"centre_id": 1, "start_time": base_time}],
            "fpl": {
                "fpl_base": [{
                    "callsign": callsign,
                    "aircraft_type": "A332",
                    "flight_rules": "I",
                    "wtc": "H",
                    "adep": "EDDF",
                    "ades": "ZSPD",
                    "equip_status_rvsm": True,
                    "time_stamp": base_time
                }]
            },
            "surveillance_track": track
        }
    
    def test_adapter_initialization(self):
        """Test SCAT adapter initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock files
            temp_path = Path(temp_dir)
            
            # Create valid flight files
            for i in range(3):
                flight_file = temp_path / f"11220{i}.json"
                with open(flight_file, 'w') as f:
                    json.dump(self.create_mock_scat_file(f"TEST{i:03d}"), f)
            
            # Create non-flight files (should be ignored)
            (temp_path / "airspace").touch()
            (temp_path / "grib_met").touch()
            
            adapter = SCATAdapter(str(temp_path))
            assert len(adapter.flight_files) == 3
    
    def test_flight_record_parsing(self):
        """Test parsing of SCAT flight record."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            flight_file = temp_path / "112200.json"
            
            mock_data = self.create_mock_scat_file("CCA936", track_points=10)
            with open(flight_file, 'w') as f:
                json.dump(mock_data, f)
            
            adapter = SCATAdapter(str(temp_path))
            record = adapter.load_flight_record(flight_file)
            
            assert record is not None
            assert record.callsign == "CCA936"
            assert record.aircraft_type == "A332"
            assert record.flight_rules == "I"
            assert record.wtc == "H"
            assert record.adep == "EDDF"
            assert record.ades == "ZSPD"
            assert record.rvsm_capable is True
            assert len(record.track_points) == 10
            assert record.centre_id == 1
    
    def test_aircraft_state_extraction(self):
        """Test extraction of aircraft states from SCAT record."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            flight_file = temp_path / "112200.json"
            
            mock_data = self.create_mock_scat_file("TEST001", track_points=5)
            with open(flight_file, 'w') as f:
                json.dump(mock_data, f)
            
            adapter = SCATAdapter(str(temp_path))
            record = adapter.load_flight_record(flight_file)
            states = adapter.extract_aircraft_states(record)
            
            assert len(states) == 5
            
            # Check first state
            state = states[0]
            assert isinstance(state, AircraftState)
            assert state.aircraft_id == "TEST001"
            assert state.latitude == 56.0
            assert state.longitude == 18.0
            assert state.altitude_ft == 35000.0  # FL350 = 35000 ft
            assert state.ground_speed_kt > 0  # Calculated from vx, vy
            assert 0 <= state.heading_deg < 360
            assert state.vertical_speed_fpm == 0.0  # rocd = 0
    
    def test_scenario_loading(self):
        """Test loading complete traffic scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple flight files
            for i in range(5):
                flight_file = temp_path / f"11220{i}.json"
                mock_data = self.create_mock_scat_file(f"TEST{i:03d}", track_points=3)
                with open(flight_file, 'w') as f:
                    json.dump(mock_data, f)
            
            adapter = SCATAdapter(str(temp_path))
            scenario_states = adapter.load_scenario(max_flights=3, time_window_minutes=60)
            
            # Should have 3 flights × 3 track points = 9 states
            assert len(scenario_states) == 9
            
            # States should be sorted by timestamp
            timestamps = [state.timestamp for state in scenario_states]
            assert timestamps == sorted(timestamps)
    
    def test_time_filtering(self):
        """Test time-based filtering of aircraft states."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            flight_file = temp_path / "112200.json"
            
            mock_data = self.create_mock_scat_file("TEST001", track_points=10)
            with open(flight_file, 'w') as f:
                json.dump(mock_data, f)
            
            adapter = SCATAdapter(str(temp_path))
            record = adapter.load_flight_record(flight_file)
            
            # Define time filter (first 3 seconds only)
            start_time = datetime(2016, 10, 18, 19, 19, 0, tzinfo=timezone.utc)
            end_time = datetime(2016, 10, 18, 19, 19, 3, tzinfo=timezone.utc)
            
            states = adapter.extract_aircraft_states(record, time_filter=(start_time, end_time))
            
            # Should get 4 states (times 0, 1, 2, 3)
            assert len(states) == 4
            
            # All states should be within time window
            for state in states:
                assert start_time <= state.timestamp <= end_time
    
    def test_dataset_summary(self):
        """Test dataset summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create flight files with different aircraft types and airports
            flight_configs = [
                ("DLH123", "A332", "EDDF", "KJFK"),
                ("BAW456", "B777", "EGLL", "KJFK"),
                ("AFR789", "A380", "LFPG", "EDDF")
            ]
            
            for i, (callsign, aircraft_type, adep, ades) in enumerate(flight_configs):
                flight_file = temp_path / f"11220{i}.json"
                mock_data = self.create_mock_scat_file(callsign, track_points=2)
                mock_data["fpl"]["fpl_base"][0]["aircraft_type"] = aircraft_type
                mock_data["fpl"]["fpl_base"][0]["adep"] = adep
                mock_data["fpl"]["fpl_base"][0]["ades"] = ades
                
                with open(flight_file, 'w') as f:
                    json.dump(mock_data, f)
            
            adapter = SCATAdapter(str(temp_path))
            summary = adapter.get_flight_summary()
            
            assert summary['total_files'] == 3
            assert set(summary['aircraft_types']) == {"A332", "A380", "B777"}
            assert "EDDF" in summary['airports']
            assert "EGLL" in summary['airports']
            assert "LFPG" in summary['airports']
            assert "KJFK" in summary['airports']
            assert set(summary['callsigns']) == {"AFR789", "BAW456", "DLH123"}
            assert summary['time_range']['earliest'] is not None
            assert summary['time_range']['latest'] is not None
    
    def test_convenience_function(self):
        """Test convenience function for loading SCAT scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a flight file
            flight_file = temp_path / "112200.json"
            mock_data = self.create_mock_scat_file("TEST001", track_points=5)
            with open(flight_file, 'w') as f:
                json.dump(mock_data, f)
            
            states = load_scat_scenario(str(temp_path), max_flights=1, time_window_minutes=60)
            
            assert len(states) == 5
            assert all(isinstance(state, AircraftState) for state in states)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed SCAT data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create file with missing required fields
            flight_file = temp_path / "112200.json"
            malformed_data = {"invalid": "data"}
            with open(flight_file, 'w') as f:
                json.dump(malformed_data, f)
            
            adapter = SCATAdapter(str(temp_path))
            record = adapter.load_flight_record(flight_file)
            
            # Should return None for malformed data
            assert record is None
    
    def test_timestamp_parsing(self):
        """Test various timestamp formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            adapter = SCATAdapter(str(temp_path))
            
            # Test standard format
            ts1 = adapter._parse_timestamp("2016-10-18T19:19:03.898437")
            assert ts1.year == 2016
            assert ts1.month == 10
            assert ts1.day == 18
            assert ts1.hour == 19
            assert ts1.minute == 19
            assert ts1.second == 3
            
            # Test format without microseconds
            ts2 = adapter._parse_timestamp("2016-10-18T19:19:03")
            assert ts2.microsecond == 0
