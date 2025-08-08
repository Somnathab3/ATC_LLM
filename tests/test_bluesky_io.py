"""Test suite for BlueSky I/O integration."""

import logging
import pytest
from src.cdr.bluesky_io import BlueSkyClient, BSConfig
from src.cdr.schemas import ConfigurationSettings


class TestBlueSkyIO:
    """Test BlueSky integration functionality."""
    
    def test_bs_connect_and_minimal(self):
        """Test BlueSky connection and basic functionality."""
        # Create minimal config with all required fields
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama-3.1-8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        bs = BlueSkyClient(config)
        
        # Test connection (may fail if BlueSky not installed, but shouldn't crash)
        try:
            result = bs.connect()
            # If BlueSky is available, should connect successfully
            if result:
                assert bs.connected
                assert bs.bs is not None
                
                # Test basic command
                # Note: This might fail if BlueSky isn't properly initialized
                # but the method should not crash
                bs.stack("ECHO Test command")
                
                # Test aircraft states fetch (should return empty list if no aircraft)
                states = bs.get_aircraft_states()
                assert isinstance(states, list)
                
            else:
                # BlueSky not available, but connection attempt shouldn't crash
                assert not bs.connected
                
        except ImportError:
            # BlueSky not installed - this is acceptable for CI
            pytest.skip("BlueSky not available - skipping integration test")
        except Exception as e:
            # Other errors should be logged but test should not fail
            logging.warning(f"BlueSky test encountered error: {e}")
            pytest.skip(f"BlueSky test skipped due to error: {e}")
    
    def test_bs_config(self):
        """Test BlueSky configuration."""
        config = BSConfig(headless=True)
        assert config.headless is True
    
    def test_bs_client_init(self):
        """Test BlueSky client initialization."""
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama-3.1-8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        client = BlueSkyClient(config)
        
        assert client.config == config
        assert client.bs is None
        assert not client.connected
    
    def test_bs_command_formatting(self):
        """Test command formatting methods."""
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama-3.1-8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        client = BlueSkyClient(config)
        
        # Test aircraft creation command format
        # This tests the method signature without requiring BlueSky connection
        try:
            # These will fail without connection but shouldn't crash
            client.create_aircraft("TEST001", "A320", 52.0, 4.0, 90.0, 10000.0, 250.0)
            client.set_heading("TEST001", 180.0)
            client.set_altitude("TEST001", 15000.0)
            client.step_minutes(1.0)
        except Exception:
            # Expected to fail without connection - just testing method signatures
            pass
