"""Smoke tests for CDR pipeline integration."""

import pytest
from datetime import datetime
from src.cdr.pipeline import CDRPipeline
from src.cdr.schemas import ConfigurationSettings, AircraftState


class TestPipelineSmoke:
    """Smoke tests for CDR pipeline functionality."""
    
    def test_pipeline_initialization(self):
        """Test that pipeline can be initialized with default config."""
        config = ConfigurationSettings()
        pipeline = CDRPipeline(config)
        
        # Verify pipeline is properly initialized
        assert pipeline is not None
        assert pipeline.config == config
        assert pipeline.running is False
        assert pipeline.cycle_count == 0
        assert hasattr(pipeline, 'bluesky_client')
        assert hasattr(pipeline, 'llm_client')
        assert hasattr(pipeline, 'metrics')
    
    def test_pipeline_initialization_custom_config(self):
        """Test pipeline initialization with custom configuration."""
        config = ConfigurationSettings(
            polling_interval_min=3.0,
            lookahead_time_min=15.0,
            min_horizontal_separation_nm=6.0,
            bluesky_port=1338
        )
        
        pipeline = CDRPipeline(config)
        
        # Verify custom config is applied
        assert pipeline.config.polling_interval_min == 3.0
        assert pipeline.config.lookahead_time_min == 15.0
        assert pipeline.config.min_horizontal_separation_nm == 6.0
        assert pipeline.config.bluesky_port == 1338
    
    def test_pipeline_components_exist(self):
        """Test that all required pipeline components are present."""
        config = ConfigurationSettings()
        pipeline = CDRPipeline(config)
        
        # Verify all required components are initialized
        assert hasattr(pipeline, 'bluesky_client')
        assert hasattr(pipeline, 'llm_client')
        assert hasattr(pipeline, 'metrics')
        assert hasattr(pipeline, 'active_resolutions')
        assert hasattr(pipeline, 'conflict_history')
        
        # Verify components have expected types
        from src.cdr.bluesky_io import BlueSkyClient
        from src.cdr.llm_client import LlamaClient
        from src.cdr.metrics import MetricsCollector
        
        assert isinstance(pipeline.bluesky_client, BlueSkyClient)
        assert isinstance(pipeline.llm_client, LlamaClient)
        assert isinstance(pipeline.metrics, MetricsCollector)
        assert isinstance(pipeline.active_resolutions, dict)
        assert isinstance(pipeline.conflict_history, list)
    
    def test_pipeline_state_tracking(self):
        """Test pipeline state tracking initialization."""
        config = ConfigurationSettings()
        pipeline = CDRPipeline(config)
        
        # Verify initial state
        assert len(pipeline.active_resolutions) == 0
        assert len(pipeline.conflict_history) == 0
        assert pipeline.cycle_count == 0
        assert not pipeline.running
    
    def test_pipeline_stop_when_not_running(self):
        """Test stopping pipeline when it's not running."""
        config = ConfigurationSettings()
        pipeline = CDRPipeline(config)
        
        # Should handle stop gracefully even when not running
        pipeline.stop()
        assert not pipeline.running
    
    def test_pipeline_find_ownship(self):
        """Test ownship identification in aircraft list."""
        config = ConfigurationSettings()
        pipeline = CDRPipeline(config)
        
        # Create test aircraft states
        aircraft_states = [
            AircraftState(
                aircraft_id="TRAFFIC1",
                timestamp=datetime.now(),
                latitude=59.3,
                longitude=18.1,
                altitude_ft=35000,
                ground_speed_kt=450,
                heading_deg=90,
                vertical_speed_fpm=0
            ),
            AircraftState(
                aircraft_id="OWNSHIP",
                timestamp=datetime.now(),
                latitude=59.4,
                longitude=18.2,
                altitude_ft=36000,
                ground_speed_kt=400,
                heading_deg=180,
                vertical_speed_fpm=0
            ),
            AircraftState(
                aircraft_id="TRAFFIC2",
                timestamp=datetime.now(),
                latitude=59.5,
                longitude=18.3,
                altitude_ft=34000,
                ground_speed_kt=480,
                heading_deg=270,
                vertical_speed_fpm=0
            )
        ]
        
        # Test finding ownship
        ownship = pipeline._find_ownship(aircraft_states, "OWNSHIP")
        assert ownship is not None
        assert ownship.aircraft_id == "OWNSHIP"
        assert ownship.altitude_ft == 36000
        
        # Test with non-existent ownship
        missing_ownship = pipeline._find_ownship(aircraft_states, "MISSING")
        assert missing_ownship is None
    
    def test_pipeline_cleanup(self):
        """Test pipeline cleanup functionality."""
        config = ConfigurationSettings()
        pipeline = CDRPipeline(config)
        
        # Should handle cleanup gracefully
        pipeline._cleanup()
        
        # Verify cleanup doesn't crash
        assert True  # If we get here, cleanup worked


class TestPipelineConfiguration:
    """Test pipeline configuration handling."""
    
    def test_default_configuration_values(self):
        """Test that default configuration has reasonable values."""
        config = ConfigurationSettings()
        
        # Verify timing settings
        assert config.polling_interval_min == 5.0
        assert config.lookahead_time_min == 10.0
        
        # Verify separation standards
        assert config.min_horizontal_separation_nm == 5.0
        assert config.min_vertical_separation_ft == 1000.0
        
        # Verify LLM settings
        assert config.llm_model_name == "llama-3.1-8b"
        assert 0 <= config.llm_temperature <= 1
        assert config.llm_max_tokens > 0
        
        # Verify safety settings
        assert config.safety_buffer_factor > 1.0
        assert 0 < config.max_resolution_angle_deg <= 90
        assert config.max_altitude_change_ft > 0
        
        # Verify BlueSky settings
        assert config.bluesky_host == "localhost"
        assert 1 <= config.bluesky_port <= 65535
        assert config.bluesky_timeout_sec > 0
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Test valid configuration
        valid_config = ConfigurationSettings(
            polling_interval_min=3.0,
            lookahead_time_min=8.0,
            min_horizontal_separation_nm=4.0,
            min_vertical_separation_ft=800.0
        )
        
        assert valid_config.polling_interval_min == 3.0
        assert valid_config.lookahead_time_min == 8.0
        
        # Test invalid configurations should raise validation errors
        with pytest.raises(Exception):  # Pydantic ValidationError
            ConfigurationSettings(polling_interval_min=-1.0)  # Negative value
        
        with pytest.raises(Exception):
            ConfigurationSettings(bluesky_port=99999)  # Port out of range
        
        with pytest.raises(Exception):
            ConfigurationSettings(llm_temperature=2.0)  # Temperature > 1


class TestPipelineIntegration:
    """Integration tests for pipeline components."""
    
    def test_pipeline_component_integration(self):
        """Test that pipeline components can work together."""
        config = ConfigurationSettings()
        pipeline = CDRPipeline(config)
        
        # Test that components can interact without errors
        # (Most functionality not implemented in Sprint 0, but structure should work)
        
        # Verify BlueSky client is configured correctly
        assert pipeline.bluesky_client.host == config.bluesky_host
        assert pipeline.bluesky_client.port == config.bluesky_port
        
        # Verify LLM client is configured correctly
        assert pipeline.llm_client.config == config
        
        # Verify metrics collector is initialized
        assert hasattr(pipeline.metrics, 'reset')
        assert hasattr(pipeline.metrics, 'generate_summary')
    
    def test_pipeline_execution_cycle_structure(self):
        """Test the structure of a pipeline execution cycle."""
        config = ConfigurationSettings()
        pipeline = CDRPipeline(config)
        
        # Verify pipeline has the methods needed for execution cycle
        assert hasattr(pipeline, '_execute_cycle')
        assert hasattr(pipeline, '_fetch_aircraft_states')
        assert hasattr(pipeline, '_find_ownship')
        assert hasattr(pipeline, '_predict_conflicts')
        assert hasattr(pipeline, '_handle_conflict')
        assert hasattr(pipeline, '_generate_resolution')
        assert hasattr(pipeline, '_validate_and_execute_resolution')
        assert hasattr(pipeline, '_update_metrics')
        
        # These methods may not be fully implemented yet, but should exist
        assert callable(getattr(pipeline, '_execute_cycle'))
        assert callable(getattr(pipeline, '_fetch_aircraft_states'))
        assert callable(getattr(pipeline, '_find_ownship'))


class TestPipelineMetrics:
    """Test pipeline metrics integration."""
    
    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        config = ConfigurationSettings()
        pipeline = CDRPipeline(config)
        
        # Verify metrics collector is ready
        assert hasattr(pipeline.metrics, 'start_time')
        assert hasattr(pipeline.metrics, 'cycle_times')
        assert hasattr(pipeline.metrics, 'conflicts_detected')
        assert hasattr(pipeline.metrics, 'resolutions_issued')
    
    def test_metrics_summary_generation(self):
        """Test metrics summary generation."""
        config = ConfigurationSettings()
        pipeline = CDRPipeline(config)
        
        # Should be able to generate summary even with no data
        summary = pipeline.metrics.generate_summary()
        
        assert hasattr(summary, 'total_simulation_time_min')
        assert hasattr(summary, 'total_cycles')
        assert hasattr(summary, 'total_conflicts_detected')
        assert hasattr(summary, 'total_resolutions_issued')
        
        # Initial values should be reasonable
        assert summary.total_cycles == 0
        assert summary.total_conflicts_detected == 0
        assert summary.total_resolutions_issued == 0


# Main entry point test
def test_pipeline_main_entry_point():
    """Test that pipeline main entry point exists and can be imported."""
    from src.cdr.pipeline import main
    
    # Verify main function exists
    assert callable(main)
    
    # Main function should handle being called (though it may not run in test environment)
    # We won't actually call it to avoid starting the pipeline
