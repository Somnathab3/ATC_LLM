"""
Comprehensive test module for CDR schemas.
Tests validation, serialization, enums, and configuration defaults.
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict
import json
from pydantic import ValidationError

# Import all schemas to test
from src.cdr.schemas import (
    AircraftState, ConflictPrediction, ResolutionCommand, 
    ConfigurationSettings, LLMDetectionInput, LLMResolutionInput,
    ResolutionType, ResolutionEngine, ScenarioMetrics, 
    ConflictResolutionMetrics, PathComparisonMetrics, EnhancedReportingSystem
)


class TestResolutionType:
    """Test ResolutionType enum values and functionality."""
    
    def test_all_resolution_types_exist(self):
        """Test all expected resolution types are defined."""
        assert hasattr(ResolutionType, 'HEADING_CHANGE')
        assert hasattr(ResolutionType, 'SPEED_CHANGE')
        assert hasattr(ResolutionType, 'ALTITUDE_CHANGE')
        assert hasattr(ResolutionType, 'WAYPOINT_DIRECT')
        assert hasattr(ResolutionType, 'COMBINED')
    
    def test_waypoint_direct_enum_value(self):
        """Test new WAYPOINT_DIRECT enum value."""
        assert ResolutionType.WAYPOINT_DIRECT == "waypoint_direct"
        assert ResolutionType.WAYPOINT_DIRECT.value == "waypoint_direct"
    
    def test_resolution_type_string_conversion(self):
        """Test resolution types can be converted to/from strings."""
        for resolution_type in ResolutionType:
            # Test round-trip conversion using the value, not the str representation
            string_val = resolution_type.value
            recreated = ResolutionType(string_val)
            assert recreated == resolution_type
    
    def test_resolution_type_serialization(self):
        """Test resolution types serialize correctly in JSON."""
        data = {
            "type": ResolutionType.WAYPOINT_DIRECT,
            "types": [ResolutionType.HEADING_CHANGE, ResolutionType.ALTITUDE_CHANGE]
        }
        
        # Should be able to serialize
        json_str = json.dumps(data, default=str)
        assert "waypoint_direct" in json_str
        assert "heading_change" in json_str


class TestResolutionEngine:
    """Test ResolutionEngine enum values."""
    
    def test_all_engines_exist(self):
        """Test all expected engines are defined."""
        assert hasattr(ResolutionEngine, 'HORIZONTAL')
        assert hasattr(ResolutionEngine, 'VERTICAL')
        assert hasattr(ResolutionEngine, 'DETERMINISTIC')
        assert hasattr(ResolutionEngine, 'FALLBACK')
    
    def test_engine_values(self):
        """Test engine string values."""
        assert ResolutionEngine.HORIZONTAL == "horizontal"
        assert ResolutionEngine.VERTICAL == "vertical"
        assert ResolutionEngine.DETERMINISTIC == "deterministic"
        assert ResolutionEngine.FALLBACK == "fallback"


class TestConfigurationSettings:
    """Test configuration validation and defaults."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        # Create config with minimal required fields (all have defaults)
        config = ConfigurationSettings()
        
        # Test required defaults
        assert config.polling_interval_min == 5.0
        assert config.lookahead_time_min == 10.0
        assert config.min_horizontal_separation_nm == 5.0
        assert config.min_vertical_separation_ft == 1000.0
        assert config.llm_enabled is True
        assert config.llm_model_name == "llama3.1:8b"
        assert config.llm_temperature == 0.1
        assert config.llm_max_tokens == 2048
        assert config.safety_buffer_factor == 1.2
        assert config.max_resolution_angle_deg == 45.0
        assert config.max_altitude_change_ft == 2000.0
        
        # Test new waypoint field
        assert config.max_waypoint_diversion_nm == 80.0
        
        # Test enhanced validation fields
        assert config.enforce_ownship_only is True
        assert config.max_climb_rate_fpm == 3000.0
        assert config.max_descent_rate_fpm == 3000.0
        assert config.min_flight_level == 100
        assert config.max_flight_level == 600
        assert config.max_heading_change_deg == 90.0
        
        # Test dual LLM settings
        assert config.enable_dual_llm is True
        assert config.horizontal_retry_count == 2
        assert config.vertical_retry_count == 2
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = ConfigurationSettings(
            min_horizontal_separation_nm=3.0,
            max_waypoint_diversion_nm=120.0,
            llm_temperature=0.5,
            enforce_ownship_only=False,
            enable_dual_llm=False
        )
        
        assert config.min_horizontal_separation_nm == 3.0
        assert config.max_waypoint_diversion_nm == 120.0
        assert config.llm_temperature == 0.5
        assert config.enforce_ownship_only is False
        assert config.enable_dual_llm is False
    
    def test_configuration_validation_errors(self):
        """Test configuration validation catches invalid values."""
        
        # Negative separation should fail
        with pytest.raises(ValidationError):
            ConfigurationSettings(min_horizontal_separation_nm=-1.0)
        
        # Zero separation should fail
        with pytest.raises(ValidationError):
            ConfigurationSettings(min_vertical_separation_ft=0)
        
        # Temperature out of range
        with pytest.raises(ValidationError):
            ConfigurationSettings(llm_temperature=1.5)
        
        with pytest.raises(ValidationError):
            ConfigurationSettings(llm_temperature=-0.1)
        
        # Invalid max tokens
        with pytest.raises(ValidationError):
            ConfigurationSettings(llm_max_tokens=0)
        
        # Safety buffer too low
        with pytest.raises(ValidationError):
            ConfigurationSettings(safety_buffer_factor=0.9)
        
        # Invalid angles
        with pytest.raises(ValidationError):
            ConfigurationSettings(max_resolution_angle_deg=91.0)
        
        with pytest.raises(ValidationError):
            ConfigurationSettings(max_heading_change_deg=181.0)
        
        # Invalid flight levels
        with pytest.raises(ValidationError):
            ConfigurationSettings(min_flight_level=-10)
        
        with pytest.raises(ValidationError):
            ConfigurationSettings(max_flight_level=50)  # Less than min
        
        # Invalid retry counts
        with pytest.raises(ValidationError):
            ConfigurationSettings(horizontal_retry_count=0)
        
        with pytest.raises(ValidationError):
            ConfigurationSettings(vertical_retry_count=6)  # Too high
        
        # Negative waypoint diversion
        with pytest.raises(ValidationError):
            ConfigurationSettings(max_waypoint_diversion_nm=-5.0)
    
    def test_configuration_serialization(self):
        """Test configuration can be serialized and deserialized."""
        config = ConfigurationSettings(
            max_waypoint_diversion_nm=100.0,
            enable_dual_llm=False
        )
        
        # Serialize to dict
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict['max_waypoint_diversion_nm'] == 100.0
        assert config_dict['enable_dual_llm'] is False
        
        # Recreate from dict
        config_restored = ConfigurationSettings(**config_dict)
        assert config_restored.max_waypoint_diversion_nm == 100.0
        assert config_restored.enable_dual_llm is False
    
    def test_configuration_json_roundtrip(self):
        """Test configuration JSON serialization roundtrip."""
        config = ConfigurationSettings()
        
        # Convert to JSON string
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        
        # Parse back from JSON
        parsed_dict = json.loads(json_str)
        config_restored = ConfigurationSettings(**parsed_dict)
        
        # Should match original
        assert config_restored.model_dump() == config.model_dump()


class TestAircraftState:
    """Test AircraftState validation and functionality."""
    
    def test_valid_aircraft_state(self):
        """Test creation of valid aircraft state."""
        state = AircraftState(
            aircraft_id="TEST123",
            callsign="TEST123",
            latitude=40.7128,
            longitude=-74.0060,
            altitude_ft=35000,
            heading_deg=90,
            ground_speed_kt=450,
            vertical_speed_fpm=0,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert state.aircraft_id == "TEST123"
        assert state.latitude == 40.7128
        assert state.longitude == -74.0060
        assert state.altitude_ft == 35000
        assert state.heading_deg == 90
        assert state.ground_speed_kt == 450
    
    def test_aircraft_state_validation_errors(self):
        """Test aircraft state validation catches invalid values."""
        base_data = {
            "aircraft_id": "TEST123",
            "callsign": "TEST123",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "altitude_ft": 35000,
            "heading_deg": 90,
            "ground_speed_kt": 450,
            "vertical_speed_fpm": 0,
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Invalid latitude
        with pytest.raises(ValidationError):
            AircraftState(**{**base_data, "latitude": 91.0})
        
        with pytest.raises(ValidationError):
            AircraftState(**{**base_data, "latitude": -91.0})
        
        # Invalid longitude
        with pytest.raises(ValidationError):
            AircraftState(**{**base_data, "longitude": 181.0})
        
        with pytest.raises(ValidationError):
            AircraftState(**{**base_data, "longitude": -181.0})
        
        # Invalid heading
        with pytest.raises(ValidationError):
            AircraftState(**{**base_data, "heading_deg": 361.0})
        
        with pytest.raises(ValidationError):
            AircraftState(**{**base_data, "heading_deg": -1.0})
        
        # Negative speed
        with pytest.raises(ValidationError):
            AircraftState(**{**base_data, "ground_speed_kt": -10})


class TestResolutionCommand:
    """Test ResolutionCommand validation and waypoint functionality."""
    
    def test_waypoint_direct_command(self):
        """Test creation of WAYPOINT_DIRECT resolution command."""
        resolution = ResolutionCommand(
            resolution_id="WP_001",
            target_aircraft="TEST123",
            resolution_type=ResolutionType.WAYPOINT_DIRECT,
            source_engine=ResolutionEngine.HORIZONTAL,
            new_heading_deg=None,
            new_speed_kt=None,
            new_altitude_ft=None,
            waypoint_lat=41.0,
            waypoint_lon=-73.0,
            issue_time=datetime.now(timezone.utc),
            safety_margin_nm=6.0,
            is_validated=True,
            is_ownship_command=True
        )
        
        assert resolution.resolution_type == ResolutionType.WAYPOINT_DIRECT
        assert resolution.waypoint_lat == 41.0
        assert resolution.waypoint_lon == -73.0
        assert resolution.new_heading_deg is None
    
    def test_heading_change_command(self):
        """Test creation of heading change command."""
        resolution = ResolutionCommand(
            resolution_id="HDG_001",
            target_aircraft="TEST123",
            resolution_type=ResolutionType.HEADING_CHANGE,
            source_engine=ResolutionEngine.HORIZONTAL,
            new_heading_deg=120.0,
            issue_time=datetime.now(timezone.utc),
            safety_margin_nm=6.0,
            is_validated=True,
            is_ownship_command=True
        )
        
        assert resolution.resolution_type == ResolutionType.HEADING_CHANGE
        assert resolution.new_heading_deg == 120.0
        assert resolution.waypoint_lat is None
    
    def test_resolution_command_validation(self):
        """Test resolution command validation."""
        base_data = {
            "resolution_id": "TEST_001",
            "target_aircraft": "TEST123",
            "resolution_type": ResolutionType.HEADING_CHANGE,
            "source_engine": ResolutionEngine.HORIZONTAL,
            "issue_time": datetime.now(timezone.utc),
            "safety_margin_nm": 6.0,
            "is_validated": True,
            "is_ownship_command": True
        }
        
        # Valid heading
        resolution = ResolutionCommand(**{**base_data, "new_heading_deg": 180.0})
        assert resolution.new_heading_deg == 180.0
        
        # Invalid heading (negative)
        with pytest.raises(ValidationError):
            ResolutionCommand(**{**base_data, "new_heading_deg": -10.0})
        
        # Invalid heading (too large)
        with pytest.raises(ValidationError):
            ResolutionCommand(**{**base_data, "new_heading_deg": 370.0})
        
        # Valid speed
        resolution = ResolutionCommand(**{**base_data, "new_speed_kt": 250.0})
        assert resolution.new_speed_kt == 250.0
        
        # Invalid speed (negative)
        with pytest.raises(ValidationError):
            ResolutionCommand(**{**base_data, "new_speed_kt": -50.0})


class TestConflictPrediction:
    """Test ConflictPrediction validation and functionality."""
    
    def test_valid_conflict_prediction(self):
        """Test creation of valid conflict prediction."""
        prediction = ConflictPrediction(
            ownship_id="TEST123",
            intruder_id="INTR456",
            time_to_cpa_min=5.0,
            distance_at_cpa_nm=4.5,
            altitude_diff_ft=0.0,
            is_conflict=True,
            severity_score=0.85,
            conflict_type="horizontal",
            prediction_time=datetime.now(timezone.utc)
        )
        
        assert prediction.ownship_id == "TEST123"
        assert prediction.intruder_id == "INTR456"
        assert prediction.time_to_cpa_min == 5.0
        assert prediction.distance_at_cpa_nm == 4.5
        assert prediction.is_conflict is True
        assert prediction.severity_score == 0.85
    
    def test_conflict_prediction_validation(self):
        """Test conflict prediction validation."""
        base_data = {
            "ownship_id": "TEST123",
            "intruder_id": "INTR456",
            "time_to_cpa_min": 5.0,
            "distance_at_cpa_nm": 4.5,
            "altitude_diff_ft": 0.0,
            "is_conflict": True,
            "severity_score": 0.85,
            "conflict_type": "horizontal",
            "prediction_time": datetime.now(timezone.utc)
        }
        
        # Valid severity score
        prediction = ConflictPrediction(**base_data)
        assert prediction.severity_score == 0.85
        
        # Invalid severity score (negative)
        with pytest.raises(ValidationError):
            ConflictPrediction(**{**base_data, "severity_score": -0.1})
        
        # Invalid severity score (too large)
        with pytest.raises(ValidationError):
            ConflictPrediction(**{**base_data, "severity_score": 1.1})
        
        # Negative time to CPA
        with pytest.raises(ValidationError):
            ConflictPrediction(**{**base_data, "time_to_cpa_min": -1.0})
        
        # Negative distance
        with pytest.raises(ValidationError):
            ConflictPrediction(**{**base_data, "distance_at_cpa_nm": -1.0})


class TestLLMInputSchemas:
    """Test LLM input schema validation."""
    
    def test_llm_detection_input(self):
        """Test LLM detection input schema."""
        detection_input = LLMDetectionInput(
            ownship={
                "aircraft_id": "TEST123",
                "callsign": "TEST123",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "altitude_ft": 35000,
                "heading_deg": 90,
                "ground_speed_kt": 450,
                "vertical_speed_fpm": 0,
                "timestamp": datetime.now(timezone.utc)
            },
            intruders=[],
            separation_standards={
                "horizontal_nm": 5.0,
                "vertical_ft": 1000.0
            },
            context="Test detection scenario"
        )
        
        assert detection_input.ownship["aircraft_id"] == "TEST123"
        assert detection_input.separation_standards["horizontal_nm"] == 5.0
        assert detection_input.context == "Test detection scenario"
    
    def test_llm_resolution_input(self):
        """Test LLM resolution input schema."""
        resolution_input = LLMResolutionInput(
            ownship={
                "aircraft_id": "TEST123",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "altitude_ft": 35000,
                "heading_deg": 90,
                "ground_speed_kt": 450
            },
            intruder={
                "aircraft_id": "INTR456",
                "latitude": 40.7200,
                "longitude": -74.0000,
                "altitude_ft": 35000,
                "heading_deg": 270,
                "ground_speed_kt": 420
            },
            conflict_severity=0.85,
            time_to_conflict_min=5.0,
            available_maneuvers=["heading_change", "altitude_change"],
            constraints={"max_heading_change": 45.0},
            context="Test resolution scenario"
        )
        
        assert resolution_input.ownship["aircraft_id"] == "TEST123"
        assert resolution_input.intruder["aircraft_id"] == "INTR456"
        assert resolution_input.conflict_severity == 0.85
        assert "heading_change" in resolution_input.available_maneuvers


class TestEnhancedReportingSchemas:
    """Test enhanced reporting schemas."""
    
    def test_scenario_metrics(self):
        """Test scenario metrics schema."""
        metrics = ScenarioMetrics(
            scenario_id="SCEN_001",
            flight_id="FL123",
            total_conflicts=3,
            resolved_conflicts=2,
            safety_violations=0,
            avg_resolution_time_sec=45.5,
            ownship_path_similarity=0.92,
            scenario_success=True,
            completion_time=datetime.now(timezone.utc)
        )
        
        assert metrics.scenario_id == "SCEN_001"
        assert metrics.flight_id == "FL123"
        assert metrics.total_conflicts == 3
        assert metrics.resolved_conflicts == 2
        assert metrics.ownship_path_similarity == 0.92
    
    def test_conflict_resolution_metrics(self):
        """Test conflict resolution metrics schema."""
        metrics = ConflictResolutionMetrics(
            conflict_id="CONF_001",
            ownship_id="TEST123",
            intruder_id="INTR456",
            resolved=True,
            engine_used="horizontal",
            resolution_type="heading_change",
            waypoint_vs_heading="heading",
            time_to_action_sec=30.5,
            conflict_detection_time=datetime.now(timezone.utc),
            resolution_command_time=datetime.now(timezone.utc),
            initial_distance_nm=8.5,
            min_sep_nm=6.2,
            final_distance_nm=12.0,
            separation_violation=False,
            ownship_cross_track_error_nm=0.5,
            ownship_along_track_error_nm=2.1,
            path_deviation_total_nm=2.16,
            resolution_effectiveness=0.95,
            operational_impact=0.15
        )
        
        assert metrics.conflict_id == "CONF_001"
        assert metrics.resolved is True
        assert metrics.engine_used == "horizontal"
        assert metrics.resolution_type == "heading_change"
        assert metrics.waypoint_vs_heading == "heading"
        assert metrics.resolution_effectiveness == 0.95
    
    def test_path_comparison_metrics(self):
        """Test path comparison metrics schema."""
        metrics = PathComparisonMetrics(
            scenario_id="SCEN_001",
            aircraft_id="TEST123",
            baseline_path_length_nm=120.5,
            actual_path_length_nm=125.2,
            max_cross_track_error_nm=1.8,
            avg_cross_track_error_nm=0.6,
            path_efficiency_ratio=0.96,
            total_path_deviation_nm=4.7,
            path_similarity_score=0.94
        )
        
        assert metrics.scenario_id == "SCEN_001"
        assert metrics.aircraft_id == "TEST123"
        assert metrics.baseline_path_length_nm == 120.5
        assert metrics.path_efficiency_ratio == 0.96
        assert metrics.path_similarity_score == 0.94


class TestEnhancedReportingSystem:
    """Test enhanced reporting system functionality."""
    
    def test_reporting_system_initialization(self):
        """Test reporting system initialization."""
        reporter = EnhancedReportingSystem()
        
        assert len(reporter.scenario_metrics) == 0
        assert len(reporter.conflict_metrics) == 0
        assert len(reporter.path_comparisons) == 0
        assert reporter.output_dir.name == "reports"
    
    def test_custom_output_directory(self):
        """Test custom output directory."""
        reporter = EnhancedReportingSystem(output_dir="custom_reports")
        assert reporter.output_dir.name == "custom_reports"
    
    def test_add_metrics(self):
        """Test adding metrics to reporter."""
        reporter = EnhancedReportingSystem()
        
        # Add scenario metrics
        scenario = ScenarioMetrics(
            scenario_id="TEST_001",
            flight_id="FL123",
            total_conflicts=1,
            resolved_conflicts=1,
            safety_violations=0,
            avg_resolution_time_sec=30.0,
            ownship_path_similarity=0.95,
            scenario_success=True,
            completion_time=datetime.now(timezone.utc)
        )
        reporter.add_scenario_completion(scenario)
        assert len(reporter.scenario_metrics) == 1
        
        # Add conflict metrics
        conflict = ConflictResolutionMetrics(
            conflict_id="CONF_001",
            ownship_id="TEST123",
            intruder_id="INTR456",
            resolved=True,
            engine_used="horizontal",
            resolution_type="heading_change",
            waypoint_vs_heading="heading",
            time_to_action_sec=25.0,
            conflict_detection_time=datetime.now(timezone.utc),
            resolution_command_time=datetime.now(timezone.utc),
            initial_distance_nm=7.0,
            min_sep_nm=5.5,
            final_distance_nm=10.0,
            separation_violation=False,
            ownship_cross_track_error_nm=0.3,
            ownship_along_track_error_nm=1.2,
            path_deviation_total_nm=1.23,
            resolution_effectiveness=0.92,
            operational_impact=0.12
        )
        reporter.add_conflict_resolution(conflict)
        assert len(reporter.conflict_metrics) == 1
        
        # Add path comparison
        path_comp = PathComparisonMetrics(
            scenario_id="TEST_001",
            aircraft_id="TEST123",
            baseline_path_length_nm=100.0,
            actual_path_length_nm=102.0,
            max_cross_track_error_nm=1.5,
            avg_cross_track_error_nm=0.4,
            path_efficiency_ratio=0.98,
            total_path_deviation_nm=2.0,
            path_similarity_score=0.95
        )
        reporter.add_path_comparison(path_comp)
        assert len(reporter.path_comparisons) == 1


class TestSchemaIntegration:
    """Test schema integration and edge cases."""
    
    def test_waypoint_resolution_flow(self):
        """Test complete waypoint resolution workflow."""
        # Create configuration with waypoint settings
        config = ConfigurationSettings(
            max_waypoint_diversion_nm=150.0,
            enforce_ownship_only=True
        )
        
        # Create waypoint resolution command
        resolution = ResolutionCommand(
            resolution_id="WP_TEST_001",
            target_aircraft="TEST123",
            resolution_type=ResolutionType.WAYPOINT_DIRECT,
            source_engine=ResolutionEngine.HORIZONTAL,
            waypoint_lat=41.0,
            waypoint_lon=-73.5,
            issue_time=datetime.now(timezone.utc),
            safety_margin_nm=7.0,
            is_validated=True,
            is_ownship_command=True
        )
        
        # Verify resolution properties
        assert resolution.resolution_type == ResolutionType.WAYPOINT_DIRECT
        assert resolution.waypoint_lat == 41.0
        assert resolution.waypoint_lon == -73.5
        assert config.max_waypoint_diversion_nm == 150.0
    
    def test_dual_llm_configuration(self):
        """Test dual LLM configuration settings."""
        config = ConfigurationSettings(
            enable_dual_llm=True,
            horizontal_retry_count=3,
            vertical_retry_count=1
        )
        
        assert config.enable_dual_llm is True
        assert config.horizontal_retry_count == 3
        assert config.vertical_retry_count == 1
    
    def test_enhanced_validation_settings(self):
        """Test enhanced validation configuration."""
        config = ConfigurationSettings(
            enforce_ownship_only=False,
            max_climb_rate_fpm=2500.0,
            max_descent_rate_fpm=4000.0,
            min_flight_level=50,
            max_flight_level=700,
            max_heading_change_deg=120.0
        )
        
        assert config.enforce_ownship_only is False
        assert config.max_climb_rate_fpm == 2500.0
        assert config.max_descent_rate_fpm == 4000.0
        assert config.min_flight_level == 50
        assert config.max_flight_level == 700
        assert config.max_heading_change_deg == 120.0
    
    def test_schema_serialization_roundtrip(self):
        """Test complete serialization roundtrip for all major schemas."""
        # Test aircraft state
        aircraft = AircraftState(
            aircraft_id="TEST123",
            callsign="TEST123",
            latitude=40.7128,
            longitude=-74.0060,
            altitude_ft=35000,
            heading_deg=90,
            ground_speed_kt=450,
            vertical_speed_fpm=0,
            timestamp=datetime.now(timezone.utc)
        )
        
        aircraft_dict = aircraft.model_dump()
        aircraft_restored = AircraftState(**aircraft_dict)
        assert aircraft_restored.aircraft_id == aircraft.aircraft_id
        
        # Test configuration
        config = ConfigurationSettings(max_waypoint_diversion_nm=200.0)
        config_dict = config.model_dump()
        config_restored = ConfigurationSettings(**config_dict)
        assert config_restored.max_waypoint_diversion_nm == 200.0
        
        # Test resolution command
        resolution = ResolutionCommand(
            resolution_id="TEST_001",
            target_aircraft="TEST123",
            resolution_type=ResolutionType.WAYPOINT_DIRECT,
            source_engine=ResolutionEngine.HORIZONTAL,
            waypoint_lat=41.0,
            waypoint_lon=-73.0,
            issue_time=datetime.now(timezone.utc),
            safety_margin_nm=6.0,
            is_validated=True,
            is_ownship_command=True
        )
        
        resolution_dict = resolution.model_dump()
        resolution_restored = ResolutionCommand(**resolution_dict)
        assert resolution_restored.resolution_type == ResolutionType.WAYPOINT_DIRECT


if __name__ == "__main__":
    pytest.main([__file__])
