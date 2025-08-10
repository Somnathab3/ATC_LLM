"""
Enhanced comprehensive test module for CDR schemas.
Tests validation, serialization, enums, and configuration defaults.
Targets lines 447-572 in schemas.py for 95%+ coverage.
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict
import json
from pydantic import ValidationError
from pathlib import Path
import tempfile
import csv

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
            # Test round-trip conversion using the value
            string_val = resolution_type.value
            recreated = ResolutionType(string_val)
            assert recreated == resolution_type
    
    def test_resolution_type_serialization(self):
        """Test resolution types serialize correctly in JSON."""
        data = {
            "type": ResolutionType.WAYPOINT_DIRECT.value,
            "types": [ResolutionType.HEADING_CHANGE.value, ResolutionType.ALTITUDE_CHANGE.value]
        }
        
        # Should be able to serialize
        json_str = json.dumps(data)
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
        # Should work with no arguments due to Field defaults
        config = ConfigurationSettings()
        
        # Test timing defaults
        assert config.polling_interval_min == 5.0
        assert config.lookahead_time_min == 10.0
        assert config.snapshot_interval_min == 1.0
        
        # Test prompt builder defaults
        assert config.max_intruders_in_prompt == 5
        assert config.intruder_proximity_nm == 100.0
        assert config.intruder_altitude_diff_ft == 5000.0
        assert config.trend_analysis_window_min == 2.0
        
        # Test separation standards
        assert config.min_horizontal_separation_nm == 5.0
        assert config.min_vertical_separation_ft == 1000.0
        
        # Test LLM defaults
        assert config.llm_enabled is True
        assert config.llm_model_name == "llama3.1:8b"
        assert config.llm_temperature == 0.1
        assert config.llm_max_tokens == 2048
        
        # Test safety defaults
        assert config.safety_buffer_factor == 1.2
        assert config.max_resolution_angle_deg == 45.0
        assert config.max_altitude_change_ft == 2000.0
        assert config.max_waypoint_diversion_nm == 80.0
        
        # Test enhanced validation defaults
        assert config.enforce_ownship_only is True
        assert config.max_climb_rate_fpm == 3000.0
        assert config.max_descent_rate_fpm == 3000.0
        assert config.min_flight_level == 100
        assert config.max_flight_level == 600
        assert config.max_heading_change_deg == 90.0
        
        # Test dual LLM defaults
        assert config.enable_dual_llm is True
        assert config.horizontal_retry_count == 2
        assert config.vertical_retry_count == 2
        
        # Test BlueSky defaults
        assert config.bluesky_host == "localhost"
        assert config.bluesky_port == 1337
        assert config.bluesky_timeout_sec == 5.0
        
        # Test simulation defaults
        assert config.fast_time is True
        assert config.sim_accel_factor == 1.0
    
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
        
        # Test various validation errors
        with pytest.raises(ValidationError, match="greater than 0"):
            ConfigurationSettings(min_horizontal_separation_nm=-1.0)
        
        with pytest.raises(ValidationError, match="greater than 0"):
            ConfigurationSettings(min_vertical_separation_ft=0)
        
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            ConfigurationSettings(llm_temperature=1.5)
        
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ConfigurationSettings(llm_temperature=-0.1)
        
        with pytest.raises(ValidationError, match="greater than 0"):
            ConfigurationSettings(llm_max_tokens=0)
        
        with pytest.raises(ValidationError, match="greater than 1"):
            ConfigurationSettings(safety_buffer_factor=0.9)
        
        with pytest.raises(ValidationError, match="less than or equal to 90"):
            ConfigurationSettings(max_resolution_angle_deg=91.0)
        
        with pytest.raises(ValidationError, match="less than or equal to 180"):
            ConfigurationSettings(max_heading_change_deg=181.0)
        
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ConfigurationSettings(min_flight_level=-10)
        
        with pytest.raises(ValidationError, match="greater than or equal to 100"):
            ConfigurationSettings(max_flight_level=50)  # Less than minimum required
        
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ConfigurationSettings(horizontal_retry_count=0)
        
        with pytest.raises(ValidationError, match="less than or equal to 5"):
            ConfigurationSettings(vertical_retry_count=6)  # Too high
        
        with pytest.raises(ValidationError, match="greater than 0"):
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
            timestamp=datetime.now(timezone.utc),
            latitude=40.7128,
            longitude=-74.0060,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0,
            callsign="TEST123"
        )
        
        assert state.aircraft_id == "TEST123"
        assert state.latitude == 40.7128
        assert state.longitude == -74.0060
        assert state.altitude_ft == 35000
        assert state.heading_deg == 90
        assert state.ground_speed_kt == 450
        assert state.spawn_offset_min == 0.0  # Default value
    
    def test_aircraft_state_validation_errors(self):
        """Test aircraft state validation catches invalid values."""
        base_data = {
            "aircraft_id": "TEST123",
            "timestamp": datetime.now(timezone.utc),
            "latitude": 40.7128,
            "longitude": -74.0060,
            "altitude_ft": 35000,
            "ground_speed_kt": 450,
            "heading_deg": 90,
            "vertical_speed_fpm": 0
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
        resolution = ResolutionCommand(**{**base_data, "new_heading_deg": 120.0})
        assert resolution.new_heading_deg == 120.0
        
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


class TestEnhancedReportingSystem:
    """Test enhanced reporting system functionality - covers lines 447-572."""
    
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
            conflicts_resolved=1,
            resolution_success_rate=1.0,
            scenario_duration_min=10.0,
            avg_time_to_action_sec=30.0,
            min_separation_achieved_nm=5.5,
            safety_violations=0,
            separation_standards_maintained=True,
            path_efficiency_score=0.95,
            total_path_deviation_nm=2.0,
            scenario_success=True
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
    
    def test_csv_report_generation(self):
        """Test CSV report generation functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = EnhancedReportingSystem(output_dir=tmpdir)
            
            # Add test data
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
            
            conflict = ConflictResolutionMetrics(
                conflict_id="TEST_001_CONF_001",  # Starts with scenario_id
                ownship_id="TEST123",
                intruder_id="INTR456",
                resolved=True,
                engine_used="horizontal",
                resolution_type="waypoint_direct",
                waypoint_vs_heading="waypoint",
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
            
            # Generate CSV report
            csv_path = reporter.generate_csv_report("test_report.csv")
            
            # Verify file was created
            assert Path(csv_path).exists()
            
            # Verify content
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                assert len(rows) == 1
                
                row = rows[0]
                assert row['scenario_id'] == "TEST_001"
                assert row['conflict_id'] == "TEST_001_CONF_001"
                assert row['resolved'] == 'Y'
                assert row['engine_used'] == "horizontal"
                assert row['resolution_type'] == "waypoint_direct"
                assert row['waypoint_vs_heading'] == "waypoint"
    
    def test_csv_report_orphaned_conflict(self):
        """Test CSV report with conflict not matching any scenario."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = EnhancedReportingSystem(output_dir=tmpdir)
            
            # Add conflict without matching scenario
            conflict = ConflictResolutionMetrics(
                conflict_id="ORPHAN_CONF_001",
                ownship_id="TEST123",
                intruder_id="INTR456",
                resolved=False,
                engine_used="fallback",
                resolution_type="combined",
                waypoint_vs_heading="heading",
                time_to_action_sec=45.0,
                conflict_detection_time=datetime.now(timezone.utc),
                resolution_command_time=datetime.now(timezone.utc),
                initial_distance_nm=3.0,
                min_sep_nm=2.5,
                final_distance_nm=4.0,
                separation_violation=True,
                ownship_cross_track_error_nm=1.5,
                ownship_along_track_error_nm=2.8,
                path_deviation_total_nm=3.2,
                resolution_effectiveness=0.45,
                operational_impact=0.78
            )
            reporter.add_conflict_resolution(conflict)
            
            # Generate CSV report
            csv_path = reporter.generate_csv_report("orphan_test.csv")
            
            # Verify content handles orphaned conflict
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                assert len(rows) == 1
                
                row = rows[0]
                assert row['scenario_id'] == "unknown"
                assert row['flight_id'] == "unknown"
                assert row['resolved'] == 'N'
                assert row['separation_violation'] == 'Y'
    
    def test_json_report_generation(self):
        """Test JSON report generation functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = EnhancedReportingSystem(output_dir=tmpdir)
            
            # Add test data
            scenario = ScenarioMetrics(
                scenario_id="JSON_001",
                flight_id="FL456",
                total_conflicts=2,
                resolved_conflicts=1,
                safety_violations=1,
                avg_resolution_time_sec=35.5,
                ownship_path_similarity=0.88,
                scenario_success=False,
                completion_time=datetime.now(timezone.utc)
            )
            reporter.add_scenario_completion(scenario)
            
            # Generate JSON report
            json_path = reporter.generate_json_report("test_report.json")
            
            # Verify file was created
            assert Path(json_path).exists()
            
            # Verify content
            with open(json_path, 'r') as jsonfile:
                data = json.load(jsonfile)
                
                assert 'metadata' in data
                assert 'summary_statistics' in data
                assert 'scenario_details' in data
                assert 'conflict_details' in data
                assert 'path_comparisons' in data
                
                metadata = data['metadata']
                assert metadata['total_scenarios'] == 1
                assert metadata['total_conflicts'] == 0  # No conflicts added
                assert metadata['total_path_comparisons'] == 0
    
    def test_summary_statistics_calculation(self):
        """Test summary statistics calculation."""
        reporter = EnhancedReportingSystem()
        
        # Test with no conflicts
        stats = reporter._calculate_summary_statistics()
        assert stats == {}
        
        # Add multiple conflicts with different engines
        conflicts = [
            ConflictResolutionMetrics(
                conflict_id="CONF_001",
                ownship_id="TEST123",
                intruder_id="INTR456",
                resolved=True,
                engine_used="horizontal",
                resolution_type="heading_change",
                waypoint_vs_heading="heading",
                time_to_action_sec=20.0,
                conflict_detection_time=datetime.now(timezone.utc),
                resolution_command_time=datetime.now(timezone.utc),
                initial_distance_nm=8.0,
                min_sep_nm=6.0,
                final_distance_nm=12.0,
                separation_violation=False,
                ownship_cross_track_error_nm=0.2,
                ownship_along_track_error_nm=0.8,
                path_deviation_total_nm=1.0,
                resolution_effectiveness=0.95,
                operational_impact=0.10
            ),
            ConflictResolutionMetrics(
                conflict_id="CONF_002",
                ownship_id="TEST456",
                intruder_id="INTR789",
                resolved=False,
                engine_used="vertical",
                resolution_type="altitude_change",
                waypoint_vs_heading="heading",
                time_to_action_sec=40.0,
                conflict_detection_time=datetime.now(timezone.utc),
                resolution_command_time=datetime.now(timezone.utc),
                initial_distance_nm=5.0,
                min_sep_nm=3.0,
                final_distance_nm=4.0,
                separation_violation=True,
                ownship_cross_track_error_nm=1.0,
                ownship_along_track_error_nm=2.0,
                path_deviation_total_nm=3.0,
                resolution_effectiveness=0.60,
                operational_impact=0.40
            )
        ]
        
        for conflict in conflicts:
            reporter.add_conflict_resolution(conflict)
        
        stats = reporter._calculate_summary_statistics()
        
        assert stats['overall_success_rate'] == 50.0  # 1 of 2 resolved
        assert stats['average_time_to_action_sec'] == 30.0  # (20 + 40) / 2
        assert stats['average_min_separation_nm'] == 4.5  # (6 + 3) / 2
        assert stats['separation_violations'] == 1
        assert stats['engine_usage']['horizontal'] == 1
        assert stats['engine_usage']['vertical'] == 1
        assert stats['engine_usage']['deterministic'] == 0
        assert stats['engine_usage']['fallback'] == 0
        assert stats['resolution_types']['heading_change'] == 1
        assert stats['resolution_types']['altitude_change'] == 1
        assert stats['resolution_types']['speed_change'] == 0
        assert stats['resolution_types']['combined'] == 0
        assert stats['average_path_deviation_nm'] == 2.0  # (1 + 3) / 2
        assert stats['average_resolution_effectiveness'] == 0.775  # (0.95 + 0.60) / 2


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
            timestamp=datetime.now(timezone.utc),
            latitude=40.7128,
            longitude=-74.0060,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0,
            callsign="TEST123"
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
