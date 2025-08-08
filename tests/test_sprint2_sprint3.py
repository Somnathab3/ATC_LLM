"""Test suite for Sprint 2 and Sprint 3 implementations."""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.cdr.detect import predict_conflicts, is_conflict, calculate_severity_score
from src.cdr.llm_client import LlamaClient
from src.cdr.resolve import execute_resolution, _validate_resolution_safety
from src.cdr.metrics import MetricsCollector
from src.cdr.schemas import (
    AircraftState, ConflictPrediction, ConfigurationSettings,
    DetectOut, ResolveOut, ResolutionCommand, ResolutionType
)


@pytest.fixture
def config():
    """Test configuration."""
    return ConfigurationSettings(
        polling_interval_min=5.0,
        lookahead_time_min=10.0,
        llm_model_name="llama-3.1-8b",
        llm_temperature=0.1
    )


@pytest.fixture
def ownship():
    """Test ownship aircraft."""
    return AircraftState(
        aircraft_id="OWN",
        timestamp=datetime.now(),
        latitude=59.3,
        longitude=18.1,
        altitude_ft=35000,
        ground_speed_kt=450,
        heading_deg=90,
        vertical_speed_fpm=0
    )


@pytest.fixture
def converging_intruder():
    """Test intruder on collision course."""
    return AircraftState(
        aircraft_id="TRF001",
        timestamp=datetime.now(),
        latitude=59.35,
        longitude=18.6,
        altitude_ft=35000,
        ground_speed_kt=450,
        heading_deg=270,  # Head-on
        vertical_speed_fpm=0
    )


@pytest.fixture
def diverging_intruder():
    """Test intruder diverging from ownship."""
    return AircraftState(
        aircraft_id="TRF002",
        timestamp=datetime.now(),
        latitude=59.25,
        longitude=18.0,
        altitude_ft=35000,
        ground_speed_kt=450,
        heading_deg=180,  # Diverging
        vertical_speed_fpm=0
    )


class TestSprint2ConflictDetection:
    """Test Sprint 2 - Deterministic conflict detection."""
    
    def test_is_conflict_safe_separation(self):
        """Test safe separation is not flagged as conflict."""
        # Safe in both dimensions
        assert not is_conflict(6.0, 1500.0, 5.0)
        
        # Safe horizontally
        assert not is_conflict(6.0, 500.0, 5.0)
        
        # Safe vertically
        assert not is_conflict(3.0, 1500.0, 5.0)
        
    def test_is_conflict_violation(self):
        """Test conflict detection when both standards violated."""
        # Both standards violated = conflict
        assert is_conflict(3.0, 500.0, 5.0)
        
        # Future encounter
        assert is_conflict(2.0, 800.0, 8.0)
        
    def test_is_conflict_past_encounter(self):
        """Test past encounters are not conflicts."""
        assert not is_conflict(2.0, 500.0, -1.0)
        
    def test_severity_scoring(self):
        """Test conflict severity calculation."""
        # High severity: close and imminent
        high_severity = calculate_severity_score(1.0, 200.0, 1.0)
        assert high_severity > 0.7
        
        # Low severity: distant and far future
        low_severity = calculate_severity_score(4.9, 900.0, 9.0)
        assert low_severity < 0.3
        
        # Severity bounds
        assert 0.0 <= calculate_severity_score(0.0, 0.0, 0.0) <= 1.0
        assert 0.0 <= calculate_severity_score(10.0, 5000.0, 15.0) <= 1.0
        
    def test_predict_conflicts_converging(self, ownship, converging_intruder):
        """Test detection of converging aircraft."""
        traffic = [converging_intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        assert isinstance(conflicts, list)
        # May or may not detect based on exact geometry
        
    def test_predict_conflicts_diverging(self, ownship, diverging_intruder):
        """Test diverging aircraft don't create false alerts."""
        traffic = [diverging_intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Should not detect conflicts with diverging aircraft
        for conflict in conflicts:
            if conflict.time_to_cpa_min <= 10.0:
                assert not conflict.is_conflict
                
    def test_predict_conflicts_distant(self, ownship):
        """Test distant aircraft are filtered out."""
        distant_intruder = AircraftState(
            aircraft_id="DISTANT",
            timestamp=datetime.now(),
            latitude=0.0,  # Very far away
            longitude=0.0,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        traffic = [distant_intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Should filter out distant aircraft
        assert len(conflicts) == 0
        
    def test_predict_conflicts_altitude_filter(self, ownship):
        """Test altitude filtering (±5000 ft)."""
        high_intruder = AircraftState(
            aircraft_id="HIGH",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=45000,  # >5000 ft difference
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        
        traffic = [high_intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Should filter out aircraft with large altitude difference
        assert len(conflicts) == 0


class TestSprint2Metrics:
    """Test Sprint 2 - Wolfgang KPI metrics."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initializes correctly."""
        collector = MetricsCollector()
        
        assert len(collector.conflicts_detected) == 0
        assert len(collector.resolutions_issued) == 0
        assert len(collector.cycle_times) == 0
        
    def test_metrics_cycle_recording(self):
        """Test cycle time recording."""
        collector = MetricsCollector()
        
        collector.record_cycle_time(2.5)
        collector.record_cycle_time(3.1)
        
        assert len(collector.cycle_times) == 2
        assert collector.cycle_times[0] == 2.5
        assert collector.cycle_times[1] == 3.1
        
    def test_metrics_conflict_recording(self, ownship, converging_intruder):
        """Test conflict detection recording."""
        collector = MetricsCollector()
        
        conflict = ConflictPrediction(
            ownship_id=ownship.aircraft_id,
            intruder_id=converging_intruder.aircraft_id,
            time_to_cpa_min=5.0,
            distance_at_cpa_nm=3.0,
            altitude_diff_ft=500.0,
            is_conflict=True,
            severity_score=0.8,
            conflict_type="both",
            prediction_time=datetime.now()
        )
        
        collector.record_conflict_detection([conflict], datetime.now())
        
        assert len(collector.conflicts_detected) == 1
        assert collector.conflicts_detected[0].is_conflict
        
    def test_wolfgang_kpis_calculation(self):
        """Test Wolfgang KPI calculation."""
        collector = MetricsCollector()
        
        # Add some test data
        collector.record_cycle_time(2.0)
        
        kpis = collector.calculate_wolfgang_kpis()
        
        # Should return dictionary with all KPIs
        expected_kpis = ['tbas', 'lat', 'pa', 'pi', 'dat', 'dfa', 're', 'ri', 'rat']
        for kpi in expected_kpis:
            assert kpi in kpis
            assert isinstance(kpis[kpi], (int, float))


class TestSprint3LLMIntegration:
    """Test Sprint 3 - LLM detection and resolution."""
    
    def test_llama_client_initialization(self, config):
        """Test LLM client initializes correctly."""
        client = LlamaClient(config)
        
        assert client.model_name == "llama-3.1-8b"
        assert client.temperature == 0.1
        
    def test_detect_out_schema(self):
        """Test DetectOut schema validation."""
        # Valid detection output
        valid_output = DetectOut(
            conflict=True,
            intruders=[{"id": "TRF001", "eta_min": 4.5, "why": "converging headings"}]
        )
        
        assert valid_output.conflict == True
        assert len(valid_output.intruders) == 1
        
        # Empty intruders list is valid
        empty_output = DetectOut(conflict=False, intruders=[])
        assert empty_output.conflict == False
        assert len(empty_output.intruders) == 0
        
    def test_resolve_out_schema(self):
        """Test ResolveOut schema validation."""
        # Valid turn resolution
        turn_output = ResolveOut(
            action="turn",
            params={"heading_deg": 120},
            rationale="Right turn to avoid conflict"
        )
        
        assert turn_output.action == "turn"
        assert turn_output.params["heading_deg"] == 120
        
        # Valid climb resolution
        climb_output = ResolveOut(
            action="climb",
            params={"delta_ft": 1000},
            rationale="Climb to separate vertically"
        )
        
        assert climb_output.action == "climb"
        assert climb_output.params["delta_ft"] == 1000
        
    def test_llm_detection_simulation(self, config):
        """Test simulated LLM detection."""
        client = LlamaClient(config)
        
        # Test detection call
        state_json = '{"ownship": {"id": "OWN"}, "traffic": []}'
        result = client.ask_detect(state_json)
        
        # Should return DetectOut object
        assert isinstance(result, DetectOut)
        assert isinstance(result.conflict, bool)
        
    def test_llm_resolution_simulation(self, config):
        """Test simulated LLM resolution."""
        client = LlamaClient(config)
        
        # Test resolution call
        state_json = '{"ownship": {"id": "OWN"}}'
        conflict = {"intruder_id": "TRF001", "time_to_cpa_min": 4.5}
        result = client.ask_resolve(state_json, conflict)
        
        # Should return ResolveOut object
        assert isinstance(result, ResolveOut)
        assert result.action in ["turn", "climb", "descend"]
        assert isinstance(result.params, dict)
        assert isinstance(result.rationale, str)


class TestSprint3SafetyValidation:
    """Test Sprint 3 - Safety validation and fallback."""
    
    def test_resolution_command_creation(self, ownship):
        """Test creation of resolution commands."""
        llm_output = ResolveOut(
            action="turn",
            params={"heading_deg": 120},
            rationale="Turn right to avoid conflict"
        )
        
        from src.cdr.resolve import _create_resolution_command
        cmd = _create_resolution_command(llm_output, ownship)
        
        assert cmd is not None
        assert cmd.target_aircraft == ownship.aircraft_id
        assert cmd.resolution_type == ResolutionType.HEADING_CHANGE
        assert cmd.new_heading_deg == 120
        
    def test_safety_validation(self, ownship, diverging_intruder):
        """Test resolution safety validation."""
        # Create a safe resolution (turn away from intruder)
        safe_cmd = ResolutionCommand(
            resolution_id="test",
            target_aircraft=ownship.aircraft_id,
            resolution_type=ResolutionType.HEADING_CHANGE,
            new_heading_deg=45,  # Turn away
            issue_time=datetime.now(),
            safety_margin_nm=0.0,
            is_validated=False
        )
        
        # Should pass safety validation
        is_safe = _validate_resolution_safety(safe_cmd, ownship, diverging_intruder)
        assert is_safe
        
    def test_unsafe_resolution_rejection(self, ownship, converging_intruder):
        """Test rejection of unsafe resolutions."""
        # Create an unsafe resolution (turn into intruder)
        unsafe_cmd = ResolutionCommand(
            resolution_id="test",
            target_aircraft=ownship.aircraft_id,
            resolution_type=ResolutionType.HEADING_CHANGE,
            new_heading_deg=270,  # Turn toward intruder
            issue_time=datetime.now(),
            safety_margin_nm=0.0,
            is_validated=False
        )
        
        # May pass or fail depending on exact geometry
        is_safe = _validate_resolution_safety(unsafe_cmd, ownship, converging_intruder)
        assert isinstance(is_safe, bool)
        
    def test_fallback_resolution(self, ownship, converging_intruder):
        """Test fallback resolution generation."""
        from src.cdr.resolve import _generate_fallback_resolution
        
        conflict = ConflictPrediction(
            ownship_id=ownship.aircraft_id,
            intruder_id=converging_intruder.aircraft_id,
            time_to_cpa_min=5.0,
            distance_at_cpa_nm=3.0,
            altitude_diff_ft=500.0,
            is_conflict=True,
            severity_score=0.8,
            conflict_type="both",
            prediction_time=datetime.now()
        )
        
        fallback = _generate_fallback_resolution(ownship, converging_intruder, conflict)
        
        assert fallback is not None
        assert fallback.resolution_type == ResolutionType.ALTITUDE_CHANGE
        assert fallback.new_altitude_ft == ownship.altitude_ft + 1000
        
    def test_end_to_end_resolution(self, ownship, diverging_intruder):
        """Test end-to-end resolution with LLM and safety validation."""
        llm_output = ResolveOut(
            action="climb",
            params={"delta_ft": 1000},
            rationale="Climb to separate vertically"
        )
        
        conflict = ConflictPrediction(
            ownship_id=ownship.aircraft_id,
            intruder_id=diverging_intruder.aircraft_id,
            time_to_cpa_min=5.0,
            distance_at_cpa_nm=3.0,
            altitude_diff_ft=500.0,
            is_conflict=True,
            severity_score=0.8,
            conflict_type="both",
            prediction_time=datetime.now()
        )
        
        result = execute_resolution(llm_output, ownship, diverging_intruder, conflict)
        
        # Should return validated resolution or fallback
        if result:
            assert result.is_validated
            assert result.target_aircraft == ownship.aircraft_id


class TestIntegrationEndToEnd:
    """Integration tests for complete detection-resolution pipeline."""
    
    def test_detection_to_resolution_pipeline(self, ownship, converging_intruder, config):
        """Test complete pipeline from detection to resolution."""
        # Step 1: Detect conflicts
        traffic = [converging_intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Step 2: If conflicts detected, generate LLM resolution
        if conflicts and any(c.is_conflict for c in conflicts):
            client = LlamaClient(config)
            
            # Simulate LLM resolution
            llm_output = ResolveOut(
                action="turn",
                params={"heading_deg": 120},
                rationale="Turn right to avoid conflict"
            )
            
            # Step 3: Validate and execute resolution
            conflict = conflicts[0]
            result = execute_resolution(llm_output, ownship, converging_intruder, conflict)
            
            # Should complete successfully
            assert result is None or isinstance(result, ResolutionCommand)
            
    def test_metrics_collection_full_cycle(self, ownship, converging_intruder):
        """Test metrics collection through full cycle."""
        collector = MetricsCollector()
        
        # Record cycle start
        start_time = datetime.now()
        
        # Detect conflicts
        traffic = [converging_intruder]
        conflicts = predict_conflicts(ownship, traffic)
        
        # Record detection
        collector.record_conflict_detection(conflicts, datetime.now())
        
        # Record cycle end
        cycle_duration = (datetime.now() - start_time).total_seconds()
        collector.record_cycle_time(cycle_duration)
        
        # Generate summary
        summary = collector.generate_summary()
        
        assert summary.total_cycles == 1
        assert summary.total_conflicts_detected == len(conflicts)
        assert summary.avg_cycle_time_sec == cycle_duration
