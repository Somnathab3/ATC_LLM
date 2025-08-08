from datetime import datetime
from unittest.mock import patch

from src.cdr.llm_client import LlamaClient
from src.cdr.schemas import (
    LLMDetectionInput, LLMResolutionInput, ConfigurationSettings, AircraftState, ResolveOut
)

class TestLLMMockAndSchema:
    def test_llm_returns_valid_json_and_schema_parsing(self):
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama-3.1-8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=45.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        client = LlamaClient(config)
        ownship = AircraftState(
            aircraft_id="OWN",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        traffic: list[AircraftState] = []
        detect_input = LLMDetectionInput(
            ownship=ownship,
            traffic=traffic,
            lookahead_minutes=10.0,
            current_time=datetime.now()
        )
        # Should parse valid mock JSON
        out = client.detect_conflicts(detect_input)
        assert out is not None
        assert hasattr(out, 'assessment')

    def test_llm_invalid_json_triggers_fallback(self):
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama-3.1-8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=45.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        client = LlamaClient(config)
        # Patch _call_llm to return invalid JSON
        with patch.object(client, '_call_llm', return_value='{bad json: true,}'):  # malformed
            ownship = AircraftState(
                aircraft_id="OWN",
                timestamp=datetime.now(),
                latitude=59.3,
                longitude=18.1,
                altitude_ft=35000,
                ground_speed_kt=450,
                heading_deg=90,
                vertical_speed_fpm=0
            )
            traffic: list[AircraftState] = []
            detect_input = LLMDetectionInput(
                ownship=ownship,
                traffic=traffic,
                lookahead_minutes=10.0,
                current_time=datetime.now()
            )
            out = client.detect_conflicts(detect_input)
            # Should be None or fallback, depending on implementation
            assert out is None or hasattr(out, 'assessment')

    def test_llm_resolution_safe(self):
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama-3.1-8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=45.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        client = LlamaClient(config)
        ownship = AircraftState(
            aircraft_id="OWN",
            timestamp=datetime.now(),
            latitude=59.3,
            longitude=18.1,
            altitude_ft=35000,
            ground_speed_kt=450,
            heading_deg=90,
            vertical_speed_fpm=0
        )
        traffic: list[AircraftState] = []
        from src.cdr.schemas import ConflictPrediction
        conflict = ConflictPrediction(
            ownship_id="OWN",
            intruder_id="TRF001",
            time_to_cpa_min=4.5,
            distance_at_cpa_nm=3.0,
            altitude_diff_ft=500.0,
            is_conflict=True,
            severity_score=0.8,
            conflict_type="both",
            prediction_time=datetime.now(),
            confidence=1.0
        )
        res_input = LLMResolutionInput(
            conflict=conflict,
            ownship=ownship,
            traffic=traffic
        )
        out = client.generate_resolution(res_input)
        assert out is not None
        assert hasattr(out, 'recommended_resolution')
        assert out.recommended_resolution.resolution_type is not None
