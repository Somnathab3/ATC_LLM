"""Test suite for LLM integration with Ollama."""

import pytest
import logging
from unittest.mock import patch, MagicMock
from src.cdr.llm_client import LlamaClient
from src.cdr.schemas import ConfigurationSettings, DetectOut, ResolveOut


class TestLLMIntegration:
    """Test LLM integration functionality."""
    
    def test_llm_client_init(self):
        """Test LLM client initialization."""
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        
        client = LlamaClient(config)
        assert client.config == config
        assert client.model_name == "llama3.1:8b"
        assert client.temperature == 0.1
        assert client.max_tokens == 2048
    
    def test_mock_responses(self):
        """Test mock response generation."""
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        
        client = LlamaClient(config)
        
        # Test detection mock response
        detect_response = client._get_mock_response("predict whether ownship will violate")
        assert "conflict" in detect_response
        assert "true" in detect_response.lower()
        
        # Test resolution mock response  
        resolve_response = client._get_mock_response("Propose ONE maneuver")
        assert "action" in resolve_response
        assert "turn" in resolve_response.lower()
    
    @patch('subprocess.run')
    def test_ollama_call_success(self, mock_run):
        """Test successful Ollama call."""
        # Mock successful Ollama response
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"conflict": true, "intruders": []}'
        mock_run.return_value = mock_result
        
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        
        client = LlamaClient(config)
        response = client._call_llm("test prompt")
        
        assert response == '{"conflict": true, "intruders": []}'
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_ollama_call_fallback(self, mock_run):
        """Test fallback to mock when Ollama fails."""
        # Mock Ollama not found
        mock_run.side_effect = FileNotFoundError("ollama not found")
        
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        
        client = LlamaClient(config)
        response = client._call_llm("detect conflicts")
        
        # Should fall back to mock response
        assert response is not None
        assert "conflict" in response
    
    def test_parse_detect_response(self):
        """Test parsing of detection responses."""
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        
        client = LlamaClient(config)
        
        # Test valid JSON response
        valid_response = '{"conflict": true, "intruders": [{"id": "TEST001"}]}'
        result = client._parse_detect_response(valid_response)
        
        assert result is not None
        assert isinstance(result, DetectOut)
        assert result.conflict is True
        assert len(result.intruders) == 1
    
    def test_parse_resolve_response(self):
        """Test parsing of resolution responses."""
        config = ConfigurationSettings(
            polling_interval_min=5.0,
            lookahead_time_min=10.0,
            min_horizontal_separation_nm=5.0,
            min_vertical_separation_ft=1000.0,
            llm_model_name="llama3.1:8b",
            llm_temperature=0.1,
            llm_max_tokens=2048,
            safety_buffer_factor=1.2,
            max_resolution_angle_deg=30.0,
            max_altitude_change_ft=2000.0,
            bluesky_host="localhost",
            bluesky_port=1337,
            bluesky_timeout_sec=5.0
        )
        
        client = LlamaClient(config)
        
        # Test valid JSON response
        valid_response = '{"action": "turn", "params": {"heading_deg": 120}, "rationale": "Safe separation"}'
        result = client._parse_resolve_response(valid_response)
        
        assert result is not None
        assert isinstance(result, ResolveOut)
        assert result.action == "turn"
        assert result.params["heading_deg"] == 120
        assert "Safe separation" in result.rationale
