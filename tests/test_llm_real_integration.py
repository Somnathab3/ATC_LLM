"""Test real LLM integration with Ollama."""

import pytest
from unittest.mock import Mock, patch
import requests
from src.cdr.llm_client import LLMClient, LlamaClient
from src.cdr.schemas import ConfigurationSettings, DetectOut, ResolveOut


class TestLLMClient:
    """Test the new LLMClient with real Ollama integration."""

    def test_init(self):
        """Test LLMClient initialization."""
        client = LLMClient()
        assert client.model == "llama3.1:8b"
        assert client.host == "http://127.0.0.1:11434"
        
        # Test custom config
        client = LLMClient("custom-model", "http://custom-host:8080")
        assert client.model == "custom-model"
        assert client.host == "http://custom-host:8080"

    @patch('requests.post')
    def test_post_success(self, mock_post):
        """Test successful API call to Ollama."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"response": "test response"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        client = LLMClient()
        result = client._post("test prompt")
        
        assert result == "test response"
        mock_post.assert_called_once()
        
        # Check call arguments
        args, kwargs = mock_post.call_args
        assert kwargs['json']['model'] == "llama3.1:8b"
        assert kwargs['json']['prompt'] == "test prompt"
        assert kwargs['json']['stream'] is False
        assert kwargs['timeout'] == 60

    @patch('requests.post')
    def test_post_http_error(self, mock_post):
        """Test HTTP error handling."""
        mock_post.side_effect = requests.RequestException("Connection failed")
        
        client = LLMClient()
        with pytest.raises(requests.RequestException):
            client._post("test prompt")

    @patch.object(LLMClient, '_post')
    def test_call_json_success(self, mock_post):
        """Test successful JSON parsing."""
        mock_post.return_value = '{"test": "value"}'
        
        client = LLMClient()
        result = client.call_json("prompt", "schema")
        
        assert result == {"test": "value"}
        mock_post.assert_called_once()

    @patch.object(LLMClient, '_post')
    def test_call_json_extract_from_text(self, mock_post):
        """Test JSON extraction from text response."""
        mock_post.return_value = 'Here is the JSON: {"test": "value"} end'
        
        client = LLMClient()
        result = client.call_json("prompt", "schema")
        
        assert result == {"test": "value"}

    @patch.object(LLMClient, '_post')
    def test_call_json_retry_on_invalid(self, mock_post):
        """Test retry logic on invalid JSON."""
        # First call returns invalid JSON, second returns valid
        mock_post.side_effect = [
            "invalid json",
            '{"test": "value"}'
        ]
        
        client = LLMClient()
        result = client.call_json("prompt", "schema", max_retries=1)
        
        assert result == {"test": "value"}
        assert mock_post.call_count == 2

    @patch.object(LLMClient, '_post')
    def test_call_json_fallback_to_mock(self, mock_post):
        """Test fallback to mock response on failure."""
        mock_post.side_effect = requests.RequestException("Connection failed")
        
        client = LLMClient()
        result = client.call_json("detect conflicts", "conflict schema")
        
        # Should get mock response
        assert "conflict" in result
        assert result["conflict"] is True

    def test_mock_responses(self):
        """Test mock response generation."""
        client = LLMClient()
        
        # Test detection mock
        result = client._get_mock_json_response("detect conflicts", "conflict schema")
        assert "conflict" in result
        assert "intruders" in result
        
        # Test resolution mock
        result = client._get_mock_json_response("resolve conflict", "action schema")
        assert "action" in result
        assert "params" in result
        assert "rationale" in result
        
        # Test unknown mock
        result = client._get_mock_json_response("unknown", "unknown schema")
        assert "error" in result


class TestLlamaClientCompat:
    """Test LlamaClient backwards compatibility wrapper."""

    def test_init(self):
        """Test initialization with config."""
        config = ConfigurationSettings()
        client = LlamaClient(config)
        
        assert client.llm_client is not None
        assert client.llm_client.model == config.llm_model_name

    @patch.object(LLMClient, 'call_json')
    def test_ask_detect(self, mock_call_json):
        """Test detection request."""
        mock_call_json.return_value = {"conflict": True, "intruders": []}
        
        config = ConfigurationSettings()
        client = LlamaClient(config)
        
        result = client.ask_detect('{"test": "data"}')
        
        assert isinstance(result, DetectOut)
        assert result.conflict is True
        mock_call_json.assert_called_once()

    @patch.object(LLMClient, 'call_json')
    def test_ask_resolve(self, mock_call_json):
        """Test resolution request."""
        mock_call_json.return_value = {
            "action": "turn", 
            "params": {"heading_deg": 120}, 
            "rationale": "Safe turn"
        }
        
        config = ConfigurationSettings()
        client = LlamaClient(config)
        
        result = client.ask_resolve('{"test": "data"}', {"conflict": "test"})
        
        assert isinstance(result, ResolveOut)
        assert result.action == "turn"
        assert result.params == {"heading_deg": 120}
        mock_call_json.assert_called_once()

    @patch.object(LLMClient, 'call_json')
    def test_ask_detect_error_handling(self, mock_call_json):
        """Test error handling in detection."""
        mock_call_json.side_effect = Exception("Test error")
        
        config = ConfigurationSettings()
        client = LlamaClient(config)
        
        result = client.ask_detect('{"test": "data"}')
        assert result is None

    @patch.object(LLMClient, 'call_json')
    def test_ask_resolve_error_handling(self, mock_call_json):
        """Test error handling in resolution."""
        mock_call_json.side_effect = Exception("Test error")
        
        config = ConfigurationSettings()
        client = LlamaClient(config)
        
        result = client.ask_resolve('{"test": "data"}', {})
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
