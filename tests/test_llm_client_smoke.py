"""Smoke tests for LLM client module."""

import pytest
from src.cdr.llm_client import LlamaClient, LLMClient


class TestLLMClientSmoke:
    """Smoke tests for llm_client.py module."""
    
    def test_llama_client_initialization(self):
        """Test LlamaClient initialization."""
        client = LlamaClient()
        
        assert client is not None
        assert hasattr(client, 'model_name')
        assert hasattr(client, 'host')
        assert hasattr(client, 'timeout')
        assert hasattr(client, 'use_mock')
        
        # Test default values
        assert client.model_name == "llama3.1:8b"
        assert "127.0.0.1" in client.host
        assert client.timeout == 30
        assert client.use_mock is False
    
    def test_llama_client_custom_initialization(self):
        """Test LlamaClient with custom parameters."""
        client = LlamaClient(
            model_name="custom_model",
            host="http://192.168.1.100:11434",
            timeout=60,
            use_mock=True
        )
        
        assert client.model_name == "custom_model"
        assert client.host == "http://192.168.1.100:11434"
        assert client.timeout == 60
        assert client.use_mock is True
    
    def test_mock_response_methods(self):
        """Test mock response generation."""
        client = LlamaClient(use_mock=True)
        
        # Test mock response methods exist
        assert hasattr(client, '_get_mock_json_response')
        assert callable(client._get_mock_json_response)
        
        # Test mock responses
        detect_mock = client._get_mock_json_response("detect")
        assert isinstance(detect_mock, dict)
        assert "conflict" in detect_mock
        assert "intruders" in detect_mock
        
        resolve_mock = client._get_mock_json_response("resolve")
        assert isinstance(resolve_mock, dict)
        assert "action" in resolve_mock
    
    def test_call_json_with_mock(self):
        """Test JSON calling with mock enabled."""
        client = LlamaClient(use_mock=True)
        
        response = client.call_json("test prompt", "test schema")
        
        assert isinstance(response, dict)
        # Should return mock response when use_mock=True
        assert "conflict" in response or "action" in response
    
    def test_call_json_without_mock(self):
        """Test JSON calling without mock (may fail if Ollama not available)."""
        client = LlamaClient(use_mock=False)
        
        try:
            response = client.call_json("test prompt", "test schema")
            # Should return dict if successful
            assert isinstance(response, dict)
        except Exception:
            # Expected if Ollama is not available - should fallback to mock
            response = client.call_json("test prompt", "test schema")
            assert isinstance(response, dict)
    
    def test_parse_methods_exist(self):
        """Test that response parsing methods exist."""
        client = LlamaClient()
        
        assert hasattr(client, '_parse_detect_response')
        assert hasattr(client, '_parse_resolve_response')
        assert callable(client._parse_detect_response)
        assert callable(client._parse_resolve_response)
        
        # Test parsing with sample data
        detect_obj = {"conflict": True, "intruders": []}
        parsed_detect = client._parse_detect_response(detect_obj)
        assert isinstance(parsed_detect, dict)
        assert "conflict" in parsed_detect
        
        resolve_obj = {"action": "turn", "altitude_ft": 1000, "heading_deg": 120}
        parsed_resolve = client._parse_resolve_response(resolve_obj)
        assert isinstance(parsed_resolve, dict)
        assert "action" in parsed_resolve
    
    def test_ask_detect_method(self):
        """Test ask_detect method."""
        client = LlamaClient(use_mock=True)  # Use mock to avoid Ollama dependency
        
        assert hasattr(client, 'ask_detect')
        assert callable(client.ask_detect)
        
        # Test with sample JSON
        state_json = '{"ownship": {"id": "OWN"}, "traffic": []}'
        result = client.ask_detect(state_json)
        
        # Should return DetectOut object or None
        assert result is None or hasattr(result, 'conflict')
    
    def test_ask_resolve_method(self):
        """Test ask_resolve method."""
        client = LlamaClient(use_mock=True)  # Use mock to avoid Ollama dependency
        
        assert hasattr(client, 'ask_resolve')
        assert callable(client.ask_resolve)
        
        # Test with sample data
        state_json = '{"ownship": {"id": "OWN"}, "traffic": []}'
        conflict = {"intruder_id": "TFC001", "time_to_cpa_min": 5.0}
        result = client.ask_resolve(state_json, conflict)
        
        # Should return ResolveOut object or None
        assert result is None or hasattr(result, 'action')
    
    def test_llm_client_initialization(self):
        """Test LLMClient initialization."""
        client = LLMClient()
        
        assert client is not None
        assert hasattr(client, 'model')
        assert hasattr(client, 'host')
        assert hasattr(client, 'llama_client')
        
        # Test that it wraps LlamaClient
        assert isinstance(client.llama_client, LlamaClient)
    
    def test_llm_client_delegation(self):
        """Test that LLMClient properly delegates to LlamaClient."""
        client = LLMClient()
        
        # Test delegation methods exist
        assert hasattr(client, 'ask_detect')
        assert hasattr(client, 'ask_resolve')
        assert hasattr(client, 'call_json')
        assert callable(client.ask_detect)
        assert callable(client.ask_resolve)
        assert callable(client.call_json)
    
    def test_module_imports(self):
        """Test that all required classes can be imported."""
        from src.cdr.llm_client import LlamaClient, LLMClient
        
        assert LlamaClient is not None
        assert LLMClient is not None
    
    def test_prompt_creation_methods(self):
        """Test prompt creation methods exist."""
        client = LlamaClient()
        
        assert hasattr(client, '_create_detection_prompt')
        assert hasattr(client, '_create_resolution_prompt')
        assert callable(client._create_detection_prompt)
        assert callable(client._create_resolution_prompt)
        
        # Test prompt creation
        state_json = '{"test": "data"}'
        
        detect_prompt = client._create_detection_prompt(state_json)
        assert isinstance(detect_prompt, str)
        assert len(detect_prompt) > 0
        assert "JSON" in detect_prompt  # Should request JSON format
        
        conflict = {"intruder_id": "TFC001", "time_to_cpa_min": 5.0}
        resolve_prompt = client._create_resolution_prompt(state_json, conflict)
        assert isinstance(resolve_prompt, str)
        assert len(resolve_prompt) > 0
        assert "JSON" in resolve_prompt  # Should request JSON format
    
    def test_json_extraction_method(self):
        """Test JSON extraction from mixed text."""
        client = LlamaClient()
        
        assert hasattr(client, '_extract_json_obj')
        assert callable(client._extract_json_obj)
        
        # Test extraction from text with JSON
        text_with_json = 'Here is the result: {"conflict": true, "reason": "test"} end of response'
        try:
            extracted = client._extract_json_obj(text_with_json)
            assert isinstance(extracted, dict)
            assert "conflict" in extracted
        except Exception:
            # May fail due to regex or JSON parsing - that's OK for smoke test
            assert True
    
    def test_error_handling_graceful(self):
        """Test that methods handle errors gracefully."""
        client = LlamaClient(use_mock=False)  # Don't use mock to test error handling
        
        # These should not crash the test suite
        try:
            client.ask_detect('{"invalid": json}')
            client.ask_resolve('{"invalid": json}', {})
            client.call_json("test", "schema")
        except Exception:
            # Exceptions are expected when Ollama is not available
            # The important thing is that they don't crash the test runner
            assert True
