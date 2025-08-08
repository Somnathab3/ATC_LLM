#!/usr/bin/env python3
"""Quick test of the LlamaClient compatibility layer."""

from src.cdr.llm_client import LLMClient, LlamaClient, _extract_first_json

def test_compatibility_layer():
    """Test that both LLMClient and LlamaClient work correctly."""
    
    # Test base LLMClient
    print('Testing LLMClient...')
    base_client = LLMClient(use_mock=True)
    base_conflicts = base_client.detect_conflicts('test prompt')
    print(f'LLMClient detect_conflicts: {base_conflicts}')
    assert isinstance(base_conflicts, dict)
    assert 'conflict' in base_conflicts
    
    # Test LlamaClient compatibility wrapper
    print('\nTesting LlamaClient compatibility wrapper...')
    llama_client = LlamaClient(use_mock=True)
    llama_conflicts = llama_client.detect_conflicts('test prompt')
    print(f'LlamaClient detect_conflicts: {llama_conflicts}')
    assert isinstance(llama_conflicts, dict)
    assert 'conflict' in llama_conflicts
    
    # Test _get_mock_response method
    mock_response = llama_client._get_mock_response('detect')
    print(f'Mock response (string): {mock_response}')
    assert isinstance(mock_response, str)
    
    # Test generate_resolution
    resolution = llama_client.generate_resolution('test prompt')
    print(f'Generate resolution: {resolution}')
    assert isinstance(resolution, dict)
    assert 'action' in resolution
    
    # Test JSON extraction
    test_json = '{"conflict": true, "assessment": "test"}'
    extracted = _extract_first_json(test_json)
    print(f'JSON extraction test: {extracted}')
    assert extracted['conflict'] is True
    assert extracted['assessment'] == 'test'
    
    print('\nAll tests passed! Compatibility layer is working correctly.')

if __name__ == '__main__':
    test_compatibility_layer()
