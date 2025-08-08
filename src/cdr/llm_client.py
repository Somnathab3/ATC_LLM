"""Local Llama 3.1 8B client for conflict detection and resolution reasoning.

This module provides a safe interface to the local LLM that:
- Connects to Ollama for local Llama 3.1 8B inference
- Enforces structured JSON outputs via Pydantic schemas
- Implements retry logic with strict JSON validation
- Validates all outputs before returning to main pipeline
- Falls back to mock responses for development/testing
"""

from typing import Optional, Dict, Any
import json
import logging
import time
import requests
from datetime import datetime

from .schemas import (
    LLMDetectionInput, LLMDetectionOutput,
    LLMResolutionInput, LLMResolutionOutput,
    DetectOut, ResolveOut, ResolutionType,
    ConfigurationSettings
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for local Llama 3.1 8B model inference via Ollama."""
    
    def __init__(self, model: str = "llama3.1:8b", host: str = "http://127.0.0.1:11434"):
        """Initialize LLM client with Ollama configuration.
        
        Args:
            model: Ollama model name
            host: Ollama server endpoint
        """
        self.model = model
        self.host = host.rstrip("/")
        
        logger.info(f"Initialized LLM client with model: {self.model} at {self.host}")

    def _post(self, prompt: str) -> str:
        """Send request to Ollama API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Raw response text from LLM
            
        Raises:
            requests.RequestException: On HTTP errors
        """
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model, 
            "prompt": prompt, 
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 512
            }
        }
        
        logger.debug(f"Sending prompt to Ollama: {prompt[:100]}...")
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return data.get("response", "")

    def call_json(self, prompt: str, schema_hint: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Ask LLM to return **JSON only**; retry with schema hint if invalid.
        
        Args:
            prompt: Base prompt
            schema_hint: JSON schema description
            max_retries: Maximum retry attempts
            
        Returns:
            Parsed JSON object
            
        Raises:
            ValueError: If LLM fails to return valid JSON after retries
        """
        suffix = f"\n\nReturn **ONLY** a valid compact JSON object that matches this schema:\n{schema_hint}"
        
        for attempt in range(max_retries + 1):
            try:
                full_prompt = prompt + suffix
                response_text = self._post(full_prompt)
                
                # Try to extract JSON from response
                response_text = response_text.strip()
                
                # Sometimes LLM adds extra text, try to find JSON
                if not response_text.startswith('{'):
                    # Look for JSON object in response
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        response_text = response_text[start_idx:end_idx]
                
                # Parse JSON
                obj = json.loads(response_text)
                logger.debug(f"Successfully parsed JSON: {obj}")
                return obj
                
            except requests.RequestException as e:
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    # Fallback to mock response
                    return self._get_mock_json_response(prompt, schema_hint)
                time.sleep(0.5)
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error on attempt {attempt + 1}: {e}")
                logger.warning(f"Raw response: {response_text[:200]}")
                
                # Make suffix more strict for next attempt
                suffix = f"\n\nSTRICT: Output only valid JSON, no prose. Schema:\n{schema_hint}"
                
                if attempt == max_retries:
                    # Fallback to mock response
                    return self._get_mock_json_response(prompt, schema_hint)
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries:
                    return self._get_mock_json_response(prompt, schema_hint)
                time.sleep(0.5)
        
        raise ValueError("LLM did not return valid JSON after all retries")

    def _get_mock_json_response(self, prompt: str, schema_hint: str) -> Dict[str, Any]:
        """Generate mock JSON responses for fallback.
        
        Args:
            prompt: Original prompt
            schema_hint: Expected schema
            
        Returns:
            Mock JSON response matching expected schema
        """
        logger.info("Using mock JSON response as fallback")
        
        if "detect conflicts" in prompt.lower() or "conflict" in schema_hint.lower():
            return {
                "conflict": True, 
                "intruders": [
                    {
                        "id": "TRF001", 
                        "eta_min": 4.5, 
                        "distance_nm": 3.2,
                        "why": "converging headings - mock response"
                    }
                ]
            }
        elif "resolve" in prompt.lower() or "action" in schema_hint.lower():
            return {
                "action": "turn", 
                "params": {"heading_deg": 120}, 
                "rationale": "Right turn to 120 degrees will provide safe separation - mock response"
            }
        else:
            return {"error": "Unknown request type - mock response"}


class LlamaClient:
    """Legacy wrapper for backwards compatibility."""
    
    def __init__(self, config: ConfigurationSettings):
        """Initialize with config for backwards compatibility."""
        self.llm_client = LLMClient(
            model=config.llm_model_name,
            host="http://127.0.0.1:11434"
        )
        self.config = config
        
    def ask_detect(self, state_json: str) -> Optional[DetectOut]:
        """Ask LLM to detect conflicts from traffic state.
        
        Args:
            state_json: JSON string with ownship and traffic states
            
        Returns:
            Structured detection output or None if failed
        """
        try:
            prompt = self._create_detection_prompt(state_json)
            schema_hint = '{"conflict": boolean, "intruders": [{"id": string, "eta_min": number, "distance_nm": number, "why": string}]}'
            
            response_json = self.llm_client.call_json(prompt, schema_hint)
            return DetectOut(**response_json)
            
        except Exception as e:
            logger.error(f"Detection request failed: {e}")
            return None
    
    def ask_resolve(self, state_json: str, conflict: Dict[str, Any]) -> Optional[ResolveOut]:
        """Ask LLM to generate conflict resolution.
        
        Args:
            state_json: JSON string with current traffic state
            conflict: Detected conflict information
            
        Returns:
            Structured resolution output or None if failed
        """
        try:
            prompt = self._create_resolution_prompt(state_json, conflict)
            schema_hint = '{"action": "turn|climb|descend", "params": {"heading_deg": number} or {"delta_ft": number}, "rationale": string}'
            
            response_json = self.llm_client.call_json(prompt, schema_hint)
            return ResolveOut(**response_json)
            
        except Exception as e:
            logger.error(f"Resolution request failed: {e}")
            return None

    def detect_conflicts(self, input_data: LLMDetectionInput) -> Optional[LLMDetectionOutput]:
        """Use LLM to detect conflicts from current traffic situation.
        
        Args:
            input_data: Current aircraft states and detection parameters
            
        Returns:
            LLM conflict detection results or None if failed
        """
        try:
            # Convert input to JSON
            state_dict = {
                "ownship": input_data.ownship.model_dump(),
                "traffic": [aircraft.model_dump() for aircraft in input_data.traffic],
                "lookahead_minutes": input_data.lookahead_minutes,
                "current_time": input_data.current_time.isoformat()
            }
            state_json = json.dumps(state_dict, indent=2, default=str)
            
            # Call LLM detection
            detect_out = self.ask_detect(state_json)
            
            if detect_out:
                # Convert to LLMDetectionOutput format
                # This is a simplified conversion - in practice you'd need more logic
                return LLMDetectionOutput(
                    conflicts_detected=[],  # Would convert from detect_out.intruders
                    assessment=f"Conflicts detected: {detect_out.conflict}",
                    confidence=0.8,  # Default confidence
                    reasoning="LLM-based conflict detection"
                )
                
        except Exception as e:
            logger.error(f"LLM conflict detection failed: {e}")
            
        return None
    
    def generate_resolution(self, input_data: LLMResolutionInput) -> Optional[LLMResolutionOutput]:
        """Use LLM to generate conflict resolution strategy.
        
        Args:
            input_data: Conflict details and resolution context
            
        Returns:
            LLM resolution recommendation or None if failed
        """
        try:
            # Convert input to JSON
            state_dict = {
                "conflict": input_data.conflict.model_dump(),
                "ownship": input_data.ownship.model_dump(),
                "traffic": [aircraft.model_dump() for aircraft in input_data.traffic],
                "constraints": input_data.constraints
            }
            state_json = json.dumps(state_dict, indent=2, default=str)
            
            # Call LLM resolution
            resolve_out = self.ask_resolve(state_json, input_data.conflict.model_dump())
            
            if resolve_out:
                # Convert to LLMResolutionOutput format
                # This is a simplified conversion
                from .schemas import ResolutionCommand, ResolutionType
                
                # Create resolution command based on LLM output
                # Set required fields for ResolutionCommand
                new_heading_deg = None
                new_speed_kt = None
                new_altitude_ft = None
                if resolve_out.action == "turn" and "heading_deg" in resolve_out.params:
                    new_heading_deg = resolve_out.params["heading_deg"]
                elif resolve_out.action in ["climb", "descend"] and "delta_ft" in resolve_out.params:
                    current_alt = input_data.ownship.altitude_ft
                    if resolve_out.action == "climb":
                        new_altitude_ft = current_alt + resolve_out.params["delta_ft"]
                    else:
                        new_altitude_ft = current_alt - resolve_out.params["delta_ft"]
                resolution_command = ResolutionCommand(
                    resolution_id=f"llm_{int(time.time())}",
                    target_aircraft=input_data.ownship.aircraft_id,
                    resolution_type=self._map_action_to_type(resolve_out.action),
                    new_heading_deg=new_heading_deg,
                    new_speed_kt=new_speed_kt,
                    new_altitude_ft=new_altitude_ft,
                    issue_time=datetime.now(),
                    expected_completion_time=None,
                    is_validated=False,
                    safety_margin_nm=5.0
                )
                return LLMResolutionOutput(
                    recommended_resolution=resolution_command,
                    reasoning=resolve_out.rationale,
                    risk_assessment="LLM-generated resolution with safety validation pending",
                    confidence=0.8
                )
                
        except Exception as e:
            logger.error(f"LLM resolution generation failed: {e}")
            
        return None
    
    def _create_detection_prompt(self, state_json: str) -> str:
        """Create structured prompt for conflict detection.
        
        Args:
            state_json: JSON state data
            
        Returns:
            Formatted prompt string
        """
        return f"""System: You are an ATC safety assistant. Output JSON only.

User: Given this state snapshot (ownship + traffic within 100 NM, ±5000 ft) and
the next two waypoints, predict whether ownship will violate 5 NM & 1000 ft
within the next 10 minutes. List intruders, ETA to closest approach, and a short reason.

State data:
{state_json}

Respond with JSON in this format:
{{
  "conflict": true/false,
  "intruders": [
    {{
      "id": "aircraft_id",
      "eta_min": 5.2,
      "why": "converging headings will result in loss of separation"
    }}
  ]
}}"""
    
    def _create_resolution_prompt(self, state_json: str, conflict: Dict[str, Any]) -> str:
        """Create structured prompt for resolution generation.
        
        Args:
            state_json: JSON state data
            conflict: Conflict information
            
        Returns:
            Formatted prompt string
        """
        intruder_id = conflict.get("intruder_id", "UNKNOWN")
        eta = conflict.get("time_to_cpa_min", 0)
        
        return f"""System: You are an ATC safety assistant. Output JSON only.

User: A conflict with {intruder_id} is predicted in {eta:.1f} minutes. Propose ONE maneuver
for ownship ONLY that avoids conflict and preserves trajectory intent:
- Horizontal: heading change ≤ 30°, short-duration vectoring then DCT next waypoint.
- Vertical: climb/descent 1000–2000 ft within available band.
Do not suggest speed changes. Provide rationale.

State data:
{state_json}

Respond with JSON in this format:
{{
  "action": "turn|climb|descend",
  "params": {{"heading_deg": 90}} or {{"delta_ft": 1000}},
  "rationale": "explanation of why this maneuver resolves the conflict safely"
            LLM response string or None if failed
        """
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # For this demo, we'll simulate LLM calls
                # In production, you'd call ollama or use transformers library
                
                # Simulate detection response
                if "predict whether ownship will violate" in prompt:
                    return '''{"conflict": true, "intruders": [{"id": "TRF001", "eta_min": 4.5, "why": "converging headings"}]}'''
                
                # Simulate resolution response
                elif "Propose ONE maneuver" in prompt:
                    return '''{"action": "turn", "params": {"heading_deg": 120}, "rationale": "Right turn to 120 degrees will provide safe separation"}'''
                
                else:
                    return '''{"error": "Unknown request type"}'''
                    
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief retry delay
                    
        return None
    
    def _parse_detect_response(self, response: str) -> Optional[DetectOut]:
        """Legacy method - not used with new JSON approach."""
        logger.warning("_parse_detect_response is deprecated - use ask_detect instead")
        return None
    
    def _parse_resolve_response(self, response: str) -> Optional[ResolveOut]:
        """Legacy method - not used with new JSON approach."""  
        logger.warning("_parse_resolve_response is deprecated - use ask_resolve instead")
        return None

    def _map_action_to_type(self, action: str) -> ResolutionType:
        """Map LLM action string to ResolutionType enum.
        
        Args:
            action: Action string from LLM
            
        Returns:
            Corresponding ResolutionType
        """
        action_map = {
            "turn": ResolutionType.HEADING_CHANGE,
            "climb": ResolutionType.ALTITUDE_CHANGE,
            "descend": ResolutionType.ALTITUDE_CHANGE
        }
        
        return action_map.get(action, ResolutionType.HEADING_CHANGE)
    
    
    def _validate_json_output(self, response: str, expected_schema: type) -> Optional[Dict[str, Any]]:
        """Validate LLM JSON output against expected schema.
        
        Args:
            response: Raw LLM response string
            expected_schema: Pydantic model class for validation
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            *args, **kwargs: Function arguments
            
        Returns:
            Function result or raises last exception
        """
        # TODO: Implement in Sprint 3
        pass
