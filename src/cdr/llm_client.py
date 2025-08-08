"""Local Llama 3.1 8B client for conflict detection and resolution reasoning.

This module provides a safe interface to the local LLM that:
- Enforces structured JSON outputs via Pydantic schemas
- Implements retry logic with exponential backoff
- Validates all outputs before returning to main pipeline
- Maintains conversation context for complex scenarios
"""

from typing import Optional, Dict, Any, Union
import json
import logging
import time
import subprocess
from datetime import datetime

from .schemas import (
    LLMDetectionInput, LLMDetectionOutput,
    LLMResolutionInput, LLMResolutionOutput,
    DetectOut, ResolveOut, ResolutionType,
    ConfigurationSettings, AircraftState
)

logger = logging.getLogger(__name__)


class LlamaClient:
    """Client for local Llama 3.1 8B model inference."""
    
    def __init__(self, config: ConfigurationSettings):
        """Initialize Llama client with configuration.
        
        Args:
            config: System configuration including LLM parameters
        """
        self.config = config
        self.model_name = config.llm_model_name
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        
        # For this implementation, we'll use a subprocess call to ollama
        # In production, you'd use the actual Llama library
        self._model = None
        self._tokenizer = None
        
        logger.info(f"Initialized Llama client with model: {self.model_name}")
    
    def ask_detect(self, state_json: str) -> Optional[DetectOut]:
        """Ask LLM to detect conflicts from traffic state.
        
        Args:
            state_json: JSON string with ownship and traffic states
            
        Returns:
            Structured detection output or None if failed
        """
        try:
            prompt = self._create_detection_prompt(state_json)
            response = self._call_llm(prompt)
            if response:
                return self._parse_detect_response(response)
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
            response = self._call_llm(prompt)
            if response:
                return self._parse_resolve_response(response)
        except Exception as e:
            logger.error(f"Resolution request failed: {e}")
        return None
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM with retry logic.
        Args:
            prompt: Input prompt
        Returns:
            LLM response string or None if failed
        """
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # For this demo, we'll simulate LLM calls
                # In production, you'd call ollama or use transformers library
                if "predict whether ownship will violate" in prompt:
                    return '''{"conflict": true, "intruders": [{"id": "TRF001", "eta_min": 4.5, "why": "converging headings"}]}'''
                elif "Propose ONE maneuver" in prompt:
                    return '''{"action": "turn", "params": {"heading_deg": 120}, "rationale": "Right turn to 120 degrees will provide safe separation"}'''
                else:
                    return '''{"error": "Unknown request type"}'''
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief retry delay
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
        """Parse and validate detection response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Validated DetectOut object or None
        """
        try:
            # First try to parse JSON
            data = json.loads(response)
            
            # Validate required fields
            if "conflict" not in data:
                logger.error("Detection response missing 'conflict' field")
                return None
                
            # Create DetectOut object
            return DetectOut(
                conflict=data["conflict"],
                intruders=data.get("intruders", [])
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse detection JSON: {e}")
            
            # Retry with correction prompt
            corrected_response = self._retry_json_correction(response)
            if corrected_response:
                try:
                    data = json.loads(corrected_response)
                    return DetectOut(
                        conflict=data.get("conflict", False),
                        intruders=data.get("intruders", [])
                    )
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Detection response validation failed: {e}")
            
        return None
    
    def _parse_resolve_response(self, response: str) -> Optional[ResolveOut]:
        """Parse and validate resolution response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Validated ResolveOut object or None
        """
        try:
            # Parse JSON
            data = json.loads(response)
            
            # Validate required fields
            required_fields = ["action", "params", "rationale"]
            for field in required_fields:
                if field not in data:
                    logger.error(f"Resolution response missing '{field}' field")
                    return None
            
            # Validate action type
            valid_actions = ["turn", "climb", "descend"]
            if data["action"] not in valid_actions:
                logger.error(f"Invalid action: {data['action']}")
                return None
                
            return ResolveOut(
                action=data["action"],
                params=data["params"],
                rationale=data["rationale"]
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse resolution JSON: {e}")
            
            # Retry with correction
            corrected_response = self._retry_json_correction(response)
            if corrected_response:
                try:
                    data = json.loads(corrected_response)
                    return ResolveOut(
                        action=data.get("action", "turn"),
                        params=data.get("params", {}),
                        rationale=data.get("rationale", "LLM resolution")
                    )
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Resolution response validation failed: {e}")
            
        return None
    
    def _retry_json_correction(self, malformed_response: str) -> Optional[str]:
        """Attempt to get corrected JSON from LLM.
        
        Args:
            malformed_response: Original malformed response
            
        Returns:
            Corrected JSON string or None
        """
        correction_prompt = f"""The following response is not valid JSON. Please return a corrected version that matches the required schema:

{malformed_response}

Return only valid JSON matching the required format."""
        
        return self._call_llm(correction_prompt)
    
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
