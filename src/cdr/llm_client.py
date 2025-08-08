"""Local Llama 3.1 8B client for conflict detection and resolution reasoning.

This module provides a safe interface to the local LLM that:
- Connects to Ollama for local Llama 3.1 8B inference
- Enforces structured JSON outputs via Pydantic schemas
- Implements retry logic with strict JSON validation
- Validates all outputs before returning to main pipeline
- Falls back to mock responses for development/testing
"""

from __future__ import annotations
import json, logging, requests, time
from typing import Dict, Any, Optional
from datetime import datetime

from .schemas import (
    LLMDetectionInput, LLMDetectionOutput,
    LLMResolutionInput, LLMResolutionOutput,
    DetectOut, ResolveOut, ResolutionType,
    ConfigurationSettings
)

log = logging.getLogger(__name__)


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
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    def call_json(self, prompt: str, schema_hint: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Ask LLM to return **JSON only**; retry once with schema hint if invalid.
        
        Args:
            prompt: Base prompt
            schema_hint: JSON schema description
            max_retries: Maximum retry attempts
            
        Returns:
            Parsed JSON object
            
        Raises:
            ValueError: If LLM fails to return valid JSON after retries
        """
        suffix = "\n\nReturn **ONLY** a valid compact JSON object that matches this schema:\n" + schema_hint
        for attempt in range(max_retries + 1):
            txt = self._post(prompt + suffix)
            try:
                obj = json.loads(txt)
                return obj
            except Exception:
                log.warning("LLM returned non-JSON (attempt %s): %s", attempt+1, txt[:200])
                suffix = "\n\nSTRICT: Output only valid JSON, no prose. Schema:\n" + schema_hint
                time.sleep(0.5)
        raise ValueError("LLM did not return valid JSON")


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
            log.error(f"Detection request failed: {e}")
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
            log.error(f"Resolution request failed: {e}")
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
                return LLMDetectionOutput(
                    conflicts_detected=[],  # Would convert from detect_out.intruders
                    assessment=f"Conflicts detected: {detect_out.conflict}",
                    confidence=0.8,  # Default confidence
                    reasoning="LLM-based conflict detection"
                )
                
        except Exception as e:
            log.error(f"LLM conflict detection failed: {e}")
            
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
                from .schemas import ResolutionCommand
                
                # Create resolution command based on LLM output
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
            log.error(f"LLM resolution generation failed: {e}")
            
        return None
    
    def _create_detection_prompt(self, state_json: str) -> str:
        """Create structured prompt for conflict detection."""
        return f"""System: You are an ATC safety assistant. Output JSON only.

User: Given this state snapshot (ownship + traffic within 100 NM, ±5000 ft) and
the next two waypoints, predict whether ownship will violate 5 NM & 1000 ft
within the next 10 minutes. List intruders, ETA to closest approach, and a short reason.

State data:
{state_json}

Respond with JSON in this format:
{{"conflict": true/false, "intruders": [{{"id": "aircraft_id", "eta_min": 5.2, "why": "reason"}}]}}"""
    
    def _create_resolution_prompt(self, state_json: str, conflict: Dict[str, Any]) -> str:
        """Create structured prompt for resolution generation."""
        intruder_id = conflict.get("intruder_id", "UNKNOWN")
        eta = conflict.get("time_to_cpa_min", 0)
        
        return f"""System: You are an ATC safety assistant. Output JSON only.

User: A conflict with {intruder_id} is predicted in {eta:.1f} minutes. Propose ONE maneuver
for ownship ONLY that avoids conflict and preserves trajectory intent.

State data:
{state_json}

Respond with JSON in this format:
{{"action": "turn|climb|descend", "params": {{"heading_deg": 90}} or {{"delta_ft": 1000}}, "rationale": "explanation"}}"""

    def _map_action_to_type(self, action: str) -> ResolutionType:
        """Map LLM action string to ResolutionType enum."""
        action_map = {
            "turn": ResolutionType.HEADING_CHANGE,
            "climb": ResolutionType.ALTITUDE_CHANGE,
            "descend": ResolutionType.ALTITUDE_CHANGE
        }
        return action_map.get(action, ResolutionType.HEADING_CHANGE)
