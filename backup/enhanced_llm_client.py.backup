"""Enhanced LLM client with standardized prompts and BlueSky command parsing.

This module provides:
- Industry-standard prompt formats based on aviation and LLM best practices
- Robust parsing of LLM outputs into BlueSky commands
- Structured prompts with clear constraints and expected formats
- Error handling and fallback mechanisms
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .llm_client import LlamaClient
from .schemas import AircraftState, ConflictPrediction, ConfigurationSettings

logger = logging.getLogger(__name__)

class EnhancedLLMClient(LlamaClient):
    """Enhanced LLM client with standardized prompts and command parsing."""
    
    def __init__(self, config: ConfigurationSettings):
        super().__init__(config)
        self.config = config
        
    def build_enhanced_detect_prompt(self, ownship: AircraftState, 
                                   traffic: List[AircraftState], 
                                   config: ConfigurationSettings) -> str:
        """Build standardized conflict detection prompt based on aviation standards."""
        
        # Format aircraft states for clear LLM understanding
        ownship_info = self._format_aircraft_state(ownship, "OWNSHIP")
        traffic_info = [self._format_aircraft_state(ac, f"TRAFFIC_{i+1}") 
                       for i, ac in enumerate(traffic)]
        
        prompt = f"""You are an expert Air Traffic Controller with ICAO certification and extensive experience in conflict detection.

TASK: Analyze the given aircraft states and detect potential conflicts within the lookahead time window.

STANDARDS:
- Minimum horizontal separation: {config.min_horizontal_separation_nm} nautical miles
- Minimum vertical separation: {config.min_vertical_separation_ft} feet
- Lookahead time: {config.lookahead_time_min} minutes
- Conflict threshold: Both horizontal AND vertical separation violated simultaneously

AIRCRAFT STATES:
{ownship_info}

TRAFFIC:
{chr(10).join(traffic_info)}

ANALYSIS REQUIREMENTS:
1. Project each aircraft's trajectory forward using current heading and speed
2. Calculate Closest Point of Approach (CPA) for each traffic aircraft relative to ownship
3. Determine if separation standards will be violated at CPA
4. Consider only conflicts occurring within the {config.lookahead_time_min}-minute lookahead window

OUTPUT FORMAT (JSON only, no additional text):
{{
    "conflict": true/false,
    "intruders": ["aircraft_id1", "aircraft_id2", ...],
    "horizon_min": {config.lookahead_time_min},
    "conflicts_detected": [
        {{
            "intruder_id": "string",
            "time_to_cpa_min": float,
            "distance_at_cpa_nm": float,
            "altitude_diff_ft": float,
            "severity": float,
            "conflict_type": "horizontal|vertical|both"
        }}
    ],
    "reason": "Clear explanation of detection logic and results"
}}

CRITICAL: Return only valid JSON. No explanations outside the JSON structure."""

        return prompt
    
    def build_enhanced_resolve_prompt(self, ownship: AircraftState,
                                    conflicts: List[ConflictPrediction],
                                    config: ConfigurationSettings) -> str:
        """Build standardized conflict resolution prompt based on aviation best practices."""
        
        # Format conflicts for clear understanding
        conflicts_info = []
        for i, conflict in enumerate(conflicts):
            conflicts_info.append(f"""
CONFLICT {i+1}:
- Intruder: {conflict.intruder_id}
- Time to conflict: {conflict.time_to_cpa_min:.1f} minutes
- Distance at CPA: {conflict.distance_at_cpa_nm:.1f} NM
- Altitude difference: {conflict.altitude_diff_ft:.0f} feet
- Severity: {conflict.severity_score:.2f}
- Type: {conflict.conflict_type}""")
        
        ownship_info = self._format_aircraft_state(ownship, "OWNSHIP")
        
        prompt = f"""You are an expert Air Traffic Controller providing conflict resolution.

SITUATION:
{ownship_info}

CONFLICTS DETECTED:
{chr(10).join(conflicts_info)}

RESOLUTION CONSTRAINTS:
- Maximum heading change: {config.max_resolution_angle_deg}°
- Minimum altitude change: 1000 feet
- Maximum altitude change: {config.max_altitude_change_ft} feet
- Must maintain destination track efficiency
- Prioritize horizontal resolutions over vertical when possible
- Consider traffic flow and airspace restrictions

RESOLUTION TYPES:
1. HEADING_CHANGE: Turn left or right by specified degrees
2. ALTITUDE_CHANGE: Climb or descend to specified flight level
3. SPEED_CHANGE: Adjust speed within aircraft performance limits
4. DIRECT_TO_WAYPOINT: Navigate direct to specified waypoint (if available)

BLUSKY COMMAND FORMAT:
Your resolution will be converted to BlueSky commands:
- Heading: "CALLSIGN HDG xxx" (where xxx is new heading 000-359)
- Altitude: "CALLSIGN ALT xxxxx" (where xxxxx is new altitude in feet)
- Speed: "CALLSIGN SPD xxx" (where xxx is new speed in knots)

OUTPUT FORMAT (JSON only, no additional text):
{{
    "action": "HEADING_CHANGE|ALTITUDE_CHANGE|SPEED_CHANGE|DIRECT_TO_WAYPOINT",
    "params": {{
        "new_heading_deg": float,     // For HEADING_CHANGE
        "heading_delta_deg": float,   // Relative change amount
        "turn_direction": "left|right", // Turn direction
        "new_altitude_ft": float,     // For ALTITUDE_CHANGE  
        "altitude_delta_ft": float,   // Relative change amount
        "climb_descend": "climb|descend", // Vertical direction
        "new_speed_kt": float,        // For SPEED_CHANGE
        "speed_delta_kt": float,      // Relative change amount
        "waypoint_name": "string"     // For DIRECT_TO_WAYPOINT
    }},
    "bluesky_command": "exact BlueSky command string",
    "rationale": "Clear explanation of resolution strategy",
    "expected_separation_improvement": float,  // Expected improvement in NM
    "estimated_delay_min": float,             // Estimated delay to destination
    "confidence": float,                      // Confidence in resolution (0-1)
    "backup_actions": ["alternative action if primary fails"]
}}

EXAMPLE OUTPUT:
{{
    "action": "HEADING_CHANGE",
    "params": {{
        "new_heading_deg": 95.0,
        "heading_delta_deg": 15.0,
        "turn_direction": "right"
    }},
    "bluesky_command": "{ownship.aircraft_id} HDG 095",
    "rationale": "Right turn of 15° to avoid horizontal conflict with TRAFFIC_1, maintaining efficient track to destination",
    "expected_separation_improvement": 2.5,
    "estimated_delay_min": 1.2,
    "confidence": 0.85,
    "backup_actions": ["ALTITUDE_CHANGE to FL370 if horizontal resolution insufficient"]
}}

CRITICAL: Return only valid JSON. No explanations outside the JSON structure."""

        return prompt
    
    def _format_aircraft_state(self, aircraft: AircraftState, label: str) -> str:
        """Format aircraft state for prompt clarity."""
        return f"""{label}:
- Aircraft ID: {aircraft.aircraft_id}
- Position: {aircraft.latitude:.6f}°N, {aircraft.longitude:.6f}°E
- Altitude: {aircraft.altitude_ft:.0f} feet
- Heading: {aircraft.heading_deg:.0f}° True
- Ground Speed: {aircraft.ground_speed_kt:.0f} knots
- Timestamp: {aircraft.timestamp}"""
    
    def parse_enhanced_detection_response(self, response_text: str) -> Dict[str, Any]:
        """Parse enhanced detection response with robust error handling."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in detection response")
                return self._fallback_detection_response()
            
            response_data = json.loads(json_match.group())
            
            # Validate required fields
            required_fields = ['conflict', 'intruders', 'horizon_min', 'reason']
            for field in required_fields:
                if field not in response_data:
                    logger.warning(f"Missing required field '{field}' in detection response")
                    response_data[field] = self._get_default_detection_value(field)
            
            # Validate data types and ranges
            response_data['conflict'] = bool(response_data['conflict'])
            response_data['intruders'] = list(response_data['intruders'])
            response_data['horizon_min'] = float(response_data['horizon_min'])
            
            # Validate conflicts_detected if present
            if 'conflicts_detected' in response_data:
                validated_conflicts = []
                for conflict in response_data['conflicts_detected']:
                    if self._validate_conflict_data(conflict):
                        validated_conflicts.append(conflict)
                response_data['conflicts_detected'] = validated_conflicts
            
            logger.debug(f"Parsed detection response: {len(response_data.get('intruders', []))} intruders")
            return response_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in detection response: {e}")
            return self._fallback_detection_response()
        except Exception as e:
            logger.error(f"Error parsing detection response: {e}")
            return self._fallback_detection_response()
    
    def parse_enhanced_resolution_response(self, response_text: str) -> Dict[str, Any]:
        """Parse enhanced resolution response with robust error handling."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in resolution response")
                return self._fallback_resolution_response()
            
            response_data = json.loads(json_match.group())
            
            # Validate required fields
            required_fields = ['action', 'params', 'bluesky_command', 'rationale']
            for field in required_fields:
                if field not in response_data:
                    logger.warning(f"Missing required field '{field}' in resolution response")
                    response_data[field] = self._get_default_resolution_value(field)
            
            # Validate action type
            valid_actions = ['HEADING_CHANGE', 'ALTITUDE_CHANGE', 'SPEED_CHANGE', 'DIRECT_TO_WAYPOINT']
            if response_data['action'] not in valid_actions:
                logger.warning(f"Invalid action type: {response_data['action']}")
                response_data['action'] = 'HEADING_CHANGE'  # Default to heading change
            
            # Validate and sanitize BlueSky command
            response_data['bluesky_command'] = self._sanitize_bluesky_command(
                response_data['bluesky_command']
            )
            
            # Set defaults for optional fields
            response_data.setdefault('confidence', 0.5)
            response_data.setdefault('expected_separation_improvement', 1.0)
            response_data.setdefault('estimated_delay_min', 0.0)
            response_data.setdefault('backup_actions', [])
            
            logger.debug(f"Parsed resolution response: {response_data['action']} - {response_data['bluesky_command']}")
            return response_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in resolution response: {e}")
            return self._fallback_resolution_response()
        except Exception as e:
            logger.error(f"Error parsing resolution response: {e}")
            return self._fallback_resolution_response()
    
    def _sanitize_bluesky_command(self, command: str) -> str:
        """Sanitize and validate BlueSky command format."""
        try:
            command = command.strip().upper()
            
            # Basic BlueSky command patterns
            patterns = [
                r'^[A-Z0-9]+\s+HDG\s+\d{1,3}$',           # Heading command
                r'^[A-Z0-9]+\s+ALT\s+\d{1,5}$',           # Altitude command  
                r'^[A-Z0-9]+\s+SPD\s+\d{1,3}$',           # Speed command
                r'^[A-Z0-9]+\s+DCT\s+[A-Z0-9]+$',         # Direct to waypoint
            ]
            
            # Check if command matches any valid pattern
            for pattern in patterns:
                if re.match(pattern, command):
                    return command
            
            # If no pattern matches, try to fix common issues
            parts = command.split()
            if len(parts) >= 3:
                callsign = parts[0]
                cmd_type = parts[1]
                value = parts[2]
                
                # Fix command type
                if cmd_type in ['HEADING', 'HEAD']:
                    cmd_type = 'HDG'
                elif cmd_type in ['ALTITUDE', 'ALT']:
                    cmd_type = 'ALT'
                elif cmd_type in ['SPEED', 'SPD']:
                    cmd_type = 'SPD'
                elif cmd_type in ['DIRECT', 'DCT']:
                    cmd_type = 'DCT'
                
                # Validate value ranges
                if cmd_type == 'HDG':
                    try:
                        hdg_val = int(float(value))
                        hdg_val = max(0, min(359, hdg_val))  # Clamp to valid range
                        return f"{callsign} HDG {hdg_val:03d}"
                    except ValueError:
                        pass
                elif cmd_type == 'ALT':
                    try:
                        alt_val = int(float(value))
                        alt_val = max(1000, min(50000, alt_val))  # Reasonable altitude range
                        return f"{callsign} ALT {alt_val}"
                    except ValueError:
                        pass
                elif cmd_type == 'SPD':
                    try:
                        spd_val = int(float(value))
                        spd_val = max(100, min(600, spd_val))  # Reasonable speed range
                        return f"{callsign} SPD {spd_val}"
                    except ValueError:
                        pass
            
            logger.warning(f"Could not sanitize BlueSky command: {command}")
            return command  # Return original if can't fix
            
        except Exception as e:
            logger.error(f"Error sanitizing BlueSky command: {e}")
            return command
    
    def execute_bluesky_command(self, command: str, bluesky_client) -> bool:
        """Execute sanitized BlueSky command with error handling."""
        try:
            # Final validation before execution
            sanitized_command = self._sanitize_bluesky_command(command)
            
            # Execute command
            success = bluesky_client.stack(sanitized_command)
            
            if success:
                logger.info(f"Successfully executed BlueSky command: {sanitized_command}")
            else:
                logger.error(f"Failed to execute BlueSky command: {sanitized_command}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing BlueSky command '{command}': {e}")
            return False
    
    def _validate_conflict_data(self, conflict: Dict[str, Any]) -> bool:
        """Validate conflict detection data."""
        required_fields = ['intruder_id', 'time_to_cpa_min', 'distance_at_cpa_nm', 'altitude_diff_ft']
        
        for field in required_fields:
            if field not in conflict:
                return False
        
        # Validate ranges
        try:
            if conflict['time_to_cpa_min'] < 0 or conflict['time_to_cpa_min'] > 60:
                return False
            if conflict['distance_at_cpa_nm'] < 0 or conflict['distance_at_cpa_nm'] > 100:
                return False
            if conflict['altitude_diff_ft'] < 0 or conflict['altitude_diff_ft'] > 10000:
                return False
        except (ValueError, TypeError):
            return False
        
        return True
    
    def _fallback_detection_response(self) -> Dict[str, Any]:
        """Provide fallback detection response."""
        return {
            "conflict": False,
            "intruders": [],
            "horizon_min": 10,
            "reason": "Fallback: LLM response parsing failed",
            "conflicts_detected": []
        }
    
    def _fallback_resolution_response(self) -> Dict[str, Any]:
        """Provide fallback resolution response."""
        return {
            "action": "HEADING_CHANGE",
            "params": {
                "new_heading_deg": 0.0,
                "heading_delta_deg": 10.0,
                "turn_direction": "right"
            },
            "bluesky_command": "UNKNOWN HDG 010",
            "rationale": "Fallback: LLM response parsing failed, applying safe turn",
            "confidence": 0.1,
            "expected_separation_improvement": 1.0,
            "estimated_delay_min": 0.5,
            "backup_actions": []
        }
    
    def _get_default_detection_value(self, field: str) -> Any:
        """Get default value for missing detection field."""
        defaults = {
            'conflict': False,
            'intruders': [],
            'horizon_min': 10,
            'reason': 'Default: missing field'
        }
        return defaults.get(field, None)
    
    def _get_default_resolution_value(self, field: str) -> Any:
        """Get default value for missing resolution field."""
        defaults = {
            'action': 'HEADING_CHANGE',
            'params': {},
            'bluesky_command': 'UNKNOWN HDG 010',
            'rationale': 'Default: missing field'
        }
        return defaults.get(field, None)
    
    def validate_llm_connection(self) -> bool:
        """Validate LLM connection with standardized test."""
        try:
            test_prompt = """You are an expert Air Traffic Controller.

TASK: Respond with exactly this JSON format and no additional text:

{"status": "connected", "model": "test", "capability": "conflict_detection_and_resolution"}

CRITICAL: Return only valid JSON."""

            response = self._post_ollama(test_prompt)
            
            if response and isinstance(response, dict):
                if response.get('status') == 'connected':
                    logger.info("LLM connection validated successfully")
                    return True
            
            logger.warning("LLM connection validation failed")
            return False
            
        except Exception as e:
            logger.error(f"LLM connection validation error: {e}")
            return False
