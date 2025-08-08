"""Pydantic schemas for data validation and API contracts.

Defines structured data models for:
- Aircraft state representation
- Conflict predictions  
- Resolution commands
- LLM input/output formats
- Configuration parameters
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class AircraftState(BaseModel):
    """Aircraft state at a given time."""
    
    aircraft_id: str = Field(..., description="Unique aircraft identifier")
    timestamp: datetime = Field(..., description="State timestamp")
    
    # Position
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    altitude_ft: float = Field(..., ge=0, le=60000, description="Altitude in feet")
    
    # Velocity
    ground_speed_kt: float = Field(..., ge=0, le=1000, description="Ground speed in knots")
    heading_deg: float = Field(..., ge=0, lt=360, description="True heading in degrees")
    vertical_speed_fpm: float = Field(0, description="Vertical speed in feet per minute")
    
    # Flight plan (optional)
    callsign: Optional[str] = None
    aircraft_type: Optional[str] = None
    destination: Optional[str] = None
    
    @validator('heading_deg')
    def normalize_heading(cls, v):
        """Normalize heading to [0, 360) range."""
        return v % 360.0


class ConflictPrediction(BaseModel):
    """Predicted conflict between ownship and intruder."""
    
    ownship_id: str
    intruder_id: str
    
    # Conflict geometry
    time_to_cpa_min: float = Field(..., ge=0, description="Time to closest approach in minutes")
    distance_at_cpa_nm: float = Field(..., ge=0, description="Horizontal separation at CPA in NM")
    altitude_diff_ft: float = Field(..., description="Vertical separation in feet")
    
    # Severity assessment
    is_conflict: bool = Field(..., description="True if violates separation standards")
    severity_score: float = Field(..., ge=0, le=1, description="Conflict severity [0-1]")
    
    # Conflict type
    conflict_type: str = Field(..., description="horizontal, vertical, or both")
    
    # Prediction metadata
    prediction_time: datetime = Field(..., description="When prediction was made")
    confidence: float = Field(1.0, ge=0, le=1, description="Prediction confidence")


class ResolutionType(str, Enum):
    """Types of conflict resolution maneuvers."""
    HEADING_CHANGE = "heading_change"
    SPEED_CHANGE = "speed_change"
    ALTITUDE_CHANGE = "altitude_change"
    COMBINED = "combined"


class ResolutionCommand(BaseModel):
    """Conflict resolution command."""
    
    resolution_id: str = Field(..., description="Unique resolution identifier")
    target_aircraft: str = Field(..., description="Aircraft to receive command")
    resolution_type: ResolutionType
    
    # Command parameters
    new_heading_deg: Optional[float] = Field(None, ge=0, lt=360)
    new_speed_kt: Optional[float] = Field(None, ge=50, le=1000)
    new_altitude_ft: Optional[float] = Field(None, ge=0, le=60000)
    
    # Timing
    issue_time: datetime = Field(..., description="When to issue command")
    expected_completion_time: Optional[datetime] = None
    
    # Validation
    is_validated: bool = Field(False, description="Has passed safety validation")
    safety_margin_nm: float = Field(..., ge=0, description="Expected safety margin")
    
    @validator('new_heading_deg')
    def normalize_new_heading(cls, v):
        """Normalize heading to [0, 360) range."""
        return v % 360.0 if v is not None else v


class DetectOut(BaseModel):
    """LLM conflict detection output schema."""
    
    conflict: bool = Field(..., description="Whether conflicts were detected")
    intruders: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of detected intruders with details"
    )
    assessment: str = Field(default="No conflict assessment provided", 
                          description="Natural language conflict assessment")
    
    class Config:
        extra = "allow"  # Allow additional fields from LLM


class ResolveOut(BaseModel):
    """LLM conflict resolution output schema."""
    
    action: str = Field(..., description="Resolution action: turn, climb, or descend")
    params: Dict[str, Any] = Field(..., description="Action parameters")
    rationale: str = Field(..., description="Reasoning for the resolution")
    recommended_resolution: str = Field(default="No specific resolution recommended",
                                      description="Recommended resolution command")
    
    class Config:
        extra = "allow"  # Allow additional fields from LLM


class LLMDetectionInput(BaseModel):
    """Input format for LLM conflict detection."""
    
    ownship: AircraftState
    traffic: List[AircraftState]
    lookahead_minutes: float = 10.0
    current_time: datetime
    context: Optional[str] = None


class LLMDetectionOutput(BaseModel):
    """Output format for LLM conflict detection."""
    
    conflicts_detected: List[ConflictPrediction]
    assessment: str = Field(..., description="Natural language conflict assessment")
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str = Field(..., description="Step-by-step detection reasoning")


class LLMResolutionInput(BaseModel):
    """Input format for LLM resolution generation."""
    
    conflict: ConflictPrediction
    ownship: AircraftState
    traffic: List[AircraftState]
    constraints: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[str] = None


class LLMResolutionOutput(BaseModel):
    """Output format for LLM resolution generation."""
    
    recommended_resolution: ResolutionCommand
    alternative_resolutions: List[ResolutionCommand] = Field(default_factory=list)
    reasoning: str = Field(..., description="Resolution reasoning and safety analysis")
    risk_assessment: str = Field(..., description="Risk evaluation of proposed resolution")
    confidence: float = Field(..., ge=0, le=1)


class ConfigurationSettings(BaseModel):
    """System configuration parameters."""
    
    # Timing settings
    polling_interval_min: float = Field(5.0, gt=0, description="Polling interval in minutes")
    lookahead_time_min: float = Field(10.0, gt=0, description="Prediction horizon in minutes")
    
    # Separation standards
    min_horizontal_separation_nm: float = Field(5.0, gt=0)
    min_vertical_separation_ft: float = Field(1000.0, gt=0)
    
    # LLM settings
    llm_model_name: str = Field("llama-3.1-8b", description="LLM model identifier")
    llm_temperature: float = Field(0.1, ge=0, le=1)
    llm_max_tokens: int = Field(2048, gt=0)
    
    # Safety settings
    safety_buffer_factor: float = Field(1.2, gt=1.0, description="Safety margin multiplier")
    max_resolution_angle_deg: float = Field(45.0, gt=0, le=90)
    max_altitude_change_ft: float = Field(2000.0, gt=0)
    
    # BlueSky integration
    bluesky_host: str = Field("localhost")
    bluesky_port: int = Field(1337, gt=0, le=65535)
    bluesky_timeout_sec: float = Field(5.0, gt=0)
    
    # Fast-time simulation
    fast_time: bool = Field(True, description="If True, do not wall-sleep; advance sim time only")
    sim_accel_factor: float = Field(1.0, gt=0, description="Multiply simulated step length per cycle")
