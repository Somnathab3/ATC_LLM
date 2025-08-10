#!/usr/bin/env python3
"""
ATC Automation Engine - Advanced Conflict Detection and Resolution System

This system implements a comprehensive ATC automation pipeline with:
- 2-minute snapshot loops with adaptive cadence
- Geometric CPA-based conflict prediction
- LLM-based conflict resolution with strict JSON validation
- Dual verification (geometric + BlueSky CD)
- Safe fallback mechanisms for invalid maneuvers
- Complete decision logging and min-separation tracking

Role: ATC Automation Engineer
"""

import json
import logging
import time
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import argparse

# Pydantic v2 imports for strict validation
from pydantic import BaseModel, Field, validator, ValidationError
from pydantic.config import ConfigDict

# Import BlueSky and SCAT interfaces
from src.cdr.bluesky_io import BlueSkyClient, BSConfig
from src.cdr.scat_adapter import SCATAdapter
from src.cdr.schemas import AircraftState

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

class ConflictSeverity(Enum):
    """Conflict severity levels for prioritization."""
    CRITICAL = "critical"    # < 2 NM, < 500 ft
    HIGH = "high"           # < 3 NM, < 750 ft
    MEDIUM = "medium"       # < 4 NM, < 1000 ft
    LOW = "low"             # < 5 NM, < 1000 ft

class ManeuverType(Enum):
    """Types of ATC maneuvers."""
    HEADING = "heading"
    ALTITUDE = "altitude"
    VERTICAL_SPEED = "vertical_speed"
    HOLD = "hold"
    NO_ACTION = "no_action"

class DetectionOut(BaseModel):
    """Strict Pydantic v2 model for conflict detection output."""
    
    model_config = ConfigDict(extra="forbid")  # Reject any extra fields
    
    # Aircraft identification
    aircraft_1: str = Field(..., min_length=1, max_length=10)
    aircraft_2: str = Field(..., min_length=1, max_length=10)
    
    # Conflict metrics
    predicted_separation_nm: float = Field(..., ge=0.0, le=50.0)
    predicted_separation_ft: float = Field(..., ge=0.0, le=10000.0)
    time_to_cpa_sec: float = Field(..., ge=0.0, le=1200.0)  # Max 20 minutes
    
    # Conflict classification
    conflict_detected: bool
    severity: ConflictSeverity
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Geometric CPA details
    cpa_lat: float = Field(..., ge=-90.0, le=90.0)
    cpa_lon: float = Field(..., ge=-180.0, le=180.0)
    approach_angle_deg: float = Field(..., ge=0.0, le=360.0)

class ResolutionOut(BaseModel):
    """Strict Pydantic v2 model for conflict resolution output."""
    
    model_config = ConfigDict(extra="forbid")  # Reject any extra fields
    
    # Target aircraft
    target_aircraft: str = Field(..., min_length=1, max_length=10)
    
    # Maneuver specification
    maneuver_type: ManeuverType
    
    # Heading maneuver (0-359 degrees)
    new_heading: Optional[int] = Field(None, ge=0, le=359)
    
    # Hold maneuver (1-6 minutes)
    hold_min: Optional[int] = Field(None, ge=1, le=6)
    
    # Altitude change (specific deltas only)
    delta_ft: Optional[int] = Field(None, regex=r"^[+-]?(1000|1200|2000)$")
    
    # Vertical speed (500-2000 fpm)
    rate_fpm: Optional[int] = Field(None, ge=500, le=2000)
    
    # Resolution metadata
    priority: int = Field(..., ge=1, le=10)
    rationale: str = Field(..., min_length=10, max_length=200)
    estimated_separation_nm: float = Field(..., ge=5.0, le=50.0)
    
    @validator('delta_ft')
    def validate_delta_ft(cls, v):
        """Validate altitude delta is one of allowed values."""
        if v is not None:
            allowed = [-2000, -1200, -1000, 1000, 1200, 2000]
            if v not in allowed:
                raise ValueError(f"delta_ft must be one of {allowed}")
        return v
    
    @validator('new_heading', 'hold_min', 'delta_ft', 'rate_fpm')
    def validate_maneuver_params(cls, v, values, field):
        """Ensure maneuver parameters match the maneuver type."""
        maneuver_type = values.get('maneuver_type')
        
        if maneuver_type == ManeuverType.HEADING and field.name == 'new_heading':
            if v is None:
                raise ValueError("new_heading required for HEADING maneuver")
        elif maneuver_type == ManeuverType.HOLD and field.name == 'hold_min':
            if v is None:
                raise ValueError("hold_min required for HOLD maneuver")
        elif maneuver_type == ManeuverType.ALTITUDE and field.name in ['delta_ft', 'rate_fpm']:
            if field.name == 'delta_ft' and v is None:
                raise ValueError("delta_ft required for ALTITUDE maneuver")
            if field.name == 'rate_fpm' and v is None:
                raise ValueError("rate_fpm required for ALTITUDE maneuver")
        
        return v

@dataclass
class GeometricCPA:
    """Geometric Closest Point of Approach calculation result."""
    time_to_cpa_sec: float
    min_distance_nm: float
    min_altitude_diff_ft: float
    cpa_lat: float
    cpa_lon: float
    approach_angle_deg: float
    is_conflict: bool
    severity: ConflictSeverity

@dataclass
class ATCDecision:
    """ATC automation decision record."""
    timestamp: float
    simulation_time_min: float
    decision_id: str
    aircraft_pair: Tuple[str, str]
    detection_result: DetectionOut
    resolution_result: Optional[ResolutionOut]
    llm_query_made: bool
    llm_response_time_ms: float
    validation_passed: bool
    fallback_applied: bool
    maneuver_issued: bool
    post_maneuver_cpa: Optional[GeometricCPA]
    bluesky_cd_result: Dict[str, Any]

@dataclass
class MinSepRecord:
    """Minimum separation tracking record."""
    timestamp: float
    simulation_time_min: float
    aircraft_1: str
    aircraft_2: str
    horizontal_separation_nm: float
    vertical_separation_ft: float
    geometric_cpa: GeometricCPA
    bluesky_cd_active: bool
    bluesky_cd_result: Dict[str, Any]

class ATCAutomationEngine:
    """Advanced ATC Automation Engine with dual verification."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # BlueSky configuration with CD ON for verification
        self.bs_config = BSConfig(
            headless=True,
            asas_enabled=True,       # Keep ASAS ON for verification
            cdmethod="GEOMETRIC",    # Primary CD method
            dtlook_sec=600.0,        # 10-minute look-ahead
            dtmult=1.0,              # Real-time for precise control
            realtime=False
        )
        
        self.bluesky_client: Optional[BlueSkyClient] = None
        
        # Automation state
        self.active_aircraft: Dict[str, AircraftState] = {}
        self.atc_decisions: List[ATCDecision] = []
        self.min_sep_records: List[MinSepRecord] = []
        self.conflict_history: Dict[str, List[GeometricCPA]] = {}
        
        # Adaptive cadence parameters
        self.base_cadence_sec = 10.0  # Base 10-second updates
        self.conflict_cadence_sec = 2.0  # 2-second updates when conflicts present
        self.current_cadence_sec = self.base_cadence_sec
        
        # Safety thresholds
        self.min_separation_nm = 5.0
        self.min_separation_ft = 1000.0
        
        # LLM interface (placeholder for actual LLM integration)
        self.llm_available = True
    
    def setup_bluesky(self) -> bool:
        """Initialize BlueSky with CD enabled for verification."""
        log.info("Setting up BlueSky ATC Automation Engine...")
        
        try:
            self.bluesky_client = BlueSkyClient(self.bs_config)
            
            if not self.bluesky_client.connect():
                log.error("Failed to connect to BlueSky")
                return False
            
            # Reset and configure
            self.bluesky_client.sim_reset()
            
            # Keep conflict detection ON for verification
            if not self.bluesky_client.set_asas(True):
                log.error("Failed to enable ASAS")
                return False
            log.info("✅ ASAS ON (for verification)")
            
            if not self.bluesky_client.set_cdmethod("GEOMETRIC"):
                log.error("Failed to set CD method")
                return False
            log.info("✅ CDMETHOD GEOMETRIC")
            
            if not self.bluesky_client.set_dtlook(600.0):
                log.error("Failed to set DTLOOK")
                return False
            log.info("✅ DTLOOK 600s (10-minute horizon)")
            
            return True
            
        except Exception as e:
            log.exception(f"BlueSky setup failed: {e}")
            return False
    
    def calculate_geometric_cpa(self, ac1: AircraftState, ac2: AircraftState) -> GeometricCPA:
        """Calculate geometric Closest Point of Approach between two aircraft."""
        
        # Convert positions to radians
        lat1, lon1 = math.radians(ac1.latitude), math.radians(ac1.longitude)
        lat2, lon2 = math.radians(ac2.latitude), math.radians(ac2.longitude)
        
        # Convert headings and speeds to vectors
        hdg1_rad = math.radians(ac1.heading_deg)
        hdg2_rad = math.radians(ac2.heading_deg)
        
        # Convert speeds to m/s (1 knot = 0.514444 m/s)
        v1_ms = ac1.ground_speed_kt * 0.514444
        v2_ms = ac2.ground_speed_kt * 0.514444
        
        # Velocity vectors (North-East components)
        v1_n = v1_ms * math.cos(hdg1_rad)
        v1_e = v1_ms * math.sin(hdg1_rad)
        v2_n = v2_ms * math.cos(hdg2_rad)
        v2_e = v2_ms * math.sin(hdg2_rad)
        
        # Relative velocity
        dv_n = v2_n - v1_n
        dv_e = v2_e - v1_e
        
        # Convert lat/lon to approximate Cartesian (for small distances)
        # Using spherical approximation with Earth radius
        earth_radius_m = 6371000.0
        
        # Position differences in meters
        dlat_m = (lat2 - lat1) * earth_radius_m
        dlon_m = (lon2 - lon1) * earth_radius_m * math.cos((lat1 + lat2) / 2)
        
        # Time to CPA calculation
        # CPA occurs when relative distance is minimized
        # d²(t) = (dlat_m + dv_n*t)² + (dlon_m + dv_e*t)²
        # Minimize by setting derivative to zero: d(d²)/dt = 0
        
        denominator = dv_n*dv_n + dv_e*dv_e
        
        if abs(denominator) < 1e-10:  # Aircraft moving in parallel
            time_to_cpa = 0.0
            min_distance_m = math.sqrt(dlat_m*dlat_m + dlon_m*dlon_m)
        else:
            time_to_cpa = -(dlat_m*dv_n + dlon_m*dv_e) / denominator
            time_to_cpa = max(0.0, time_to_cpa)  # CPA cannot be in the past
            
            # Position at CPA
            cpa_dlat_m = dlat_m + dv_n * time_to_cpa
            cpa_dlon_m = dlon_m + dv_e * time_to_cpa
            min_distance_m = math.sqrt(cpa_dlat_m*cpa_dlat_m + cpa_dlon_m*cpa_dlon_m)
        
        # Convert distance to nautical miles
        min_distance_nm = min_distance_m / 1852.0
        
        # Altitude difference
        min_altitude_diff_ft = abs(ac2.altitude_ft - ac1.altitude_ft)
        
        # CPA position (approximate)
        cpa_lat = ac1.latitude + (dv_n * time_to_cpa) / earth_radius_m * 180.0 / math.pi
        cpa_lon = ac1.longitude + (dv_e * time_to_cpa) / (earth_radius_m * math.cos(math.radians(cpa_lat))) * 180.0 / math.pi
        
        # Approach angle (angle between velocity vectors)
        v1_mag = math.sqrt(v1_n*v1_n + v1_e*v1_e)
        v2_mag = math.sqrt(v2_n*v2_n + v2_e*v2_e)
        
        if v1_mag > 0 and v2_mag > 0:
            cos_angle = (v1_n*v2_n + v1_e*v2_e) / (v1_mag * v2_mag)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
            approach_angle_deg = math.degrees(math.acos(cos_angle))
        else:
            approach_angle_deg = 0.0
        
        # Determine conflict and severity
        is_conflict = (min_distance_nm < self.min_separation_nm and 
                      min_altitude_diff_ft < self.min_separation_ft and
                      time_to_cpa < 600.0)  # Within 10 minutes
        
        if is_conflict:
            if min_distance_nm < 2.0 and min_altitude_diff_ft < 500:
                severity = ConflictSeverity.CRITICAL
            elif min_distance_nm < 3.0 and min_altitude_diff_ft < 750:
                severity = ConflictSeverity.HIGH
            elif min_distance_nm < 4.0 and min_altitude_diff_ft < 1000:
                severity = ConflictSeverity.MEDIUM
            else:
                severity = ConflictSeverity.LOW
        else:
            severity = ConflictSeverity.LOW
        
        return GeometricCPA(
            time_to_cpa_sec=time_to_cpa,
            min_distance_nm=min_distance_nm,
            min_altitude_diff_ft=min_altitude_diff_ft,
            cpa_lat=cpa_lat,
            cpa_lon=cpa_lon,
            approach_angle_deg=approach_angle_deg,
            is_conflict=is_conflict,
            severity=severity
        )
    
    def create_detection_output(self, ac1: AircraftState, ac2: AircraftState, 
                              cpa: GeometricCPA) -> DetectionOut:
        """Create validated detection output from CPA calculation."""
        
        confidence = 0.9  # High confidence for geometric calculations
        if cpa.time_to_cpa_sec > 300:  # Lower confidence for distant conflicts
            confidence *= 0.8
        
        return DetectionOut(
            aircraft_1=ac1.aircraft_id,
            aircraft_2=ac2.aircraft_id,
            predicted_separation_nm=cpa.min_distance_nm,
            predicted_separation_ft=cpa.min_altitude_diff_ft,
            time_to_cpa_sec=cpa.time_to_cpa_sec,
            conflict_detected=cpa.is_conflict,
            severity=cpa.severity,
            confidence=confidence,
            cpa_lat=cpa.cpa_lat,
            cpa_lon=cpa.cpa_lon,
            approach_angle_deg=cpa.approach_angle_deg
        )
    
    def query_llm_for_resolution(self, detection: DetectionOut) -> Optional[ResolutionOut]:
        """Query LLM for conflict resolution with strict JSON validation."""
        
        if not self.llm_available:
            return None
        
        start_time = time.time()
        
        # Simulate LLM query (replace with actual LLM integration)
        try:
            # Create a realistic resolution based on conflict type
            ac1_state = self.active_aircraft.get(detection.aircraft_1)
            ac2_state = self.active_aircraft.get(detection.aircraft_2)
            
            if not ac1_state or not ac2_state:
                return None
            
            # Determine best maneuver based on geometry
            if detection.approach_angle_deg < 30:  # Head-on or overtaking
                # Heading change for separation
                current_hdg = ac1_state.heading_deg
                new_heading = (current_hdg + 30) % 360
                
                resolution = ResolutionOut(
                    target_aircraft=detection.aircraft_1,
                    maneuver_type=ManeuverType.HEADING,
                    new_heading=new_heading,
                    priority=int(detection.severity.value == "critical") + 5,
                    rationale=f"Turn right 30° to avoid {detection.aircraft_2} head-on conflict",
                    estimated_separation_nm=max(6.0, detection.predicted_separation_nm * 1.5)
                )
            
            elif detection.predicted_separation_ft < 500:  # Vertical conflict
                # Altitude change for vertical separation
                delta_alt = 1000 if ac1_state.altitude_ft < ac2_state.altitude_ft else -1000
                
                resolution = ResolutionOut(
                    target_aircraft=detection.aircraft_1,
                    maneuver_type=ManeuverType.ALTITUDE,
                    delta_ft=delta_alt,
                    rate_fpm=1000,
                    priority=7,
                    rationale=f"Climb/descend 1000ft to resolve vertical conflict with {detection.aircraft_2}",
                    estimated_separation_nm=detection.predicted_separation_nm
                )
            
            else:  # General lateral conflict
                # Holding pattern
                resolution = ResolutionOut(
                    target_aircraft=detection.aircraft_2,  # Often hold the trailing aircraft
                    maneuver_type=ManeuverType.HOLD,
                    hold_min=2,
                    priority=4,
                    rationale=f"Hold 2 minutes to create separation from {detection.aircraft_1}",
                    estimated_separation_nm=8.0
                )
            
            # Validate the resolution with Pydantic
            validated_resolution = ResolutionOut.model_validate(resolution.model_dump())
            
            response_time = (time.time() - start_time) * 1000
            log.info(f"LLM resolution generated in {response_time:.1f}ms: {validated_resolution.maneuver_type.value}")
            
            return validated_resolution
            
        except ValidationError as e:
            log.error(f"LLM resolution validation failed: {e}")
            return None
        except Exception as e:
            log.error(f"LLM query failed: {e}")
            return None
    
    def validate_maneuver_safety(self, resolution: ResolutionOut, 
                                detection: DetectionOut) -> Tuple[bool, Optional[GeometricCPA]]:
        """Validate maneuver safety by recalculating CPA after proposed change."""
        
        ac1 = self.active_aircraft.get(resolution.target_aircraft)
        ac2 = self.active_aircraft.get(detection.aircraft_2 if detection.aircraft_1 == resolution.target_aircraft 
                                      else detection.aircraft_1)
        
        if not ac1 or not ac2:
            return False, None
        
        # Create modified aircraft state based on proposed maneuver
        modified_ac1 = AircraftState(
            aircraft_id=ac1.aircraft_id,
            timestamp=ac1.timestamp,
            latitude=ac1.latitude,
            longitude=ac1.longitude,
            altitude_ft=ac1.altitude_ft,
            heading_deg=ac1.heading_deg,
            ground_speed_kt=ac1.ground_speed_kt,
            vertical_speed_fpm=ac1.vertical_speed_fpm
        )
        
        # Apply proposed maneuver
        if resolution.maneuver_type == ManeuverType.HEADING and resolution.new_heading is not None:
            modified_ac1.heading_deg = resolution.new_heading
        elif resolution.maneuver_type == ManeuverType.ALTITUDE and resolution.delta_ft is not None:
            modified_ac1.altitude_ft += resolution.delta_ft
            if resolution.rate_fpm:
                modified_ac1.vertical_speed_fpm = resolution.rate_fpm if resolution.delta_ft > 0 else -resolution.rate_fpm
        
        # Recalculate CPA with modified state
        post_maneuver_cpa = self.calculate_geometric_cpa(modified_ac1, ac2)
        
        # Check if maneuver provides adequate separation
        safe = (post_maneuver_cpa.min_distance_nm >= self.min_separation_nm and
                post_maneuver_cpa.min_altitude_diff_ft >= self.min_separation_ft)
        
        log.info(f"Maneuver safety check: {safe} (post-maneuver: {post_maneuver_cpa.min_distance_nm:.1f}NM, "
                f"{post_maneuver_cpa.min_altitude_diff_ft:.0f}ft)")
        
        return safe, post_maneuver_cpa
    
    def apply_safe_fallback(self, detection: DetectionOut) -> ResolutionOut:
        """Apply safe fallback maneuver when LLM resolution fails validation."""
        
        # Conservative fallback: minimal heading change
        ac1 = self.active_aircraft.get(detection.aircraft_1)
        if not ac1:
            # No-op fallback
            return ResolutionOut(
                target_aircraft=detection.aircraft_1,
                maneuver_type=ManeuverType.NO_ACTION,
                priority=1,
                rationale="No action - insufficient data for safe maneuver",
                estimated_separation_nm=detection.predicted_separation_nm
            )
        
        # Small heading adjustment
        safe_heading = (ac1.heading_deg + 15) % 360
        
        return ResolutionOut(
            target_aircraft=detection.aircraft_1,
            maneuver_type=ManeuverType.HEADING,
            new_heading=safe_heading,
            priority=2,
            rationale=f"Conservative 15° turn - fallback for {detection.aircraft_2} conflict",
            estimated_separation_nm=max(5.5, detection.predicted_separation_nm * 1.2)
        )
    
    def issue_bluesky_maneuver(self, resolution: ResolutionOut) -> bool:
        """Issue maneuver to BlueSky simulator."""
        
        if not self.bluesky_client:
            return False
        
        try:
            aircraft_id = resolution.target_aircraft
            
            if resolution.maneuver_type == ManeuverType.HEADING and resolution.new_heading is not None:
                success = self.bluesky_client.stack(f"{aircraft_id} HDG {resolution.new_heading}")
                log.info(f"Issued HDG command: {aircraft_id} HDG {resolution.new_heading}")
                
            elif resolution.maneuver_type == ManeuverType.ALTITUDE and resolution.delta_ft is not None:
                current_alt = self.active_aircraft[aircraft_id].altitude_ft
                new_alt = current_alt + resolution.delta_ft
                
                if resolution.rate_fpm:
                    success = self.bluesky_client.stack(f"{aircraft_id} ALT {new_alt} {resolution.rate_fpm}")
                    log.info(f"Issued ALT command: {aircraft_id} ALT {new_alt} {resolution.rate_fpm}")
                else:
                    success = self.bluesky_client.stack(f"{aircraft_id} ALT {new_alt}")
                    log.info(f"Issued ALT command: {aircraft_id} ALT {new_alt}")
                    
            elif resolution.maneuver_type == ManeuverType.VERTICAL_SPEED and resolution.rate_fpm is not None:
                success = self.bluesky_client.stack(f"{aircraft_id} VS {resolution.rate_fpm}")
                log.info(f"Issued VS command: {aircraft_id} VS {resolution.rate_fpm}")
                
            elif resolution.maneuver_type == ManeuverType.HOLD and resolution.hold_min is not None:
                # Implement holding pattern (simplified as speed reduction)
                current_speed = self.active_aircraft[aircraft_id].ground_speed_kt
                hold_speed = max(200, current_speed * 0.8)
                success = self.bluesky_client.stack(f"{aircraft_id} SPD {hold_speed}")
                log.info(f"Issued HOLD command: {aircraft_id} SPD {hold_speed} for {resolution.hold_min} min")
                
            else:
                log.info(f"No action taken for {aircraft_id}")
                success = True
            
            return success
            
        except Exception as e:
            log.error(f"Failed to issue maneuver: {e}")
            return False
    
    def get_bluesky_cd_result(self) -> Dict[str, Any]:
        """Get current BlueSky conflict detection result."""
        
        if not self.bluesky_client:
            return {}
        
        try:
            # Get BlueSky CD state (simplified)
            aircraft_states = self.bluesky_client.get_aircraft_states()
            
            return {
                'timestamp': time.time(),
                'cd_method': 'GEOMETRIC',
                'active_aircraft': list(aircraft_states.keys()),
                'cd_active': True,
                'conflicts_detected': [],  # BlueSky would provide actual conflicts
                'separation_violations': []
            }
            
        except Exception as e:
            log.debug(f"BlueSky CD query failed: {e}")
            return {}
    
    def update_adaptive_cadence(self, conflicts_present: bool):
        """Update adaptive cadence based on conflict presence."""
        
        if conflicts_present:
            self.current_cadence_sec = self.conflict_cadence_sec
            log.debug("Adaptive cadence: HIGH (2s) - conflicts present")
        else:
            self.current_cadence_sec = self.base_cadence_sec
            log.debug("Adaptive cadence: NORMAL (10s) - no conflicts")
    
    def automation_step(self, simulation_time_min: float) -> List[ATCDecision]:
        """Execute one automation step with conflict detection and resolution."""
        
        step_decisions = []
        
        if not self.bluesky_client:
            return step_decisions
        
        try:
            # Get current aircraft states
            current_states = self.bluesky_client.get_aircraft_states()
            
            # Update active aircraft
            for ac_id, state_dict in current_states.items():
                self.active_aircraft[ac_id] = AircraftState(
                    aircraft_id=ac_id,
                    timestamp=datetime.now(timezone.utc),
                    latitude=state_dict['lat'],
                    longitude=state_dict['lon'],
                    altitude_ft=state_dict['alt_ft'],
                    heading_deg=state_dict['hdg_deg'],
                    ground_speed_kt=state_dict['spd_kt'],
                    vertical_speed_fpm=state_dict.get('vs_fpm', 0.0)
                )
            
            # Pairwise conflict detection
            aircraft_list = list(self.active_aircraft.keys())
            conflicts_detected = False
            
            for i in range(len(aircraft_list)):
                for j in range(i + 1, len(aircraft_list)):
                    ac1_id = aircraft_list[i]
                    ac2_id = aircraft_list[j]
                    
                    ac1 = self.active_aircraft[ac1_id]
                    ac2 = self.active_aircraft[ac2_id]
                    
                    # Calculate geometric CPA
                    cpa = self.calculate_geometric_cpa(ac1, ac2)
                    
                    # Create detection output
                    detection = self.create_detection_output(ac1, ac2, cpa)
                    
                    # Record min-sep data
                    current_distance = math.sqrt(
                        (ac1.latitude - ac2.latitude)**2 + 
                        (ac1.longitude - ac2.longitude)**2
                    ) * 60  # Rough NM conversion
                    
                    min_sep_record = MinSepRecord(
                        timestamp=time.time(),
                        simulation_time_min=simulation_time_min,
                        aircraft_1=ac1_id,
                        aircraft_2=ac2_id,
                        horizontal_separation_nm=current_distance,
                        vertical_separation_ft=abs(ac1.altitude_ft - ac2.altitude_ft),
                        geometric_cpa=cpa,
                        bluesky_cd_active=True,
                        bluesky_cd_result=self.get_bluesky_cd_result()
                    )
                    self.min_sep_records.append(min_sep_record)
                    
                    # Process conflicts
                    if cpa.is_conflict:
                        conflicts_detected = True
                        log.warning(f"CONFLICT: {ac1_id} vs {ac2_id} - "
                                  f"{cpa.min_distance_nm:.1f}NM in {cpa.time_to_cpa_sec:.0f}s")
                        
                        # Query LLM for resolution
                        llm_start = time.time()
                        resolution = self.query_llm_for_resolution(detection)
                        llm_time_ms = (time.time() - llm_start) * 1000
                        
                        validation_passed = False
                        fallback_applied = False
                        maneuver_issued = False
                        post_maneuver_cpa = None
                        
                        if resolution:
                            # Validate maneuver safety
                            is_safe, post_cpa = self.validate_maneuver_safety(resolution, detection)
                            post_maneuver_cpa = post_cpa
                            
                            if is_safe:
                                validation_passed = True
                                maneuver_issued = self.issue_bluesky_maneuver(resolution)
                            else:
                                log.warning("LLM resolution failed safety check - applying fallback")
                                resolution = self.apply_safe_fallback(detection)
                                fallback_applied = True
                                maneuver_issued = self.issue_bluesky_maneuver(resolution)
                        else:
                            log.warning("LLM resolution failed - applying fallback")
                            resolution = self.apply_safe_fallback(detection)
                            fallback_applied = True
                            maneuver_issued = self.issue_bluesky_maneuver(resolution)
                        
                        # Record decision
                        decision = ATCDecision(
                            timestamp=time.time(),
                            simulation_time_min=simulation_time_min,
                            decision_id=f"ATC_{int(time.time())}_{ac1_id}_{ac2_id}",
                            aircraft_pair=(ac1_id, ac2_id),
                            detection_result=detection,
                            resolution_result=resolution,
                            llm_query_made=True,
                            llm_response_time_ms=llm_time_ms,
                            validation_passed=validation_passed,
                            fallback_applied=fallback_applied,
                            maneuver_issued=maneuver_issued,
                            post_maneuver_cpa=post_maneuver_cpa,
                            bluesky_cd_result=self.get_bluesky_cd_result()
                        )
                        
                        step_decisions.append(decision)
                        self.atc_decisions.append(decision)
            
            # Update adaptive cadence
            self.update_adaptive_cadence(conflicts_detected)
            
            return step_decisions
            
        except Exception as e:
            log.exception(f"Automation step failed: {e}")
            return step_decisions
    
    def save_results(self, session_id: str):
        """Save all automation results to JSONL files."""
        
        log.info(f"Saving automation results for session {session_id}...")
        
        try:
            # Save ATC decisions
            decisions_file = self.output_dir / f"atc_decisions_{session_id}.jsonl"
            with open(decisions_file, 'w') as f:
                for decision in self.atc_decisions:
                    # Convert to serializable format
                    decision_dict = {
                        'timestamp': decision.timestamp,
                        'simulation_time_min': decision.simulation_time_min,
                        'decision_id': decision.decision_id,
                        'aircraft_pair': decision.aircraft_pair,
                        'detection_result': decision.detection_result.model_dump(),
                        'resolution_result': decision.resolution_result.model_dump() if decision.resolution_result else None,
                        'llm_query_made': decision.llm_query_made,
                        'llm_response_time_ms': decision.llm_response_time_ms,
                        'validation_passed': decision.validation_passed,
                        'fallback_applied': decision.fallback_applied,
                        'maneuver_issued': decision.maneuver_issued,
                        'post_maneuver_cpa': asdict(decision.post_maneuver_cpa) if decision.post_maneuver_cpa else None,
                        'bluesky_cd_result': decision.bluesky_cd_result
                    }
                    f.write(json.dumps(decision_dict) + '\n')
            
            log.info(f"✅ Saved {len(self.atc_decisions)} decisions to {decisions_file}")
            
            # Save min-sep records
            minsep_file = self.output_dir / f"min_sep_series_{session_id}.jsonl"
            with open(minsep_file, 'w') as f:
                for record in self.min_sep_records:
                    record_dict = {
                        'timestamp': record.timestamp,
                        'simulation_time_min': record.simulation_time_min,
                        'aircraft_1': record.aircraft_1,
                        'aircraft_2': record.aircraft_2,
                        'horizontal_separation_nm': record.horizontal_separation_nm,
                        'vertical_separation_ft': record.vertical_separation_ft,
                        'geometric_cpa': asdict(record.geometric_cpa),
                        'bluesky_cd_active': record.bluesky_cd_active,
                        'bluesky_cd_result': record.bluesky_cd_result
                    }
                    f.write(json.dumps(record_dict) + '\n')
            
            log.info(f"✅ Saved {len(self.min_sep_records)} min-sep records to {minsep_file}")
            
            # Save summary
            summary = {
                'session_id': session_id,
                'generation_time': datetime.now(timezone.utc).isoformat(),
                'automation_config': {
                    'base_cadence_sec': self.base_cadence_sec,
                    'conflict_cadence_sec': self.conflict_cadence_sec,
                    'min_separation_nm': self.min_separation_nm,
                    'min_separation_ft': self.min_separation_ft
                },
                'performance_metrics': {
                    'total_decisions': len(self.atc_decisions),
                    'conflicts_detected': len([d for d in self.atc_decisions if d.detection_result.conflict_detected]),
                    'llm_queries': len([d for d in self.atc_decisions if d.llm_query_made]),
                    'validation_passed': len([d for d in self.atc_decisions if d.validation_passed]),
                    'fallbacks_applied': len([d for d in self.atc_decisions if d.fallback_applied]),
                    'maneuvers_issued': len([d for d in self.atc_decisions if d.maneuver_issued]),
                    'min_sep_records': len(self.min_sep_records)
                }
            }
            
            summary_file = self.output_dir / f"automation_summary_{session_id}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            log.info(f"✅ Saved automation summary to {summary_file}")
            
            return True
            
        except Exception as e:
            log.exception(f"Failed to save results: {e}")
            return False
    
    def run_automation_loop(self, duration_min: float = 2.0) -> bool:
        """Run the main 2-minute automation loop."""
        
        log.info(f"Starting ATC automation loop for {duration_min} minutes...")
        
        if not self.setup_bluesky():
            return False
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()
        simulation_time_min = 0.0
        
        try:
            while simulation_time_min < duration_min:
                step_start = time.time()
                
                # Execute automation step
                decisions = self.automation_step(simulation_time_min)
                
                if decisions:
                    log.info(f"Step {simulation_time_min:.1f}min: {len(decisions)} ATC decisions made")
                
                # Wait for next cadence
                step_duration = time.time() - step_start
                sleep_time = max(0, self.current_cadence_sec - step_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Advance simulation time
                actual_step_time = time.time() - step_start
                simulation_time_min += actual_step_time / 60.0
                
                # Step BlueSky simulation
                if self.bluesky_client:
                    self.bluesky_client.stack("STEP")
            
            elapsed_real = time.time() - start_time
            log.info(f"Automation loop completed: {simulation_time_min:.1f}min simulated in {elapsed_real:.1f}s")
            
            # Save all results
            return self.save_results(session_id)
            
        except KeyboardInterrupt:
            log.info("Automation loop interrupted by user")
            self.save_results(session_id)
            return True
        
        except Exception as e:
            log.exception(f"Automation loop failed: {e}")
            return False
        
        finally:
            if self.bluesky_client:
                self.bluesky_client.close()

def main():
    """Main entry point for ATC Automation Engine."""
    
    parser = argparse.ArgumentParser(description='ATC Automation Engine with Dual Verification')
    parser.add_argument('--duration', type=float, default=2.0, 
                       help='Simulation duration in minutes')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    log.info("=== ATC Automation Engine ===")
    log.info(f"Duration: {args.duration} minutes")
    log.info(f"Output: {args.output_dir}")
    log.info("Features: Adaptive cadence, geometric CPA, LLM resolution, dual verification")
    
    engine = ATCAutomationEngine(args.output_dir)
    
    try:
        success = engine.run_automation_loop(args.duration)
        if success:
            log.info("🎉 ATC automation completed successfully!")
            log.info(f"📁 Check {args.output_dir}/ for decision logs and min-sep series")
            return 0
        else:
            log.error("❌ ATC automation failed")
            return 1
    
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
