"""Dual verification system for conflict detection.

This module implements Gap 7 requirements:
- Dual verification: BlueSky CD + geometric CPA/min-sep check
- Discrepancy logging between BlueSky ASAS and geometric calculations
- Cross-validation of conflict detection methods

The system compares:
1. BlueSky ASAS conflict detection results
2. Independent geometric CPA calculations
3. Logs discrepancies for analysis and validation
"""

import logging
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .schemas import ConflictPrediction, AircraftState
from .geodesy import haversine_nm
from .asas_integration import BlueSkyASAS

logger = logging.getLogger(__name__)


@dataclass
class VerificationDiscrepancy:
    """Record of a discrepancy between BlueSky CD and geometric CPA."""
    
    timestamp: datetime
    aircraft_pair: Tuple[str, str]
    
    # BlueSky ASAS results
    bluesky_conflict: bool
    bluesky_cpa_nm: float
    bluesky_tcpa_min: float
    
    # Geometric calculations
    geometric_conflict: bool
    geometric_cpa_nm: float
    geometric_tcpa_min: float
    
    # Discrepancy analysis
    conflict_agreement: bool
    cpa_difference_nm: float
    tcpa_difference_min: float
    severity: str  # low, medium, high
    
    # Context
    aircraft1_state: Dict[str, Any]
    aircraft2_state: Dict[str, Any]


class DualVerificationSystem:
    """GAP 7 FIX: Dual verification system for conflict detection."""
    
    def __init__(self, min_horizontal_sep_nm: float = 5.0, 
                 min_vertical_sep_ft: float = 1000.0,
                 lookahead_time_min: float = 10.0):
        """Initialize dual verification system.
        
        Args:
            min_horizontal_sep_nm: Minimum horizontal separation in NM
            min_vertical_sep_ft: Minimum vertical separation in feet
            lookahead_time_min: Lookahead time for conflict prediction
        """
        self.min_horizontal_sep_nm = min_horizontal_sep_nm
        self.min_vertical_sep_ft = min_vertical_sep_ft
        self.lookahead_time_min = lookahead_time_min
        
        # Initialize BlueSky ASAS integration (will be set up later)
        self.bluesky_asas = None
        
        # Track discrepancies
        self.discrepancies: List[VerificationDiscrepancy] = []
        self.verification_history: List[Dict[str, Any]] = []
        
        logger.info("Dual verification system initialized")
    
    def setup_bluesky_asas(self, bluesky_client, config):
        """Setup BlueSky ASAS integration after clients are available."""
        try:
            from .asas_integration import BlueSkyASAS
            self.bluesky_asas = BlueSkyASAS(bluesky_client, config)
            logger.info("BlueSky ASAS integration setup completed")
        except Exception as e:
            logger.warning(f"BlueSky ASAS setup failed: {e}")
            self.bluesky_asas = None
    
    def verify_conflicts(self, traffic_states: List[Dict[str, Any]], 
                        bluesky_client) -> List[ConflictPrediction]:
        """Perform dual verification of conflict detection.
        
        Args:
            traffic_states: Current aircraft states
            bluesky_client: BlueSky client for ASAS data
            
        Returns:
            List of verified conflict predictions
        """
        timestamp = datetime.now()
        
        # Step 1: Get BlueSky ASAS conflict detection results
        bluesky_conflicts = self._get_bluesky_conflicts(bluesky_client, traffic_states)
        
        # Step 2: Perform independent geometric CPA calculations
        geometric_conflicts = self._calculate_geometric_conflicts(traffic_states)
        
        # Step 3: Compare and log discrepancies
        verified_conflicts = self._compare_and_verify(
            bluesky_conflicts, geometric_conflicts, traffic_states, timestamp
        )
        
        # Step 4: Log verification results
        self._log_verification_results(
            len(bluesky_conflicts), len(geometric_conflicts), 
            len(verified_conflicts), len(self.discrepancies), timestamp
        )
        
        return verified_conflicts
    
    def _get_bluesky_conflicts(self, bluesky_client, 
                              traffic_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get conflict detection results from BlueSky ASAS."""
        try:
            # Get ASAS conflicts if available
            if self.bluesky_asas and hasattr(bluesky_client, 'traf') and bluesky_client.traf:
                asas_conflicts = self.bluesky_asas.get_conflicts()
                return [self._format_asas_conflict(c) for c in asas_conflicts]
            else:
                logger.debug("BlueSky ASAS not available - using mock conflicts")
                return []
                
        except Exception as e:
            logger.error(f"Error getting BlueSky conflicts: {e}")
            return []
    
    def _format_asas_conflict(self, asas_conflict) -> Dict[str, Any]:
        """Format ASAS conflict for comparison."""
        return {
            'aircraft_pair': asas_conflict.aircraft_pair,
            'conflict': asas_conflict.is_conflict,
            'cpa_nm': asas_conflict.distance_at_cpa_nm,
            'tcpa_min': asas_conflict.time_to_cpa_min,
            'source': 'bluesky_asas'
        }
    
    def _calculate_geometric_conflicts(self, traffic_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform independent geometric CPA calculations."""
        geometric_conflicts = []
        
        # Check all pairs of aircraft
        for i, aircraft1 in enumerate(traffic_states):
            for aircraft2 in traffic_states[i+1:]:
                conflict_data = self._calculate_geometric_cpa(aircraft1, aircraft2)
                if conflict_data:
                    geometric_conflicts.append(conflict_data)
        
        return geometric_conflicts
    
    def _calculate_geometric_cpa(self, aircraft1: Dict[str, Any], 
                                aircraft2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate geometric CPA between two aircraft."""
        try:
            # Extract positions and velocities
            lat1, lon1 = aircraft1['lat'], aircraft1['lon']
            lat2, lon2 = aircraft2['lat'], aircraft2['lon']
            alt1, alt2 = aircraft1['alt_ft'], aircraft2['alt_ft']
            
            # Convert heading/speed to velocity components
            hdg1_rad = math.radians(aircraft1['hdg_deg'])
            hdg2_rad = math.radians(aircraft2['hdg_deg'])
            spd1_kt, spd2_kt = aircraft1['spd_kt'], aircraft2['spd_kt']
            
            # Velocity components (NM/h)
            vx1 = spd1_kt * math.sin(hdg1_rad)
            vy1 = spd1_kt * math.cos(hdg1_rad)
            vx2 = spd2_kt * math.sin(hdg2_rad)
            vy2 = spd2_kt * math.cos(hdg2_rad)
            
            # Relative position and velocity
            # Simplified: convert lat/lon differences to NM (approximate)
            dx_nm = (lon2 - lon1) * 60 * math.cos(math.radians((lat1 + lat2) / 2))
            dy_nm = (lat2 - lat1) * 60
            
            dvx_kt = vx2 - vx1
            dvy_kt = vy2 - vy1
            
            # Time to CPA calculation
            relative_speed_sq = dvx_kt**2 + dvy_kt**2
            
            if relative_speed_sq < 1e-6:  # Aircraft moving at same velocity
                # Static separation
                cpa_distance_nm = math.sqrt(dx_nm**2 + dy_nm**2)
                tcpa_min = float('inf')
            else:
                # Time to closest approach
                tcpa_hours = -(dx_nm * dvx_kt + dy_nm * dvy_kt) / relative_speed_sq
                tcpa_min = tcpa_hours * 60  # Convert to minutes
                
                if tcpa_min < 0:  # CPA in the past
                    tcpa_min = 0
                    cpa_distance_nm = math.sqrt(dx_nm**2 + dy_nm**2)
                else:
                    # Position at CPA
                    x_cpa = dx_nm + dvx_kt * tcpa_hours
                    y_cpa = dy_nm + dvy_kt * tcpa_hours
                    cpa_distance_nm = math.sqrt(x_cpa**2 + y_cpa**2)
            
            # Vertical separation at CPA (simplified - assume constant rates)
            alt_diff_ft = abs(alt2 - alt1)
            
            # Determine if this is a conflict
            is_conflict = (
                tcpa_min <= self.lookahead_time_min and
                cpa_distance_nm < self.min_horizontal_sep_nm and
                alt_diff_ft < self.min_vertical_sep_ft
            )
            
            return {
                'aircraft_pair': (aircraft1['id'], aircraft2['id']),
                'conflict': is_conflict,
                'cpa_nm': cpa_distance_nm,
                'tcpa_min': tcpa_min,
                'alt_diff_ft': alt_diff_ft,
                'source': 'geometric'
            }
            
        except Exception as e:
            logger.error(f"Error calculating geometric CPA: {e}")
            return None
    
    def _compare_and_verify(self, bluesky_conflicts: List[Dict[str, Any]],
                           geometric_conflicts: List[Dict[str, Any]],
                           traffic_states: List[Dict[str, Any]],
                           timestamp: datetime) -> List[ConflictPrediction]:
        """Compare BlueSky and geometric results, log discrepancies."""
        
        # Create lookup dictionaries
        bluesky_lookup = {tuple(sorted(c['aircraft_pair'])): c for c in bluesky_conflicts}
        geometric_lookup = {tuple(sorted(c['aircraft_pair'])): c for c in geometric_conflicts}
        
        # Get all aircraft pairs
        all_pairs = set(bluesky_lookup.keys()) | set(geometric_lookup.keys())
        
        verified_conflicts = []
        
        for pair in all_pairs:
            bluesky_result = bluesky_lookup.get(pair)
            geometric_result = geometric_lookup.get(pair)
            
            # Check for discrepancies
            discrepancy = self._analyze_discrepancy(
                pair, bluesky_result, geometric_result, traffic_states, timestamp
            )
            
            if discrepancy:
                self.discrepancies.append(discrepancy)
                self._log_discrepancy(discrepancy)
            
            # Create verified conflict prediction
            verified_conflict = self._create_verified_conflict(
                pair, bluesky_result, geometric_result, timestamp
            )
            
            if verified_conflict:
                verified_conflicts.append(verified_conflict)
        
        return verified_conflicts
    
    def _analyze_discrepancy(self, pair: Tuple[str, str],
                           bluesky_result: Optional[Dict[str, Any]],
                           geometric_result: Optional[Dict[str, Any]],
                           traffic_states: List[Dict[str, Any]],
                           timestamp: datetime) -> Optional[VerificationDiscrepancy]:
        """Analyze discrepancy between BlueSky and geometric results."""
        
        # Default values for missing results
        bs_conflict = bluesky_result['conflict'] if bluesky_result else False
        bs_cpa = bluesky_result['cpa_nm'] if bluesky_result else 0.0
        bs_tcpa = bluesky_result['tcpa_min'] if bluesky_result else 0.0
        
        geom_conflict = geometric_result['conflict'] if geometric_result else False
        geom_cpa = geometric_result['cpa_nm'] if geometric_result else 0.0
        geom_tcpa = geometric_result['tcpa_min'] if geometric_result else 0.0
        
        # Check for significant discrepancies
        conflict_agreement = bs_conflict == geom_conflict
        cpa_diff = abs(bs_cpa - geom_cpa)
        tcpa_diff = abs(bs_tcpa - geom_tcpa)
        
        # Determine if this is a significant discrepancy
        significant_discrepancy = (
            not conflict_agreement or
            cpa_diff > 1.0 or  # > 1 NM difference
            tcpa_diff > 0.5    # > 30 seconds difference
        )
        
        if not significant_discrepancy:
            return None
        
        # Determine severity
        if not conflict_agreement:
            severity = "high"
        elif cpa_diff > 2.0 or tcpa_diff > 1.0:
            severity = "medium"
        else:
            severity = "low"
        
        # Get aircraft states
        aircraft_lookup = {ac['id']: ac for ac in traffic_states}
        ac1_state = aircraft_lookup.get(pair[0], {})
        ac2_state = aircraft_lookup.get(pair[1], {})
        
        return VerificationDiscrepancy(
            timestamp=timestamp,
            aircraft_pair=pair,
            bluesky_conflict=bs_conflict,
            bluesky_cpa_nm=bs_cpa,
            bluesky_tcpa_min=bs_tcpa,
            geometric_conflict=geom_conflict,
            geometric_cpa_nm=geom_cpa,
            geometric_tcpa_min=geom_tcpa,
            conflict_agreement=conflict_agreement,
            cpa_difference_nm=cpa_diff,
            tcpa_difference_min=tcpa_diff,
            severity=severity,
            aircraft1_state=ac1_state,
            aircraft2_state=ac2_state
        )
    
    def _log_discrepancy(self, discrepancy: VerificationDiscrepancy):
        """Log verification discrepancy."""
        pair_str = f"{discrepancy.aircraft_pair[0]}-{discrepancy.aircraft_pair[1]}"
        
        logger.warning(
            f"VERIFICATION DISCREPANCY ({discrepancy.severity.upper()}): {pair_str} "
            f"- BlueSky conflict: {discrepancy.bluesky_conflict}, "
            f"Geometric conflict: {discrepancy.geometric_conflict}, "
            f"CPA diff: {discrepancy.cpa_difference_nm:.2f} NM, "
            f"TCPA diff: {discrepancy.tcpa_difference_min:.2f} min"
        )
    
    def _create_verified_conflict(self, pair: Tuple[str, str],
                                 bluesky_result: Optional[Dict[str, Any]],
                                 geometric_result: Optional[Dict[str, Any]],
                                 timestamp: datetime) -> Optional[ConflictPrediction]:
        """Create verified conflict prediction."""
        
        # Use geometric result as primary, BlueSky as validation
        if geometric_result and geometric_result['conflict']:
            return ConflictPrediction(
                ownship_id=pair[0],
                intruder_id=pair[1],
                time_to_cpa_min=geometric_result['tcpa_min'],
                distance_at_cpa_nm=geometric_result['cpa_nm'],
                altitude_diff_ft=geometric_result.get('alt_diff_ft', 0.0),
                is_conflict=True,
                severity_score=self._calculate_severity(geometric_result),
                conflict_type="horizontal",
                prediction_time=timestamp,
                confidence=1.0 if bluesky_result and bluesky_result['conflict'] else 0.8
            )
        
        return None
    
    def _calculate_severity(self, conflict_data: Dict[str, Any]) -> float:
        """Calculate conflict severity score."""
        cpa_nm = conflict_data['cpa_nm']
        tcpa_min = conflict_data['tcpa_min']
        
        # Normalize to 0-1 scale
        cpa_score = max(0.0, 1.0 - cpa_nm / self.min_horizontal_sep_nm)
        time_score = max(0.0, 1.0 - tcpa_min / self.lookahead_time_min)
        
        return min(1.0, (cpa_score + time_score) / 2.0)
    
    def _log_verification_results(self, bluesky_count: int, geometric_count: int,
                                 verified_count: int, discrepancy_count: int,
                                 timestamp: datetime):
        """Log verification results summary."""
        
        verification_record = {
            'timestamp': timestamp.isoformat(),
            'bluesky_conflicts': bluesky_count,
            'geometric_conflicts': geometric_count,
            'verified_conflicts': verified_count,
            'discrepancies': discrepancy_count,
            'agreement_rate': 1.0 - (discrepancy_count / max(1, verified_count))
        }
        
        self.verification_history.append(verification_record)
        
        logger.info(
            f"DUAL VERIFICATION: BlueSky: {bluesky_count}, "
            f"Geometric: {geometric_count}, Verified: {verified_count}, "
            f"Discrepancies: {discrepancy_count}"
        )
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """Get summary of verification performance."""
        if not self.verification_history:
            return {"error": "No verification data available"}
        
        total_discrepancies = len(self.discrepancies)
        total_verifications = len(self.verification_history)
        
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        for disc in self.discrepancies:
            severity_counts[disc.severity] += 1
        
        return {
            'total_verification_cycles': total_verifications,
            'total_discrepancies': total_discrepancies,
            'discrepancy_rate': total_discrepancies / max(1, total_verifications),
            'severity_breakdown': severity_counts,
            'recent_verification': self.verification_history[-1] if self.verification_history else None
        }
