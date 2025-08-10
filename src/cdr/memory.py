"""
LLM Memory (Experience Library) System for ATC LLM

This module implements an experience library that stores past conflict resolution
scenarios and retrieves similar experiences to enhance LLM decision-making.
The system maintains:
- Past movements and their outcomes
- Situation features (relative bearing, closure rate, altitude difference, TCA)
- Vector-based similarity search for experience retrieval
- JSONL persistence for experience storage

Based on embodied ATC agent research for function-calling + experience library.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SituationFeatures:
    """Normalized features describing a conflict scenario."""
    relative_bearing_deg: float  # 0-360, relative bearing from ownship to intruder
    closure_rate_kt: float       # Positive = closing, negative = separating
    alt_diff_ft: float          # Intruder altitude - ownship altitude
    tca_min: float              # Time to closest approach in minutes
    distance_nm: float          # Current separation distance in NM
    
    def to_vector(self) -> np.ndarray:
        """Convert to normalized feature vector for similarity comparison."""
        # Normalize relative bearing to [-1, 1] using sine/cosine
        bearing_rad = np.radians(self.relative_bearing_deg)
        bearing_x = np.cos(bearing_rad)
        bearing_y = np.sin(bearing_rad)
        
        # Normalize other features to reasonable ranges
        closure_norm = np.tanh(self.closure_rate_kt / 500.0)  # Normalize around 500kt max
        alt_norm = np.tanh(self.alt_diff_ft / 10000.0)        # Normalize around 10,000ft
        tca_norm = np.tanh(self.tca_min / 20.0)               # Normalize around 20min
        dist_norm = np.tanh(self.distance_nm / 100.0)         # Normalize around 100NM
        
        return np.array([bearing_x, bearing_y, closure_norm, alt_norm, tca_norm, dist_norm])


@dataclass
class ExperienceRecord:
    """A single experience record in the memory system."""
    timestamp: str
    features: SituationFeatures
    prompt: str
    llm_output: Dict[str, Any]
    command: str
    cpa_result: Dict[str, Any]
    outcome: str  # "pass" or "fail"
    min_sep_nm: float
    min_sep_ft: float
    scenario_id: str


class LLMMemorySystem:
    """Experience library for LLM conflict resolution."""
    
    def __init__(self, memory_file: Path, max_records: int = 10000):
        """
        Initialize memory system.
        
        Args:
            memory_file: Path to JSONL file for persistent storage
            max_records: Maximum number of records to keep in memory
        """
        self.memory_file = Path(memory_file)
        self.max_records = max_records
        self.experiences: List[ExperienceRecord] = []
        self.feature_vectors: List[np.ndarray] = []
        
        # Create directory if it doesn't exist
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing experiences
        self._load_experiences()
        
        logger.info(f"Initialized LLM memory system with {len(self.experiences)} experiences")
    
    def _load_experiences(self):
        """Load experiences from persistent storage."""
        if not self.memory_file.exists():
            logger.info(f"Memory file {self.memory_file} does not exist, starting fresh")
            return
        
        try:
            with open(self.memory_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        experience = self._dict_to_experience(data)
                        self.experiences.append(experience)
                        self.feature_vectors.append(experience.features.to_vector())
                    except Exception as e:
                        logger.warning(f"Failed to parse line {line_num} in memory file: {e}")
            
            logger.info(f"Loaded {len(self.experiences)} experiences from {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Failed to load memory file {self.memory_file}: {e}")
    
    def _dict_to_experience(self, data: Dict[str, Any]) -> ExperienceRecord:
        """Convert dictionary to ExperienceRecord."""
        features_data = data['features']
        features = SituationFeatures(**features_data)
        
        return ExperienceRecord(
            timestamp=data['timestamp'],
            features=features,
            prompt=data['prompt'],
            llm_output=data['llm_output'],
            command=data['command'],
            cpa_result=data['cpa_result'],
            outcome=data['outcome'],
            min_sep_nm=data['min_sep_nm'],
            min_sep_ft=data['min_sep_ft'],
            scenario_id=data['scenario_id']
        )
    
    def _experience_to_dict(self, experience: ExperienceRecord) -> Dict[str, Any]:
        """Convert ExperienceRecord to dictionary for serialization."""
        return {
            'timestamp': experience.timestamp,
            'features': asdict(experience.features),
            'prompt': experience.prompt,
            'llm_output': experience.llm_output,
            'command': experience.command,
            'cpa_result': experience.cpa_result,
            'outcome': experience.outcome,
            'min_sep_nm': experience.min_sep_nm,
            'min_sep_ft': experience.min_sep_ft,
            'scenario_id': experience.scenario_id
        }
    
    def add_experience(self,
                      ownship_state: Dict[str, Any],
                      intruder_state: Dict[str, Any],
                      prompt: str,
                      llm_output: Dict[str, Any],
                      command: str,
                      cpa_result: Dict[str, Any],
                      outcome: str,
                      scenario_id: Optional[str] = None) -> None:
        """
        Add new experience to memory.
        
        Args:
            ownship_state: Current ownship aircraft state
            intruder_state: Conflicting intruder state
            prompt: LLM prompt that was used
            llm_output: LLM's response
            command: BlueSky command that was executed
            cpa_result: CPA computation result
            outcome: "pass" or "fail" based on verification
            scenario_id: Optional unique scenario identifier
        """
        # Extract situation features
        features = self._extract_features(ownship_state, intruder_state, cpa_result)
        
        # Generate scenario ID if not provided
        if scenario_id is None:
            scenario_id = self._generate_scenario_id(features, datetime.now().isoformat())
        
        # Create experience record
        experience = ExperienceRecord(
            timestamp=datetime.now().isoformat(),
            features=features,
            prompt=prompt,
            llm_output=llm_output,
            command=command,
            cpa_result=cpa_result,
            outcome=outcome,
            min_sep_nm=cpa_result.get('min_sep_nm', 0.0),
            min_sep_ft=cpa_result.get('min_sep_ft', 0.0),
            scenario_id=scenario_id
        )
        
        # Add to memory
        self.experiences.append(experience)
        self.feature_vectors.append(features.to_vector())
        
        # Maintain max size
        if len(self.experiences) > self.max_records:
            self.experiences.pop(0)
            self.feature_vectors.pop(0)
        
        # Persist to file
        self._append_to_file(experience)
        
        logger.debug(f"Added experience {scenario_id} with outcome {outcome}")
    
    def _extract_features(self,
                         ownship_state: Dict[str, Any],
                         intruder_state: Dict[str, Any],
                         cpa_result: Dict[str, Any]) -> SituationFeatures:
        """Extract normalized features from aircraft states."""
        # Get positions and velocities
        own_lat = ownship_state.get('lat', ownship_state.get('latitude', 0.0))
        own_lon = ownship_state.get('lon', ownship_state.get('longitude', 0.0))
        own_alt = ownship_state.get('alt_ft', ownship_state.get('altitude_ft', 0.0))
        own_hdg = ownship_state.get('hdg', ownship_state.get('heading', 0.0))
        own_spd = ownship_state.get('spd_kt', ownship_state.get('ground_speed_kt', 0.0))
        
        int_lat = intruder_state.get('lat', intruder_state.get('latitude', 0.0))
        int_lon = intruder_state.get('lon', intruder_state.get('longitude', 0.0))
        int_alt = intruder_state.get('alt_ft', intruder_state.get('altitude_ft', 0.0))
        int_hdg = intruder_state.get('hdg', intruder_state.get('heading', 0.0))
        int_spd = intruder_state.get('spd_kt', intruder_state.get('ground_speed_kt', 0.0))
        
        # Calculate relative bearing
        # Vector from ownship to intruder
        dlat = int_lat - own_lat
        dlon = int_lon - own_lon
        bearing_to_intruder = np.degrees(np.arctan2(dlon, dlat)) % 360
        
        # Relative bearing (from ownship heading)
        relative_bearing = (bearing_to_intruder - own_hdg) % 360
        
        # Estimate closure rate (simplified)
        # This is a rough approximation; proper calculation requires vector math
        distance_nm = np.sqrt((dlat * 60)**2 + (dlon * 60 * np.cos(np.radians(own_lat)))**2)
        
        # Use relative velocities for closure rate approximation
        closure_rate = (own_spd - int_spd) * np.cos(np.radians(relative_bearing))
        
        return SituationFeatures(
            relative_bearing_deg=relative_bearing,
            closure_rate_kt=closure_rate,
            alt_diff_ft=int_alt - own_alt,
            tca_min=cpa_result.get('tca_min', 999.0),
            distance_nm=distance_nm
        )
    
    def _generate_scenario_id(self, features: SituationFeatures, timestamp: str) -> str:
        """Generate unique scenario ID based on features and timestamp."""
        # Create hash from features and timestamp
        data_str = f"{features.relative_bearing_deg:.1f}_{features.closure_rate_kt:.1f}_{features.alt_diff_ft:.0f}_{timestamp}"
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def _append_to_file(self, experience: ExperienceRecord):
        """Append experience to persistent storage."""
        try:
            with open(self.memory_file, 'a') as f:
                json.dump(self._experience_to_dict(experience), f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to append experience to file: {e}")
    
    def retrieve_similar_experiences(self,
                                   ownship_state: Dict[str, Any],
                                   intruder_state: Dict[str, Any],
                                   cpa_result: Dict[str, Any],
                                   top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-K most similar past experiences.
        
        Args:
            ownship_state: Current ownship state
            intruder_state: Current intruder state
            cpa_result: Current CPA computation
            top_k: Number of similar experiences to retrieve
            
        Returns:
            List of similar experience dictionaries for prompt inclusion
        """
        if not self.experiences:
            return []
        
        # Extract features for current situation
        current_features = self._extract_features(ownship_state, intruder_state, cpa_result)
        current_vector = current_features.to_vector()
        
        # Compute cosine similarities
        similarities = []
        for i, stored_vector in enumerate(self.feature_vectors):
            # Cosine similarity
            dot_product = np.dot(current_vector, stored_vector)
            norm_product = np.linalg.norm(current_vector) * np.linalg.norm(stored_vector)
            similarity = dot_product / norm_product if norm_product > 0 else 0.0
            similarities.append((similarity, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        # Return top-K experiences formatted for prompt inclusion
        similar_experiences = []
        for similarity, idx in similarities[:top_k]:
            if similarity > 0.1:  # Minimum similarity threshold
                exp = self.experiences[idx]
                similar_experiences.append({
                    'situation': asdict(exp.features),
                    'action': exp.llm_output,
                    'outcome': {
                        'min_sep_nm': exp.min_sep_nm,
                        'min_sep_ft': exp.min_sep_ft,
                        'result': exp.outcome
                    },
                    'similarity': similarity
                })
        
        logger.debug(f"Retrieved {len(similar_experiences)} similar experiences")
        return similar_experiences
    
    def get_recent_movements(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent movements for prompt context.
        
        Args:
            limit: Maximum number of recent movements to return
            
        Returns:
            List of recent movement dictionaries
        """
        recent = []
        for exp in self.experiences[-limit:]:
            recent.append({
                'timestamp': exp.timestamp,
                'type': self._extract_movement_type(exp.command),
                'details': exp.llm_output,
                'result': exp.outcome
            })
        
        return recent
    
    def _extract_movement_type(self, command: str) -> str:
        """Extract movement type from BlueSky command."""
        command = command.upper()
        if 'DIRECT' in command or 'DIRTO' in command:
            return 'waypoint'
        elif 'HDG' in command:
            return 'heading'
        elif 'ALT' in command or 'VS' in command:
            return 'altitude'
        else:
            return 'unknown'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self.experiences:
            return {'total_experiences': 0}
        
        outcomes = [exp.outcome for exp in self.experiences]
        success_rate = outcomes.count('pass') / len(outcomes) if outcomes else 0.0
        
        return {
            'total_experiences': len(self.experiences),
            'success_rate': success_rate,
            'recent_experiences': len([exp for exp in self.experiences if 
                                     (datetime.now() - datetime.fromisoformat(exp.timestamp)).days < 7]),
            'memory_file': str(self.memory_file),
            'max_records': self.max_records
        }