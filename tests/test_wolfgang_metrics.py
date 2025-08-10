"""Comprehensive test suite for Wolfgang (2011) metrics calculator.

Tests all Wolfgang KPIs with synthetic data frames:
- TBAS: Time-Based Alerting Score
- LAT: Loss of Alerting Time
- DAT: Delay in Alert Time
- DFA: Delay in First Alert
- RE: Resolution Efficiency
- RI: Resolution Intrusiveness
- RAT: Resolution Alert Time

Coverage target: ≥75%
"""

import pytest
import pandas as pd
import numpy as np
import math
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
import warnings

from src.cdr.wolfgang_metrics import (
    WolfgangMetricsCalculator,
    ConflictData,
    TrackData,
    great_circle_distance_nm,
    clamp,
    EARTH_RADIUS_NM,
    DEFAULT_SEP_THRESHOLD_NM,
    DEFAULT_ALT_THRESHOLD_FT,
    DEFAULT_MARGIN_MIN,
    DEFAULT_SEP_TARGET_NM
)


class TestGreatCircleDistance:
    """Test great circle distance calculations."""
    
    def test_same_point(self):
        """Test distance between same point is zero."""
        lat, lon = 59.3, 18.1
        distance = great_circle_distance_nm(lat, lon, lat, lon)
        assert abs(distance) < 1e-6
    
    def test_known_distance(self):
        """Test distance between known points."""
        # Stockholm to Gothenburg (approximate)
        stockholm_lat, stockholm_lon = 59.3293, 18.0686
        gothenburg_lat, gothenburg_lon = 57.7089, 11.9746
        
        distance = great_circle_distance_nm(
            stockholm_lat, stockholm_lon, 
            gothenburg_lat, gothenburg_lon
        )
        
        # Should be approximately 240 nautical miles
        assert 200 < distance < 280
    
    def test_antipodal_points(self):
        """Test distance between antipodal points."""
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, 180.0
        
        distance = great_circle_distance_nm(lat1, lon1, lat2, lon2)
        
        # Should be approximately half Earth circumference
        expected = math.pi * EARTH_RADIUS_NM
        assert abs(distance - expected) < 100  # Allow some tolerance
    
    def test_equatorial_distance(self):
        """Test distance along equator."""
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, 1.0  # 1 degree longitude difference
        
        distance = great_circle_distance_nm(lat1, lon1, lat2, lon2)
        
        # 1 degree longitude at equator ≈ 60 nautical miles
        assert 55 < distance < 65


class TestClampFunction:
    """Test clamp utility function."""
    
    def test_clamp_within_range(self):
        """Test value within range is unchanged."""
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0
    
    def test_clamp_below_minimum(self):
        """Test value below minimum is clamped."""
        assert clamp(-5.0, 0.0, 10.0) == 0.0
        assert clamp(-1.0, 0.0, 10.0) == 0.0
    
    def test_clamp_above_maximum(self):
        """Test value above maximum is clamped."""
        assert clamp(15.0, 0.0, 10.0) == 10.0
        assert clamp(100.0, 0.0, 10.0) == 10.0


class TestConflictData:
    """Test ConflictData structure."""
    
    def test_conflict_data_creation(self):
        """Test ConflictData can be created with defaults."""
        conflict = ConflictData(scenario_id="SCEN001", conflict_id="CONF001")
        
        assert conflict.scenario_id == "SCEN001"
        assert conflict.conflict_id == "CONF001"
        assert conflict.t_first_alert_min is None
        assert conflict.baseline_separations == []
        assert conflict.resolved_separations == []
        assert conflict.TBAS is None
    
    def test_conflict_data_with_values(self):
        """Test ConflictData with populated values."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=5.0,
            t_los_min=8.0
        )
        
        assert conflict.t_first_alert_min == 5.0
        assert conflict.t_los_min == 8.0
        assert conflict.TBAS is None  # Computed later


class TestTrackData:
    """Test TrackData structure and calculations."""
    
    def test_track_data_creation(self):
        """Test TrackData creation."""
        track = TrackData(scenario_id="SCEN001")
        
        assert track.scenario_id == "SCEN001"
        assert track.track_points == []
        assert track.get_track_length_nm() == 0.0
    
    def test_track_length_single_point(self):
        """Test track length with single point."""
        track = TrackData(scenario_id="SCEN001")
        track.track_points = [(0.0, 59.3, 18.1)]
        
        assert track.get_track_length_nm() == 0.0
    
    def test_track_length_multiple_points(self):
        """Test track length with multiple points."""
        track = TrackData(scenario_id="SCEN001")
        track.track_points = [
            (0.0, 59.0, 18.0),
            (1.0, 59.1, 18.0),  # ~6 nm north
            (2.0, 59.1, 18.1),  # ~3 nm east
        ]
        
        length = track.get_track_length_nm()
        assert length > 8  # Should be > 6 + 3 nm
        assert length < 12  # But not too much more


class TestWolfgangCalculatorInit:
    """Test WolfgangMetricsCalculator initialization."""
    
    def test_default_initialization(self):
        """Test calculator with default parameters."""
        calc = WolfgangMetricsCalculator()
        
        assert calc.sep_threshold_nm == DEFAULT_SEP_THRESHOLD_NM
        assert calc.alt_threshold_ft == DEFAULT_ALT_THRESHOLD_FT
        assert calc.margin_min == DEFAULT_MARGIN_MIN
        assert calc.sep_target_nm == DEFAULT_SEP_TARGET_NM
        assert calc.conflicts == {}
        assert calc.planned_tracks == {}
        assert calc.resolved_tracks == {}
    
    def test_custom_initialization(self):
        """Test calculator with custom parameters."""
        calc = WolfgangMetricsCalculator(
            sep_threshold_nm=3.0,
            alt_threshold_ft=500.0,
            margin_min=3.0,
            sep_target_nm=4.0
        )
        
        assert calc.sep_threshold_nm == 3.0
        assert calc.alt_threshold_ft == 500.0
        assert calc.margin_min == 3.0
        assert calc.sep_target_nm == 4.0


class TestCSVLoading:
    """Test CSV file loading functionality."""
    
    def create_temp_events_csv(self, data: List[Dict[str, Any]]) -> Path:
        """Create temporary events CSV file."""
        df = pd.DataFrame(data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.close()  # Close the file so it can be written to
        csv_path = Path(temp_file.name)
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def test_load_events_csv_basic(self):
        """Test loading basic events CSV."""
        events_data = [
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 'event': 'alert', 't_min': 5.0},
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 'event': 'los', 't_min': 8.0},
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 'event': 'resolution_cmd', 't_min': 6.0},
        ]
        
        calc = WolfgangMetricsCalculator()
        csv_path = self.create_temp_events_csv(events_data)
        
        try:
            calc.load_events_csv(csv_path)
            
            # Check conflicts were loaded
            assert len(calc.conflicts) == 1
            key = ('SCEN001', 'CONF001')
            assert key in calc.conflicts
            
            conflict = calc.conflicts[key]
            assert conflict.t_first_alert_min == 5.0
            assert conflict.t_los_min == 8.0
            assert conflict.t_resolution_action_min == 6.0
            
        finally:
            csv_path.unlink(missing_ok=True)  # Clean up
    
    def test_load_events_csv_multiple_alerts(self):
        """Test loading events with multiple alerts."""
        events_data = [
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 'event': 'alert', 't_min': 5.0},
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 'event': 'alert_update', 't_min': 6.0},
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 'event': 'alert_update', 't_min': 7.0},
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 'event': 'los', 't_min': 8.0},
        ]
        
        calc = WolfgangMetricsCalculator()
        csv_path = self.create_temp_events_csv(events_data)
        
        try:
            calc.load_events_csv(csv_path)
            
            conflict = calc.conflicts[('SCEN001', 'CONF001')]
            assert conflict.t_first_alert_min == 5.0  # First alert
            assert conflict.t_last_alert_min == 7.0   # Last alert update
            
        finally:
            csv_path.unlink(missing_ok=True)
    
    def test_load_events_csv_missing_file(self):
        """Test loading non-existent events CSV."""
        calc = WolfgangMetricsCalculator()
        
        # Should not raise exception, just log warning
        calc.load_events_csv(Path("nonexistent.csv"))
        assert len(calc.conflicts) == 0
    
    def test_load_separation_csv_basic(self):
        """Test loading separation CSV."""
        sep_data = [
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 't_min': 0.0, 'sep_nm': 10.0},
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 't_min': 1.0, 'sep_nm': 8.0},
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 't_min': 2.0, 'sep_nm': 4.0},
            {'scenario_id': 'SCEN001', 'conflict_id': 'CONF001', 't_min': 3.0, 'sep_nm': 2.0},
        ]
        
        df = pd.DataFrame(sep_data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.close()  # Close the file so it can be written to
        csv_path = Path(temp_file.name)
        
        # Write CSV data
        df.to_csv(csv_path, index=False)
        
        calc = WolfgangMetricsCalculator()
        
        try:
            calc.load_baseline_separation_csv(csv_path)
            
            conflict = calc.conflicts[('SCEN001', 'CONF001')]
            assert len(conflict.baseline_separations) == 4
            
            # Check ordering and values
            separations = conflict.baseline_separations
            assert separations[0] == (0.0, 10.0)
            assert separations[1] == (1.0, 8.0)
            assert separations[2] == (2.0, 4.0)
            assert separations[3] == (3.0, 2.0)
            
        finally:
            csv_path.unlink(missing_ok=True)
    
    def test_load_track_csv_basic(self):
        """Test loading track CSV."""
        track_data = [
            {'scenario_id': 'SCEN001', 't_min': 0.0, 'lat': 59.0, 'lon': 18.0},
            {'scenario_id': 'SCEN001', 't_min': 1.0, 'lat': 59.1, 'lon': 18.0},
            {'scenario_id': 'SCEN001', 't_min': 2.0, 'lat': 59.2, 'lon': 18.0},
        ]
        
        df = pd.DataFrame(track_data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.close()  # Close the file so it can be written to
        csv_path = Path(temp_file.name)
        
        # Write CSV data
        df.to_csv(csv_path, index=False)
        
        calc = WolfgangMetricsCalculator()
        
        try:
            calc.load_planned_track_csv(csv_path)
            
            assert 'SCEN001' in calc.planned_tracks
            track = calc.planned_tracks['SCEN001']
            assert len(track.track_points) == 3
            assert track.track_points[0] == (0.0, 59.0, 18.0)
            assert track.track_points[2] == (2.0, 59.2, 18.0)
            
        finally:
            csv_path.unlink(missing_ok=True)


class TestWolfgangKPIsCalculation:
    """Test Wolfgang KPI calculation methods."""
    
    def setup_method(self):
        """Set up calculator for each test."""
        self.calc = WolfgangMetricsCalculator()
    
    def test_tbas_calculation_successful(self):
        """Test TBAS calculation with successful alerting."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=5.0,
            t_los_min=12.0  # Alert is 7 minutes before LoS
        )
        
        # With default margin of 5 minutes, alert should be successful
        # Alert threshold = 12.0 - 5.0 = 7.0
        # First alert at 5.0 <= 7.0, so TBAS = 1.0
        self.calc._compute_tbas(conflict)
        assert conflict.TBAS == 1.0
    
    def test_tbas_calculation_failed(self):
        """Test TBAS calculation with failed alerting."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=8.0,
            t_los_min=12.0  # Alert is only 4 minutes before LoS
        )
        
        # With default margin of 5 minutes, alert should be failed
        # Alert threshold = 12.0 - 5.0 = 7.0
        # First alert at 8.0 > 7.0, so TBAS = 0.0
        self.calc._compute_tbas(conflict)
        assert conflict.TBAS == 0.0
    
    def test_tbas_missing_data(self):
        """Test TBAS calculation with missing data."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001"
            # Missing t_first_alert_min and t_los_min
        )
        
        self.calc._compute_tbas(conflict)
        assert conflict.TBAS is None
    
    def test_lat_calculation_basic(self):
        """Test LAT (Loss of Alert Time) calculation."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_last_alert_min=10.0,
            t_los_min=12.0
        )
        
        self.calc._compute_lat(conflict)
        assert conflict.LAT_min == 2.0  # 12.0 - 10.0
    
    def test_lat_negative_value(self):
        """Test LAT with last alert after LoS."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_last_alert_min=15.0,
            t_los_min=12.0
        )
        
        self.calc._compute_lat(conflict)
        assert conflict.LAT_min == -3.0  # 12.0 - 15.0
    
    def test_dat_calculation_basic(self):
        """Test DAT (Detection Alert Time) calculation."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=6.0,
            t_los_min=12.0
        )
        
        self.calc._compute_dat(conflict)
        assert conflict.DAT_min == 6.0  # 12.0 - 6.0
    
    def test_dfa_calculation_successful(self):
        """Test DFA (Detection of First Alert) calculation."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=5.0,
            t_los_min=12.0
        )
        
        # Same logic as TBAS
        self.calc._compute_dfa(conflict)
        assert conflict.DFA == 1.0
    
    def test_dfa_calculation_failed(self):
        """Test DFA calculation with late first alert."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=8.0,
            t_los_min=12.0
        )
        
        self.calc._compute_dfa(conflict)
        assert conflict.DFA == 0.0
    
    def test_re_calculation_improvement(self):
        """Test RE (Resolution Efficiency) with improvement."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            baseline_separations=[(0.0, 8.0), (1.0, 4.0), (2.0, 2.0)],  # Min = 2.0
            resolved_separations=[(0.0, 8.0), (1.0, 6.0), (2.0, 4.0)]   # Min = 4.0
        )
        
        self.calc._compute_re(conflict)
        
        # Baseline min = 2.0, resolved min = 4.0, target = 5.0
        # Gain = 4.0 - 2.0 = 2.0
        # Max possible gain = 5.0 - 2.0 = 3.0
        # RE = 2.0 / 3.0 = 0.667
        assert conflict.RE is not None
        assert abs(conflict.RE - 2.0/3.0) < 1e-6
        assert conflict.min_sep_baseline_nm == 2.0
        assert conflict.min_sep_resolved_nm == 4.0
    
    def test_re_calculation_no_improvement(self):
        """Test RE calculation with no improvement."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            baseline_separations=[(0.0, 8.0), (1.0, 4.0)],  # Min = 4.0
            resolved_separations=[(0.0, 6.0), (1.0, 3.0)]   # Min = 3.0 (worse)
        )
        
        self.calc._compute_re(conflict)
        
        # Negative gain should result in RE = 0 (clamped)
        assert conflict.RE == 0.0
    
    def test_re_missing_data(self):
        """Test RE calculation with missing separation data."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001"
            # Missing separation data
        )
        
        self.calc._compute_re(conflict)
        assert conflict.RE is None
    
    def test_ri_calculation_basic(self):
        """Test RI (Resolution Intrusiveness) calculation."""
        # Set up tracks
        self.calc.planned_tracks['SCEN001'] = TrackData(scenario_id='SCEN001')
        self.calc.planned_tracks['SCEN001'].track_points = [
            (0.0, 59.0, 18.0),
            (1.0, 59.1, 18.0),  # ~6 nm north
        ]
        
        self.calc.resolved_tracks['SCEN001'] = TrackData(scenario_id='SCEN001')
        self.calc.resolved_tracks['SCEN001'].track_points = [
            (0.0, 59.0, 18.0),
            (1.0, 59.1, 18.1),  # Longer path due to deviation
            (2.0, 59.2, 18.1),
        ]
        
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001"
        )
        
        self.calc._compute_ri(conflict)
        
        # Should have positive RI since resolved track is longer
        assert conflict.RI is not None
        assert conflict.RI > 0.0
    
    def test_ri_missing_tracks(self):
        """Test RI calculation with missing track data."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001"
        )
        
        self.calc._compute_ri(conflict)
        assert conflict.RI is None
    
    def test_rat_calculation_basic(self):
        """Test RAT (Resolution Alert Time) calculation."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=5.0,
            t_resolution_action_min=7.0
        )
        
        self.calc._compute_rat(conflict)
        assert conflict.RAT_min == 2.0  # 7.0 - 5.0
    
    def test_rat_missing_data(self):
        """Test RAT calculation with missing data."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=5.0
            # Missing t_resolution_action_min
        )
        
        self.calc._compute_rat(conflict)
        assert conflict.RAT_min is None


class TestLosTimeComputation:
    """Test Loss of Separation time computation."""
    
    def setup_method(self):
        """Set up calculator for each test."""
        self.calc = WolfgangMetricsCalculator()
    
    def test_los_time_from_resolved_separation(self):
        """Test LoS time computed from resolved separation data."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            resolved_separations=[(0.0, 8.0), (1.0, 6.0), (2.0, 4.0), (3.0, 3.0)]
        )
        
        # With default threshold of 5.0 NM, LoS occurs at t=2.0 (first time < 5.0)
        los_time = self.calc.compute_los_time(conflict)
        assert los_time == 2.0
    
    def test_los_time_from_baseline_separation(self):
        """Test LoS time from baseline separation (no resolved data)."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            baseline_separations=[(0.0, 10.0), (1.0, 7.0), (2.0, 4.0), (3.0, 2.0)]
        )
        
        los_time = self.calc.compute_los_time(conflict)
        assert los_time == 2.0
    
    def test_los_time_from_event(self):
        """Test LoS time from explicit event."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_los_min=8.5
        )
        
        los_time = self.calc.compute_los_time(conflict)
        assert los_time == 8.5
    
    def test_los_time_no_violation(self):
        """Test LoS time when no separation violation occurs."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            resolved_separations=[(0.0, 10.0), (1.0, 8.0), (2.0, 6.0)]
        )
        
        los_time = self.calc.compute_los_time(conflict)
        assert los_time is None


class TestTemporalValidation:
    """Test temporal ordering validation."""
    
    def setup_method(self):
        """Set up calculator for each test."""
        self.calc = WolfgangMetricsCalculator()
    
    def test_temporal_validation_correct_order(self):
        """Test validation with correct temporal order."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=5.0,
            t_last_alert_min=7.0,
            t_los_min=10.0
        )
        
        # Should not raise any warnings for correct order
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.calc._validate_temporal_order(conflict)
            # No warnings should be generated
    
    def test_temporal_validation_inverted_order(self):
        """Test validation with inverted temporal order."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=10.0,
            t_last_alert_min=7.0,  # Invalid: last alert before first
            t_los_min=5.0         # Invalid: LoS before alerts
        )
        
        # Should log warnings but not raise exceptions
        self.calc._validate_temporal_order(conflict)
        # Test passes if no exception is raised


class TestMetricsIntegration:
    """Integration tests for complete metrics computation."""
    
    def setup_method(self):
        """Set up calculator for each test."""
        self.calc = WolfgangMetricsCalculator()
    
    def test_compute_metrics_complete_data(self):
        """Test complete metrics computation with full data."""
        # Add conflict with complete data
        self.calc.conflicts[('SCEN001', 'CONF001')] = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=5.0,
            t_last_alert_min=7.0,
            t_resolution_action_min=6.0,
            t_los_min=10.0,
            baseline_separations=[(0.0, 8.0), (5.0, 3.0), (10.0, 1.0)],
            resolved_separations=[(0.0, 8.0), (5.0, 4.0), (10.0, 3.0)]
        )
        
        # Add track data
        self.calc.planned_tracks['SCEN001'] = TrackData(scenario_id='SCEN001')
        self.calc.planned_tracks['SCEN001'].track_points = [
            (0.0, 59.0, 18.0), (10.0, 59.1, 18.0)
        ]
        
        self.calc.resolved_tracks['SCEN001'] = TrackData(scenario_id='SCEN001')
        self.calc.resolved_tracks['SCEN001'].track_points = [
            (0.0, 59.0, 18.0), (5.0, 59.05, 18.05), (10.0, 59.1, 18.0)
        ]
        
        # Compute all metrics
        self.calc.compute_metrics()
        
        conflict = self.calc.conflicts[('SCEN001', 'CONF001')]
        
        # Verify all metrics are computed
        assert conflict.TBAS is not None
        assert conflict.LAT_min is not None
        assert conflict.DAT_min is not None
        assert conflict.DFA is not None
        assert conflict.RE is not None
        assert conflict.RI is not None
        assert conflict.RAT_min is not None
    
    def test_compute_metrics_missing_data(self):
        """Test metrics computation with missing data."""
        # Add conflict with minimal data
        self.calc.conflicts[('SCEN001', 'CONF001')] = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=5.0
            # Missing most other data
        )
        
        # Compute metrics (should not crash)
        self.calc.compute_metrics()
        
        conflict = self.calc.conflicts[('SCEN001', 'CONF001')]
        
        # Most metrics should be None due to missing data
        assert conflict.TBAS is None
        assert conflict.LAT_min is None
        assert conflict.RE is None
        assert conflict.RI is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up calculator for each test."""
        self.calc = WolfgangMetricsCalculator()
    
    def test_no_events_null_tbas(self):
        """Test that no events results in null TBAS."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001"
            # No events
        )
        
        self.calc._compute_tbas(conflict)
        assert conflict.TBAS is None
    
    def test_inverted_time_order_computation(self):
        """Test computation proceeds despite inverted time order."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=10.0,  # After LoS
            t_los_min=5.0
        )
        
        # Should still compute metrics despite temporal violation
        self.calc._compute_tbas(conflict)
        self.calc._compute_dat(conflict)
        
        # TBAS should be 0 since alert is after LoS threshold
        assert conflict.TBAS == 0.0
        # DAT should be negative
        assert conflict.DAT_min == -5.0  # 5.0 - 10.0
    
    def test_absent_los_null_metrics(self):
        """Test that absent LoS makes metrics requiring LoS go null."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=5.0,
            t_last_alert_min=7.0
            # No t_los_min
        )
        
        self.calc._compute_tbas(conflict)
        self.calc._compute_lat(conflict)
        self.calc._compute_dat(conflict)
        self.calc._compute_dfa(conflict)
        
        # All LoS-dependent metrics should be None
        assert conflict.TBAS is None
        assert conflict.LAT_min is None
        assert conflict.DAT_min is None
        assert conflict.DFA is None
    
    def test_zero_track_length_ri(self):
        """Test RI calculation with zero planned track length."""
        self.calc.planned_tracks['SCEN001'] = TrackData(scenario_id='SCEN001')
        self.calc.planned_tracks['SCEN001'].track_points = [
            (0.0, 59.0, 18.0)  # Single point = zero length
        ]
        
        self.calc.resolved_tracks['SCEN001'] = TrackData(scenario_id='SCEN001')
        self.calc.resolved_tracks['SCEN001'].track_points = [
            (0.0, 59.0, 18.0), (1.0, 59.1, 18.0)
        ]
        
        conflict = ConflictData(scenario_id="SCEN001", conflict_id="CONF001")
        
        self.calc._compute_ri(conflict)
        assert conflict.RI is None
    
    def test_extreme_separation_values(self):
        """Test with extreme separation values."""
        conflict = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            baseline_separations=[(0.0, 0.1)],  # Very close
            resolved_separations=[(0.0, 100.0)]  # Very far
        )
        
        self.calc._compute_re(conflict)
        
        # Should be clamped to maximum RE = 1.0
        assert conflict.RE == 1.0


class TestCSVExport:
    """Test CSV export functionality."""
    
    def setup_method(self):
        """Set up calculator with sample data."""
        self.calc = WolfgangMetricsCalculator()
        
        # Add sample conflicts
        self.calc.conflicts[('SCEN001', 'CONF001')] = ConflictData(
            scenario_id="SCEN001",
            conflict_id="CONF001",
            t_first_alert_min=5.0,
            t_los_min=10.0,
            TBAS=1.0,
            DAT_min=5.0,
            RE=0.8
        )
        
        self.calc.conflicts[('SCEN002', 'CONF002')] = ConflictData(
            scenario_id="SCEN002",
            conflict_id="CONF002",
            t_first_alert_min=8.0,
            t_los_min=12.0,
            TBAS=0.0,
            DAT_min=4.0
        )
    
    def test_export_to_csv(self):
        """Test exporting metrics to CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            output_path = Path(temp_file.name)
        
        try:
            self.calc.export_to_csv(output_path)
            
            # Verify file exists and has content
            assert output_path.exists()
            
            # Read back and verify structure
            df = pd.read_csv(output_path)
            
            assert len(df) == 2
            assert 'scenario_id' in df.columns
            assert 'conflict_id' in df.columns
            assert 'TBAS' in df.columns
            assert 'DAT_min' in df.columns
            assert 'RE' in df.columns
            
            # Check specific values
            row1 = df[df['conflict_id'] == 'CONF001'].iloc[0]
            assert row1['TBAS'] == 1.0
            assert row1['DAT_min'] == 5.0
            assert row1['RE'] == 0.8
            
        finally:
            if output_path.exists():
                output_path.unlink()


class TestSummaryStatistics:
    """Test summary statistics functionality."""
    
    def setup_method(self):
        """Set up calculator with sample data."""
        self.calc = WolfgangMetricsCalculator()
        
        # Add conflicts with varying metrics
        for i in range(5):
            self.calc.conflicts[(f'SCEN{i:03d}', f'CONF{i:03d}')] = ConflictData(
                scenario_id=f"SCEN{i:03d}",
                conflict_id=f"CONF{i:03d}",
                TBAS=float(i % 2),  # Alternating 0 and 1
                DAT_min=float(5 + i),
                RE=0.1 * i if i > 0 else None  # First one is None
            )
    
    def test_print_summary(self, capsys):
        """Test summary printing."""
        self.calc.print_summary()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check that summary contains expected information
        assert "WOLFGANG (2011) METRICS SUMMARY" in output
        assert "Total conflicts processed: 5" in output
        assert "Non-null TBAS: 5" in output
        assert "Non-null DAT: 5" in output
        assert "Non-null RE: 4" in output  # One is None
        assert "Average TBAS:" in output
        assert "Average DAT:" in output


class TestFullIntegrationScenario:
    """Full integration test with realistic scenario."""
    
    def test_realistic_conflict_scenario(self):
        """Test complete workflow with realistic conflict scenario."""
        calc = WolfgangMetricsCalculator()
        
        # Create realistic events CSV data
        events_data = [
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 'event': 'alert', 't_min': 3.0},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 'event': 'alert_update', 't_min': 4.0},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 'event': 'resolution_cmd', 't_min': 4.5},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 'event': 'los', 't_min': 8.0},
            
            {'scenario_id': 'REAL001', 'conflict_id': 'AC003_AC004', 'event': 'alert', 't_min': 7.0},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC003_AC004', 'event': 'los', 't_min': 10.0},
        ]
        
        # Create realistic separation data
        baseline_sep_data = [
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 't_min': 0.0, 'sep_nm': 12.0},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 't_min': 2.0, 'sep_nm': 8.0},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 't_min': 4.0, 'sep_nm': 5.0},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 't_min': 6.0, 'sep_nm': 3.0},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 't_min': 8.0, 'sep_nm': 1.5},
        ]
        
        resolved_sep_data = [
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 't_min': 0.0, 'sep_nm': 12.0},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 't_min': 2.0, 'sep_nm': 8.0},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 't_min': 4.0, 'sep_nm': 5.5},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 't_min': 6.0, 'sep_nm': 6.0},
            {'scenario_id': 'REAL001', 'conflict_id': 'AC001_AC002', 't_min': 8.0, 'sep_nm': 7.0},
        ]
        
        # Create temporary CSV files
        temp_files = []
        
        try:
            # Events CSV
            events_df = pd.DataFrame(events_data)
            events_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            events_file.close()
            events_path = Path(events_file.name)
            events_df.to_csv(events_path, index=False)
            temp_files.append(events_path)
            
            # Baseline separation CSV
            baseline_df = pd.DataFrame(baseline_sep_data)
            baseline_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            baseline_file.close()
            baseline_path = Path(baseline_file.name)
            baseline_df.to_csv(baseline_path, index=False)
            temp_files.append(baseline_path)
            
            # Resolved separation CSV
            resolved_df = pd.DataFrame(resolved_sep_data)
            resolved_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            resolved_file.close()
            resolved_path = Path(resolved_file.name)
            resolved_df.to_csv(resolved_path, index=False)
            temp_files.append(resolved_path)
            
            # Load data
            calc.load_events_csv(temp_files[0])
            calc.load_baseline_separation_csv(temp_files[1])
            calc.load_resolved_separation_csv(temp_files[2])
            
            # Compute metrics
            calc.compute_metrics()
            
            # Verify results
            assert len(calc.conflicts) == 2
            
            # Check first conflict (with resolution)
            conflict1 = calc.conflicts[('REAL001', 'AC001_AC002')]
            assert conflict1.TBAS == 1.0  # Alert at 3.0, LoS at 8.0, threshold = 3.0
            assert conflict1.LAT_min == 4.0  # LoS(8.0) - last_alert(4.0)
            assert conflict1.DAT_min == 5.0  # LoS(8.0) - first_alert(3.0)
            assert conflict1.DFA == 1.0  # Same as TBAS logic
            assert conflict1.RE is not None  # Should have improvement
            assert conflict1.RAT_min == 1.5  # resolution(4.5) - first_alert(3.0)
            
            # Check second conflict (no resolution)
            conflict2 = calc.conflicts[('REAL001', 'AC003_AC004')]
            assert conflict2.TBAS == 0.0  # Alert at 7.0, LoS at 10.0, threshold = 5.0
            assert conflict2.RAT_min is None  # No resolution action
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
