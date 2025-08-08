# LLM-BlueSky Conflict Detection & Resolution System

An advanced LLM-driven conflict detection and resolution system built on top of BlueSky air traffic simulator using SCAT trajectories, implementing safety-first aviation standards with comprehensive testing and metrics collection.

⚠️ **IMPLEMENTATION STATUS**: Architecture complete, but requires BlueSky TCP integration and LLM deployment for end-to-end execution.

## Project Overview

This system implements a sophisticated real-time conflict detection and resolution pipeline that:
- **Polls every 5 simulation minutes** for real-time traffic monitoring
- **Predicts conflicts 10 minutes ahead** using deterministic algorithms
- **Issues horizontal or vertical resolutions** for ownship only (ATC standard)
- **Uses Llama 3.1 8B** for intelligent detection and resolution reasoning
- **Benchmarks against BlueSky baseline** with Wolfgang (2011) aviation KPIs
- **Enforces safety validation** before any command execution
- **Implements oscillation guards** to prevent command thrashing
- **Provides comprehensive metrics** for performance evaluation

## Implementation Status & Critical Gaps

### ✅ **Completed Architecture**
- **Safety-first design patterns** with validation pipelines
- **Comprehensive type system** with Pydantic schemas
- **Modular component architecture** with clean interfaces
- **Geodesy calculations** fully implemented and tested
- **Conflict detection algorithms** with deterministic CPA calculations
- **Resolution validation framework** with oscillation guards
- **Metrics collection framework** (metrics definitions need correction)

### ❌ **Critical Implementation Gaps (Blocking End-to-End Execution)**

#### 1. **BlueSky Integration Not Implemented**
- **`bluesky_io.py:get_aircraft_states()`** - Returns empty list (TODO)
- **`pipeline.py:_fetch_aircraft_states()`** - No actual BlueSky TCP communication
- **Missing**: Aircraft creation, simulation stepping, real-time state fetching
- **Impact**: Cannot run with live simulator data

#### 2. **LLM Integration Not Connected**
- **`llm_client.py:_call_llm()`** - Uses hardcoded mock responses
- **Missing**: Actual Ollama subprocess calls and JSON parsing
- **Missing**: Local Llama 3.1 8B model deployment
- **Impact**: No real LLM reasoning, only simulated responses

#### 3. **Wolfgang (2011) Metrics Incorrectly Defined**
- **Current definitions are incorrect** - need correction per Wolfgang (2011)
- **Corrected definitions**:
  - **TBAS**: Time Between Analysis and Scenario (not "Time-Based Alerting Score")
  - **LAT**: Look-Ahead Time (not "Loss of Alerting Time")
  - **PA**: Prediction Accuracy (not "Predicted Alerts")
  - **PI**: Prediction Integrity (not "Predicted Intrusions") 
  - **DAT**: Detection Analysis Time (not "Delay in Alert Time")
  - **DFA**: Detection False Alert (not "Delay in First Alert")
  - **RE/RI/RAT**: Resolution Efficiency / Resolution Integrity / Resolution Analysis Time
- **Impact**: Incorrect performance benchmarking

#### 4. **Test Coverage Unverified**
- **Claims of 95% coverage** not validated without CI execution
- **Test suites exist** but execution status unknown
- **Impact**: Code quality and reliability unverified

### 🔧 **Required for End-to-End Execution**

1. **Implement BlueSky TCP Interface**
   ```python
   # In bluesky_io.py
   def get_aircraft_states(self) -> List[AircraftState]:
       # Send "STATE" command to BlueSky TCP socket
       # Parse response and convert to AircraftState objects
   ```

2. **Deploy Local LLM with Ollama**
   ```python
   # In llm_client.py  
   def _call_llm(self, prompt: str) -> Optional[str]:
       # subprocess.run(["ollama", "run", "llama3.1:8b", prompt])
       # Parse JSON response with error handling
   ```

3. **Correct Wolfgang Metrics Implementation**
   ```python
   # In metrics.py - fix metric definitions and calculations
   ```

4. **Validate Test Coverage with CI**
   ```bash
   pytest --cov=src --cov-report=html
   ```

## Quick Start Guide

### Prerequisites
- Python 3.11+
- ❌ BlueSky simulator (TCP interface not implemented)
- ❌ Llama 3.1 8B model with Ollama (not connected)

### Current Installation (Architecture Only)

```bash
# Clone repository
git clone <repository-url>
cd ATC_LLM

# Install dependencies
pip install -r requirements.txt

# Run tests (coverage unverified)
pytest

# Code quality checks (execution status unknown)
black . && ruff . && mypy .
```

### ❌ **Cannot Run End-to-End** (Missing Implementation)

```bash
# These commands will fail due to missing BlueSky/LLM integration:
# python -m src.cdr.pipeline --scenario scenarios/sample_scat_ownship.json
# python -m src.cdr.pipeline --bluesky --scenario scenarios/blue_sky_init.txt
```

## System Architecture

```
src/
├── cdr/                     # Core Conflict Detection & Resolution Package
│   ├── __init__.py          # Package initialization with exported geodesy functions
│   ├── geodesy.py           # Aviation geodesy calculations
│   ├── detect.py            # Conflict detection algorithms
│   ├── resolve.py           # Resolution generation and validation
│   ├── llm_client.py        # LLM integration with safety wrappers
│   ├── schemas.py           # Pydantic data models and validation
│   ├── pipeline.py          # Main execution pipeline
│   ├── bluesky_io.py        # BlueSky simulator interface
│   └── metrics.py           # Wolfgang (2011) KPI calculations
└── api/
    ├── __init__.py          # API package initialization
    └── service.py           # FastAPI REST service for monitoring
```

## Detailed File Documentation

### Core CDR Package (`src/cdr/`)

#### 📍 `__init__.py` - Package Initialization
**Purpose**: Defines package metadata and exports core geodesy functions
**Exports**: `haversine_nm`, `bearing_rad`, `cpa_nm`
**Version**: 0.1.0
**Author**: Somnath (somnathab3@gmail.com)

#### 🌍 `geodesy.py` - Aviation Geodesy Calculations
**Purpose**: Provides mathematical functions for aviation navigation and conflict geometry
**Key Constants**:
- `R_NM = 3440.065` - Earth radius in nautical miles

**Functions**:
- **`haversine_nm(a: Coordinate, b: Coordinate) -> float`**
  - Calculates great circle distance using haversine formula
  - Input: Two coordinate tuples (lat, lon) in degrees
  - Output: Distance in nautical miles
  - Used for: Initial traffic filtering, separation validation
  
- **`bearing_rad(a: Coordinate, b: Coordinate) -> float`**
  - Calculates initial bearing from point A to point B
  - Input: Two coordinate tuples (lat, lon) in degrees
  - Output: Bearing in radians (0 = North, π/2 = East)
  - Used for: Trajectory calculations, resolution planning
  
- **`cpa_nm(own: Aircraft, intr: Aircraft) -> Tuple[float, float]`**
  - Predicts Closest Point of Approach for constant-velocity aircraft
  - Input: Aircraft dictionaries with lat, lon, spd_kt, hdg_deg
  - Output: (minimum_distance_nm, time_to_cpa_minutes)
  - Uses: Flat Earth approximation suitable for 10-minute predictions
  - Core algorithm for conflict detection
  
- **`cross_track_distance_nm(point: Coordinate, track_start: Coordinate, track_end: Coordinate) -> float`**
  - Calculates perpendicular distance from point to great circle track
  - Input: Point coordinate and track endpoints
  - Output: Cross-track distance in nautical miles (signed)
  - Used for: Route deviation calculations

#### 🔍 `detect.py` - Conflict Detection Algorithms
**Purpose**: Implements deterministic conflict prediction with 10-minute lookahead
**Constants**:
- `MIN_HORIZONTAL_SEP_NM = 5.0` - ICAO horizontal separation standard
- `MIN_VERTICAL_SEP_FT = 1000.0` - ICAO vertical separation standard

**Functions**:
- **`predict_conflicts(ownship: AircraftState, traffic: List[AircraftState], lookahead_minutes: float = 10.0, time_step_seconds: float = 30.0) -> List[ConflictPrediction]`**
  - Main conflict detection function
  - Pre-filters traffic within 100 NM horizontally and ±5000 ft vertically
  - Computes CPA for each eligible intruder
  - Flags conflicts when dmin < 5 NM AND |Δalt| < 1000 ft within 10 minutes
  - Returns conflicts sorted by urgency (time to CPA)
  
- **`is_conflict(distance_nm: float, altitude_diff_ft: float, time_to_cpa_min: float) -> bool`**
  - Validates conflict criteria
  - Requires BOTH horizontal (< 5 NM) AND vertical (< 1000 ft) violations
  - Ignores past encounters (tmin < 0)
  - Implements strict separation standards per aviation requirements
  
- **`calculate_severity_score(distance_nm: float, altitude_diff_ft: float, time_to_cpa_min: float) -> float`**
  - Calculates normalized severity score [0-1]
  - Combines horizontal (40%), vertical (40%), and time (20%) factors
  - Used for conflict prioritization and resolution urgency
  
- **`project_trajectory(aircraft: AircraftState, time_horizon_minutes: float, time_step_seconds: float = 30.0) -> List[Tuple[float, float, float, float]]`**
  - Projects aircraft trajectory assuming constant velocity
  - Returns list of (time_min, lat, lon, alt_ft) waypoints
  - Used for visualization and advanced conflict analysis

#### ⚡ `resolve.py` - Conflict Resolution with Safety Validation
**Purpose**: Processes LLM-generated resolutions with comprehensive safety validation
**Constants**:
- `MAX_HEADING_CHANGE_DEG = 30.0` - Maximum allowable heading change
- `MIN_ALTITUDE_CHANGE_FT = 1000.0` - Minimum altitude change for effectiveness
- `MAX_ALTITUDE_CHANGE_FT = 2000.0` - Maximum altitude change limit
- `MIN_SAFE_SEPARATION_NM = 5.0` - Required safety margin
- `OSCILLATION_WINDOW_MIN = 10.0` - Time window for oscillation detection
- `MIN_NET_BENEFIT_THRESHOLD = 0.5` - Minimum separation improvement (nm)

**Key Classes**:
- **`CommandHistory`** - Tracks command history for oscillation detection
  - Fields: aircraft_id, command_type, timestamp, heading_change, altitude_change, separation_benefit

**Functions**:
- **`execute_resolution(llm_resolution: ResolveOut, ownship: AircraftState, intruder: AircraftState, conflict: ConflictPrediction) -> Optional[ResolutionCommand]`**
  - Main resolution execution with safety validation
  - Checks oscillation guard before proceeding
  - Validates safety using trajectory projection
  - Implements fallback strategy if LLM resolution fails
  - Records command history for future oscillation prevention
  
- **`generate_horizontal_resolution(conflict: ConflictPrediction, ownship: AircraftState, preferred_turn: str = "right") -> Optional[ResolutionCommand]`**
  - Generates horizontal conflict resolution
  - Default 20-degree turn in preferred direction
  - Validates heading change magnitude
  
- **`generate_vertical_resolution(conflict: ConflictPrediction, ownship: AircraftState, preferred_direction: str = "climb") -> Optional[ResolutionCommand]`**
  - Generates vertical conflict resolution
  - Standard 1000ft altitude change
  - Ensures altitude within reasonable bounds (1000-45000 ft)
  
- **`_check_oscillation_guard(aircraft_id: str, proposed_command_type: str, proposed_separation_benefit: float) -> bool`**
  - Prevents command oscillation by tracking recent opposite commands
  - Blocks commands that would reverse recent actions without sufficient benefit
  - Maintains 20-minute command history per aircraft
  
- **`_validate_resolution_safety(resolution_cmd: ResolutionCommand, ownship: AircraftState, intruder: AircraftState) -> bool`**
  - Projects modified trajectory with resolution applied
  - Recalculates CPA with intruder using geodesy functions
  - Ensures resolution provides adequate separation (≥5 NM OR ≥1000 ft)
  
- **`_generate_fallback_resolution(ownship: AircraftState, intruder: AircraftState, conflict: ConflictPrediction) -> Optional[ResolutionCommand]`**
  - Deterministic vertical climb (+1000 ft) when LLM fails validation
  - Multiple fallback attempts with different parameters
  - Last resort to ensure system never fails completely

#### 🤖 `llm_client.py` - LLM Integration with Safety Wrappers
**Purpose**: Provides safe interface to local Llama 3.1 8B model
**Key Classes**:
- **`LlamaClient`** - Main LLM client with safety wrappers

**Functions**:
- **`__init__(self, config: ConfigurationSettings)`**
  - Initializes client with model configuration
  - Sets up temperature, max_tokens, and model name
  - Prepares subprocess interface to ollama
  
- **`ask_detect(self, state_json: str) -> Optional[DetectOut]`**
  - Sends structured detection prompts to LLM
  - Enforces JSON-only responses for consistency
  - Validates responses against DetectOut schema
  - Implements retry logic for malformed responses
  
- **`ask_resolve(self, state_json: str, conflict: Dict[str, Any]) -> Optional[ResolveOut]`**
  - Generates conflict resolution using LLM reasoning
  - Action-specific prompts (turn/climb/descend constraints)
  - Safety-focused reasoning requirements
  - Parameter validation and bounds checking
  
- **`detect_conflicts(self, input_data: LLMDetectionInput) -> Optional[LLMDetectionOutput]`**
  - High-level conflict detection using LLM
  - Converts structured input to JSON format
  - Returns structured LLMDetectionOutput with confidence scores
  
- **`generate_resolution(self, input_data: LLMResolutionInput) -> Optional[LLMResolutionOutput]`**
  - High-level resolution generation using LLM
  - Creates ResolutionCommand from LLM output
  - Includes reasoning and risk assessment
  
- **`_call_llm(self, prompt: str) -> Optional[str]`**
  - **STATUS**: ❌ NOT IMPLEMENTED (Uses hardcoded mock responses)
  - **PURPOSE**: Core LLM communication with retry logic
  - **REQUIRED**: Ollama subprocess integration with JSON parsing
  - **BLOCKING**: No real LLM reasoning, only simulated responses
  
- **`_create_detection_prompt(self, state_json: str) -> str`**
  - Creates structured prompts for conflict detection
  - Includes context-aware traffic state representation
  - Enforces JSON response format
  
- **`_create_resolution_prompt(self, state_json: str, conflict: Dict[str, Any]) -> str`**
  - Creates action-specific prompts for resolution generation
  - Includes safety constraints and reasoning requirements
  - Specifies parameter bounds and validation criteria

#### 📋 `schemas.py` - Pydantic Data Models and Validation
**Purpose**: Defines type-safe data structures for all system components
**Key Enums**:
- **`ResolutionType`** - HEADING_CHANGE, SPEED_CHANGE, ALTITUDE_CHANGE, COMBINED

**Core Models**:
- **`AircraftState`** - Complete aircraft state representation
  - Position: latitude, longitude, altitude_ft
  - Velocity: ground_speed_kt, heading_deg, vertical_speed_fpm
  - Metadata: aircraft_id, timestamp, callsign, aircraft_type, destination
  - Validators: Normalizes heading to [0, 360) range
  
- **`ConflictPrediction`** - Structured conflict detection results
  - Geometry: time_to_cpa_min, distance_at_cpa_nm, altitude_diff_ft
  - Assessment: is_conflict, severity_score, conflict_type
  - Metadata: prediction_time, confidence
  
- **`ResolutionCommand`** - Validated resolution commands
  - Identification: resolution_id, target_aircraft, resolution_type
  - Parameters: new_heading_deg, new_speed_kt, new_altitude_ft
  - Timing: issue_time, expected_completion_time
  - Validation: is_validated, safety_margin_nm
  
- **`DetectOut`** - LLM conflict detection output schema
  - Fields: conflict (bool), intruders (list)
  - Config: Allows additional fields from LLM
  
- **`ResolveOut`** - LLM conflict resolution output schema
  - Fields: action (str), params (dict), rationale (str)
  - Actions: "turn", "climb", "descend"
  
- **`LLMDetectionInput/Output`** - Structured LLM interfaces for detection
  - Input: ownship, traffic, lookahead_minutes, current_time, context
  - Output: conflicts_detected, assessment, confidence, reasoning
  
- **`LLMResolutionInput/Output`** - Structured LLM interfaces for resolution
  - Input: conflict, ownship, traffic, constraints, context
  - Output: recommended_resolution, alternatives, reasoning, risk_assessment
  
- **`ConfigurationSettings`** - System parameters with validation
  - Timing: polling_interval_min, lookahead_time_min
  - Separation: min_horizontal_separation_nm, min_vertical_separation_ft
  - LLM: model_name, temperature, max_tokens
  - Safety: safety_buffer_factor, max_resolution_angle_deg
  - BlueSky: host, port, timeout_sec

#### 🔄 `pipeline.py` - Main Execution Pipeline
**Purpose**: Implements the core 5-minute polling loop
**Key Classes**:
- **`CDRPipeline`** - Main pipeline orchestrator

**Functions**:
- **`__init__(self, config: ConfigurationSettings)`**
  - Initializes all system components
  - Sets up BlueSky client, LLM client, metrics collector
  - Prepares state tracking dictionaries
  
- **`run(self, max_cycles: Optional[int] = None, ownship_id: str = "OWNSHIP") -> None`**
  - Main execution loop with configurable cycle limits
  - Handles timing to maintain 5-minute polling intervals
  - Implements graceful shutdown on KeyboardInterrupt
  - Logs cycle performance and sleep times
  
- **`_execute_cycle(self, ownship_id: str) -> None`**
  - Single cycle execution:
    1. Fetch aircraft states from BlueSky
    2. Identify ownship and traffic
    3. Predict conflicts using deterministic algorithms
    4. Generate and execute resolutions for conflicts
    5. Update performance metrics
  
- **`_fetch_aircraft_states(self) -> List[AircraftState]`**
  - **STATUS**: ❌ NOT IMPLEMENTED (Returns empty list)
  - **PURPOSE**: Interfaces with BlueSky to get current traffic picture
  - **REQUIRED**: BlueSky TCP integration
  - **BLOCKING**: Cannot execute real CDR cycles
  
- **`_predict_conflicts(self, ownship: AircraftState, traffic: List[AircraftState]) -> List[ConflictPrediction]`**
  - Calls conflict detection algorithms
  - Returns prioritized list of conflicts
  
- **`_handle_conflict(self, conflict: ConflictPrediction, ownship: AircraftState, traffic: List[AircraftState]) -> None`**
  - Orchestrates conflict resolution process
  - Generates resolution options
  - Validates and executes safe resolutions
  - Maintains active resolution tracking
  
- **`_generate_resolution(self, conflict: ConflictPrediction, ownship: AircraftState, traffic: List[AircraftState]) -> Optional[ResolutionCommand]`**
  - Integrates LLM-based and deterministic resolution methods
  - Applies safety validation before execution
  - Returns validated resolution command

#### 🔗 `bluesky_io.py` - BlueSky Simulator Interface
**Purpose**: Provides clean interface to BlueSky TCP socket
**Key Classes**:
- **`BlueSkyClient`** - TCP client for BlueSky communication

**Functions**:
- **`__init__(self, config: ConfigurationSettings)`**
  - Configures connection parameters (host, port, timeout)
  - Initializes socket interface
  
- **`connect(self) -> bool`**
  - Establishes TCP connection to BlueSky simulator
  - Handles connection errors gracefully
  - Returns success status
  
- **`send_command(self, command: str) -> bool`**
  - Sends commands to BlueSky with error handling
  - Formats commands with required newline termination
  - Logs all command transactions
  
- **`get_aircraft_states(self) -> List[AircraftState]`**
  - **STATUS**: ❌ NOT IMPLEMENTED (Returns empty list)
  - **PURPOSE**: Fetches real-time aircraft states from BlueSky
  - **REQUIRED**: BlueSky TCP "STATE" command implementation
  - **BLOCKING**: Cannot run with live simulator data
  
- **`execute_command(self, resolution: ResolutionCommand) -> bool`**
  - Executes resolution commands via BlueSky
  - Converts ResolutionCommand to BlueSky format
  - Returns execution success status
  
- **`create_aircraft(self, aircraft_id: str, aircraft_type: str, lat: float, lon: float, hdg: float, alt: float, spd: float) -> bool`**
  - Creates new aircraft in simulation
  - BlueSky CRE command format
  
- **`set_heading(self, aircraft_id: str, heading_deg: float) -> bool`**
  - Issues heading change command (HDG)
  - Formats heading to 3-digit string
  
- **`set_altitude(self, aircraft_id: str, altitude_ft: float) -> bool`**
  - Issues altitude change command (ALT)
  - Formats altitude to integer feet
  
- **`set_speed(self, aircraft_id: str, speed_kt: float) -> bool`**
  - Issues speed change command (SPD)
  - Formats speed to integer knots
  
- **`_resolution_to_bluesky_command(self, resolution: ResolutionCommand) -> Optional[str]`**
  - Converts ResolutionCommand to BlueSky command format
  - Handles different resolution types (heading, altitude, speed)
  - Validates command parameters

#### 📊 `metrics.py` - Wolfgang (2011) KPI Calculations
**Purpose**: Implements comprehensive performance metrics following aviation research standards
**Key Classes**:
- **`MetricsSummary`** - Complete performance metrics summary
- **`BaselineMetrics`** - Baseline system performance for comparison
- **`ComparisonReport`** - LLM vs baseline performance comparison
- **`MetricsCollector`** - Main metrics collection and calculation engine

**MetricsSummary Fields**:
- **Timing**: total_simulation_time_min, total_cycles, avg_cycle_time_sec
- **Detection**: total_conflicts_detected, true_conflicts, false_positives, false_negatives, detection_accuracy
- **Wolfgang KPIs** ⚠️ **(NEED CORRECTION)**: tbas, lat, pa, pi, dat, dfa, re, ri, rat
- **Safety**: min_separation_achieved_nm, avg_separation_nm, safety_violations
- **Resolution**: total_resolutions_issued, successful_resolutions, resolution_success_rate

**MetricsCollector Functions**:
- **`reset(self) -> None`**
  - Resets all metrics to initial state
  - Clears tracking dictionaries and lists
  
- **`record_cycle_time(self, cycle_duration_sec: float) -> None`**
  - Records execution time for performance monitoring
  
- **`record_conflict_detection(self, conflicts: List[ConflictPrediction], detection_time: datetime) -> None`**
  - Records detection results with timestamps
  - Tracks alert times for Wolfgang metrics
  
- **`record_ground_truth(self, true_conflicts: List[ConflictPrediction]) -> None`**
  - Records actual conflicts for validation
  - Used to calculate false positives/negatives
  
- **`record_resolution_issued(self, resolution: ResolutionCommand, issue_time: datetime) -> None`**
  - Tracks resolution commands and timing
  
- **`record_separation_achieved(self, ownship_id: str, intruder_id: str, separation_nm: float, timestamp: datetime) -> None`**
  - Records actual separation achieved
  - Monitors safety violations
  
- **`calculate_wolfgang_metrics(self) -> Dict[str, float]`**
  - **STATUS**: ⚠️ INCORRECT DEFINITIONS (Need correction per Wolfgang 2011)
  - **PURPOSE**: Calculates Wolfgang (2011) KPIs with correct definitions:
    - **TBAS**: Time Between Analysis and Scenario
    - **LAT**: Look-Ahead Time
    - **PA**: Prediction Accuracy  
    - **PI**: Prediction Integrity
    - **DAT**: Detection Analysis Time
    - **DFA**: Detection False Alert
    - **RE**: Resolution Efficiency
    - **RI**: Resolution Integrity
    - **RAT**: Resolution Analysis Time
  - **REQUIRED**: Update metric calculations to match correct definitions
  
- **`generate_summary(self) -> MetricsSummary`**
  - Generates comprehensive performance summary
  - Calculates derived metrics and statistics
  
- **`export_to_json(self, filepath: str) -> None`**
  - Exports metrics to JSON for sprint reports
  - Includes timestamps and metadata

### API Service (`src/api/`)

#### 🌐 `service.py` - FastAPI REST Service
**Purpose**: Provides web interface for system monitoring and control
**Key Classes**:
- **`PipelineStatus`** - Pipeline status response model
- **`StartPipelineRequest`** - Pipeline start request model

**Endpoints**:
- **`GET /`** - Root endpoint with API information
- **`GET /health`** - Health check with component status
- **`GET /pipeline/status`** - Current pipeline status
- **`POST /pipeline/start`** - Start CDR pipeline
- **`POST /pipeline/stop`** - Stop pipeline execution
- **`GET /metrics`** - Current performance metrics
- **`GET /config`** - System configuration
- **`POST /config`** - Update configuration
- **`GET /conflicts`** - Recent conflict detections
- **`GET /resolutions`** - Recent resolution commands

**Global Variables**:
- `pipeline: Optional[CDRPipeline]` - Global pipeline instance
- `pipeline_task: Optional[asyncio.Task]` - Async pipeline task

## Test Suite (`tests/`)

### Comprehensive Test Coverage ⚠️ **(UNVERIFIED - CI EXECUTION REQUIRED)**

**Note**: Test coverage claims require validation through CI execution. Current status unknown.

#### 📍 `test_geodesy.py` - Geodesy Function Tests
**Test Classes**:
- **`TestHaversine`** - Distance calculation tests
  - Symmetry validation
  - Zero distance verification
  - Known distance comparisons (Stockholm-Gothenburg)
  - Equator and meridian calculations
  
- **`TestBearing`** - Bearing calculation tests
  - Cardinal direction validation (N, E, S, W)
  - Angle normalization
  
- **`TestCPA`** - Closest Point of Approach tests
  - Basic converging scenarios
  - Parallel flight paths
  - Diverging aircraft
  - Edge cases (same position, opposite directions)
  
- **`TestCrossTrack`** - Cross-track distance tests
  - Perpendicular distance calculations
  - Track deviation measurements

#### 🔍 `test_detect.py` - Conflict Detection Tests
**Test Coverage**:
- Conflict criteria validation (safe/unsafe separation)
- CPA-based detection with various aircraft geometries
- Severity scoring edge cases and bounds
- False alert prevention for diverging aircraft
- Pre-filtering logic (100 NM / ±5000 ft limits)

#### ⚡ `test_resolve.py` - Resolution Algorithm Tests
**Test Coverage**:
- Horizontal resolution generation and validation
- Vertical resolution generation and validation
- Safety validation with trajectory projection
- Oscillation guard functionality
- Fallback resolution strategies
- Command history tracking

#### 🤖 `test_llm_mock_schema.py` - LLM Integration Tests
**Test Coverage**:
- Schema validation for DetectOut/ResolveOut
- JSON parsing and error handling
- Mock LLM response validation
- Input/output format consistency

#### 🔄 `test_pipeline_smoke.py` - Integration Smoke Tests
**Test Coverage**:
- Pipeline initialization and configuration
- Cycle execution without BlueSky
- Component integration validation
- Error handling and recovery

#### 📊 `test_sprint2_sprint3.py` - Comprehensive Integration Tests
**Test Coverage**:
- End-to-end detection → LLM → resolution → validation pipeline
- Metrics collection through full operational cycle
- Real scenario testing with Stockholm airspace
- Wolfgang metrics calculation validation

#### 🎯 `test_sprint4_integration.py` - Advanced Integration Tests
**Test Coverage**:
- Multi-aircraft conflict scenarios
- Complex geometry testing
- Performance benchmarking
- System stress testing

#### 🏆 `test_sprint5_comprehensive.py` - Final Validation Tests
**Test Coverage**:
- Complete system validation
- Edge case scenario testing
- Performance regression testing
- Safety validation comprehensive coverage

## Configuration and Setup Files

### 📦 `pyproject.toml` - Project Configuration
**Sections**:
- **[project]**: Package metadata, dependencies, classifiers
- **[tool.black]**: Code formatting configuration (line-length: 88)
- **[tool.ruff]**: Linting rules and error codes
- **[tool.pytest]**: Test discovery and coverage settings
- **[tool.mypy]**: Type checking configuration

### 📋 `requirements.txt` - Dependencies
**Core Dependencies**:
- `numpy==1.24.3` - Numerical computations
- `pandas==2.0.3` - Data analysis and metrics
- `pydantic==2.0.3` - Data validation and schemas
- `bluesky-simulator[full]==1.3.0` - Air traffic simulation
- `fastapi==0.100.1` - Web API framework
- `uvicorn[standard]==0.22.0` - ASGI server

**Testing Framework**:
- `pytest==7.4.0` - Test framework
- `pytest-cov==4.1.0` - Coverage analysis

**Code Quality**:
- `black==23.7.0` - Code formatting
- `ruff==0.0.280` - Fast linting
- `mypy==1.4.1` - Type checking

**Visualization**:
- `matplotlib==3.7.2` - Plotting and visualization
- `seaborn==0.12.2` - Statistical plotting
- `rich==13.4.2` - Terminal formatting

## Utilities and Scripts

### 📊 `visualize_conflicts.py` - Conflict Visualization Tool
**Purpose**: Creates visualization plots for conflict scenarios
**Functions**:
- **`simple_haversine_nm(lat1, lon1, lat2, lon2) -> float`**
  - Simplified haversine calculation for plotting
  
- **`simple_cpa(own_lat, own_lon, own_spd, own_hdg, intr_lat, intr_lon, intr_spd, intr_hdg) -> Tuple[float, float]`**
  - Simplified CPA calculation for visualization
  
- **`create_conflict_plot(scenarios: List[Dict]) -> None`**
  - Generates matplotlib plots showing:
    - Aircraft positions and trajectories
    - Detected conflicts with CPA points
    - Resolution commands and outcomes
    - Safety margins and separation standards

**Usage**:
```bash
python visualize_conflicts.py
```

## Scenarios and Test Data

### 📁 `scenarios/` - Test Scenarios
- **`blue_sky_init.txt`** - BlueSky initialization script
- **`sample_scat_ownship.json`** - Sample SCAT trajectory data

### 📁 `tests/data/` - Test Fixtures
- **`README.md`** - Test data documentation
- Sample aircraft states and conflict predictions
- Test scenario data for unit tests

## Reports and Documentation

### 📁 `reports/` - Sprint Reports and Analysis
- **`sprint_0/README.md`** - Foundation sprint report
- **`sprint_05/stress_test_metrics.csv`** - Performance test results

### 📁 `htmlcov/` - Code Coverage Reports
- Detailed HTML coverage reports
- Function-level coverage analysis
- Visual coverage indicators

## Development Workflow and Standards

### Code Quality Standards ⚠️ **(UNVERIFIED)**
- **100% Type Coverage**: All functions have complete type annotations *(unverified)*
- **95%+ Test Coverage**: Comprehensive unit and integration tests *(requires CI validation)*
- **Black Formatting**: Consistent code style with 88-character lines
- **Ruff Linting**: Fast, comprehensive linting with aviation-specific rules *(unverified)*
- **Mypy Type Checking**: Static type validation *(unverified)*

### Safety Standards
- **Aviation Compliance**: ICAO separation standards (5 NM / 1000 ft)
- **Hard Validation**: All LLM outputs validated before execution
- **Oscillation Prevention**: Command history tracking prevents thrashing
- **Fallback Strategies**: Deterministic backups for all LLM failures
- **Audit Trails**: Complete logging of all decisions and validations

### Performance Standards
- **5-Minute Cycles**: Real-time polling with predictable timing
- **10-Minute Lookahead**: Standard aviation conflict prediction horizon
- **Sub-Second Response**: Conflict detection and resolution under 1 second
- **Wolfgang Metrics**: Industry-standard KPI measurements

### Testing Philosophy
- **Test-Driven Development**: Tests written before implementation
- **Edge Case Coverage**: Boundary conditions and failure modes
- **Integration Testing**: End-to-end pipeline validation
- **Performance Testing**: Stress testing with multiple aircraft
- **Safety Testing**: Validation of all safety constraints

## ⚠️ **CRITICAL: What's Missing for Real Execution**

### **Bottom Line Assessment**
- **Design**: ✅ Complete, well-architected, safety-first approach
- **Executable End-to-End Run**: ❌ **BLOCKED** by missing integrations

### **Immediate Action Items for Deployment**

#### 1. **BlueSky TCP Integration** (HIGHEST PRIORITY)
```python
# File: src/cdr/bluesky_io.py
def get_aircraft_states(self) -> List[AircraftState]:
    """IMPLEMENT: Send 'STATE' command to BlueSky TCP socket"""
    # self.socket.send(b"STATE\n")
    # response = self.socket.recv(4096)
    # return parse_aircraft_states(response)
    pass  # Currently returns []
```

#### 2. **LLM Ollama Integration** (HIGHEST PRIORITY)
```python
# File: src/cdr/llm_client.py
def _call_llm(self, prompt: str) -> Optional[str]:
    """IMPLEMENT: Call local Ollama Llama 3.1 8B model"""
    # result = subprocess.run(["ollama", "run", "llama3.1:8b", prompt], 
    #                        capture_output=True, text=True)
    # return json.loads(result.stdout)
    pass  # Currently returns hardcoded mock responses
```

#### 3. **Wolfgang Metrics Correction** (MEDIUM PRIORITY)
```python
# File: src/cdr/metrics.py
# FIX: Correct metric definitions per Wolfgang (2011)
# TBAS = Time Between Analysis and Scenario (not "Time-Based Alerting Score")
# LAT = Look-Ahead Time (not "Loss of Alerting Time")
# etc.
```

#### 4. **CI/CD Validation** (LOW PRIORITY)
```bash
# Verify claimed test coverage and code quality
pytest --cov=src --cov-report=html
black . && ruff . && mypy .
```

### **Development Priorities**

1. **Week 1**: Implement BlueSky TCP interface for aircraft state fetching
2. **Week 2**: Deploy Ollama and integrate real LLM calls with JSON parsing
3. **Week 3**: Correct Wolfgang metric definitions and calculations
4. **Week 4**: End-to-end testing with live BlueSky simulator

**Until these are complete, the system remains a well-designed prototype without execution capability.**

## Future Development Roadmap

### Sprint 6: BlueSky Integration
- Complete BlueSky TCP interface implementation
- Real-time aircraft state fetching
- Command execution validation
- Integration testing with live simulator

### Sprint 7: LLM Deployment
- Local Llama 3.1 8B model setup
- Ollama integration and optimization
- Prompt engineering refinement
- Performance benchmarking

### Sprint 8: Advanced Features
- Multi-aircraft coordination
- Route-based conflict prediction
- Weather integration
- Enhanced visualization

### Sprint 9: Performance Optimization
- Algorithm optimization for large-scale scenarios
- Memory usage optimization
- Database integration for historical analysis
- Real-time dashboard development

### Sprint 10: Production Deployment
- Docker containerization
- Kubernetes deployment
- Monitoring and alerting
- Production safety validation

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Ensure code quality (`black . && ruff . && mypy .`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Contact

**Author**: Somnath  
**Email**: somnathab3@gmail.com  
**Repository**: ATC_LLM  
**Current Branch**: main

## Contributing

[Add contribution guidelines here]
