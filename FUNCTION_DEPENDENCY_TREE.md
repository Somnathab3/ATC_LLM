# ATC-LLM System Function Dependency Tree

This document provides a comprehensive analysis of all classes, functions, and their dependencies within the ATC-LLM system.

## System Architecture Overview

```
ATC-LLM System
├── CLI Interface (cli.py, src/atc_llm_cli.py)
├── Core CDR Pipeline (src/cdr/pipeline.py)
├── BlueSky Simulation (src/cdr/bluesky_io.py)
├── LLM Integration (src/cdr/llm_client.py)
├── SCAT Data Processing (src/cdr/scat_adapter.py)
├── Conflict Detection (src/cdr/detect.py)
├── Resolution Engines (src/cdr/resolve.py)
├── Metrics & Reporting (src/cdr/metrics.py, src/cdr/reporting.py)
├── Executable Scripts (bin/)
└── Utility Scripts (scripts/)
```

## Core Module Dependencies

### 1. Data Models & Schemas (`src/cdr/schemas.py`)

**Classes:**
- `AircraftState(BaseModel)`: Aircraft position, velocity, and state
- `ConflictPrediction(BaseModel)`: Conflict detection results
- `ResolutionCommand(BaseModel)`: Resolution command structure
- `ConfigurationSettings(BaseSettings)`: System configuration
- `FlightRecord(BaseModel)`: Flight track data
- `IntruderScenario(BaseModel)`: Scenario definition
- `BatchSimulationResult(BaseModel)`: Batch processing results
- `MonteCarloParameters(BaseModel)`: Monte Carlo configuration
- `ConflictResolutionMetrics`: Detailed conflict metrics
- `ScenarioMetrics`: Scenario-level analytics
- `PathComparisonMetrics`: Path comparison analysis
- `EnhancedReportingSystem`: Advanced reporting framework

**Enums:**
- `ResolutionType`: HEADING_CHANGE, SPEED_CHANGE, ALTITUDE_CHANGE, COMBINED
- `ResolutionEngine`: HORIZONTAL, VERTICAL, DETERMINISTIC, FALLBACK

**Dependencies:**
- `pydantic`: Data validation and settings management
- `pydantic_settings`: Environment variable integration
- `datetime`: Timestamp handling
- `enum`: Type definitions
- `pathlib`: File path handling

**Key Variables:**
- `min_horizontal_separation_nm`: Safety separation distance
- `min_vertical_separation_ft`: Vertical separation standard
- `lookahead_time_min`: Conflict prediction horizon
- `polling_interval_min`: System update frequency

### 2. Aviation Mathematics (`src/cdr/geodesy.py`)

**Functions:**
- `haversine_nm(lat1, lon1, lat2, lon2)`: Great circle distance calculation
- `bearing_rad(lat1, lon1, lat2, lon2)`: Bearing calculation in radians
- `bearing_deg(lat1, lon1, lat2, lon2)`: Bearing calculation in degrees
- `destination_point_nm(lat, lon, bearing_rad, distance_nm)`: Position projection
- `cpa_nm(own, intruder)`: Closest Point of Approach calculation
- `normalize_heading_deg(heading)`: Heading normalization (0-360°)

**Dependencies:**
- `math`: Mathematical operations
- `numpy`: Numerical computations

**Usage Variables:**
- Earth radius constants for aviation calculations
- Coordinate transformation matrices
- Aviation-standard distance units

### 3. BlueSky Simulation Interface (`src/cdr/bluesky_io.py`)

**Classes:**
- `BSConfig`: BlueSky configuration parameters
- `BlueSkyClient`: Main BlueSky interface

**Key Methods:**
- `BlueSkyClient.connect()`: Establish BlueSky connection
- `BlueSkyClient.create_aircraft()`: Create aircraft in simulation
- `BlueSkyClient.get_aircraft_states()`: Retrieve current aircraft positions
- `BlueSkyClient.stack()`: Execute BlueSky commands
- `BlueSkyClient.step_simulation()`: Advance simulation time
- `BlueSkyClient.set_dtlook()`: Configure conflict detection parameters
- `BlueSkyClient.set_cdmethod()`: Set conflict detection method
- `BlueSkyClient.setup_baseline()`: Configure simulation baseline

**Dependencies:**
- `bluesky-simulator`: Core simulation engine
- `telnetlib`: Communication protocol
- `threading`: Concurrent operations
- `socket`: Network communication
- `atexit`: Cleanup handlers

**Configuration Variables:**
- `headless`: GUI mode setting
- `dtlook_sec`: Look-ahead time for conflict detection
- `dtmult`: Time multiplier for simulation speed
- `cdmethod`: Conflict detection method (BS, GEOMETRIC)
- `asas_enabled`: Automated Separation Assurance System
- `realtime`: Real-time pacing mode

### 4. LLM Integration (`src/cdr/llm_client.py`)

**Classes:**
- `LlamaClient`: Ollama LLM interface
- `PromptBuilderV2`: Enhanced prompt construction

**Key Methods:**
- `LlamaClient.detect_conflicts()`: LLM-based conflict detection
- `LlamaClient.resolve_conflicts()`: LLM-based resolution generation
- `LlamaClient.send_request()`: HTTP communication with Ollama
- `PromptBuilderV2.build_detection_prompt()`: Conflict detection prompts
- `PromptBuilderV2.build_resolution_prompt()`: Resolution prompts

**Dependencies:**
- `requests`: HTTP client for Ollama API
- `json`: Data serialization
- `logging`: Error tracking
- `typing`: Type hints

**Configuration Variables:**
- `ollama_base_url`: Ollama service endpoint
- `llm_model_name`: Model identifier (e.g., "llama3.1:8b")
- `llm_temperature`: Response randomness
- `llm_max_tokens`: Response length limit

### 5. SCAT Data Processing (`src/cdr/scat_adapter.py`)

**Classes:**
- `SCATFlightRecord`: Individual flight track
- `VicinityQueryPerformance`: Performance monitoring
- `VicinityIndex`: Spatial indexing for efficient queries
- `SCATAdapter`: Main SCAT data interface

**Key Methods:**
- `SCATAdapter.load_flights()`: Load flight data from SCAT files
- `SCATAdapter.find_vicinity_aircraft()`: Spatial proximity queries
- `SCATAdapter.get_flight_states_at_time()`: Time-based state retrieval
- `load_scat_scenario()`: Scenario loading utility

**Dependencies:**
- `json`: SCAT file parsing
- `logging`: Debug information
- `datetime`: Time handling
- `pathlib`: File system operations
- `numpy`: Performance optimizations

**Key Variables:**
- `vicinity_radius_nm`: Search radius for nearby aircraft
- `time_window_sec`: Temporal search window
- `max_flights`: Processing limits
- `flight_duration_min`: Scenario duration

### 6. Conflict Detection (`src/cdr/detect.py`)

**Functions:**
- `predict_conflicts()`: Main conflict prediction
- `predict_conflicts_enhanced()`: Enhanced CPA-based detection
- `is_conflict()`: Boolean conflict determination
- `calculate_severity_score()`: Conflict severity assessment
- `project_trajectory()`: Aircraft trajectory projection

**Dependencies:**
- `math`: Geometric calculations
- `enhanced_cpa`: Enhanced CPA calculations
- `schemas`: Data structures

**Key Variables:**
- `MIN_HORIZONTAL_SEP_NM`: 5.0 NM separation standard
- `MIN_VERTICAL_SEP_FT`: 1000 ft separation standard
- `lookahead_time_min`: Prediction horizon

### 7. Enhanced CPA Analysis (`src/cdr/enhanced_cpa.py`)

**Classes:**
- `CPAResult`: Closest Point of Approach results
- `MinSepCheck`: Minimum separation validation

**Functions:**
- `calculate_enhanced_cpa()`: Advanced CPA calculation
- `check_minimum_separation()`: Current separation verification
- `calculate_adaptive_cadence()`: Dynamic polling intervals
- `cross_validate_with_bluesky()`: BlueSky validation
- `_project_position()`: Future position calculation
- `_calculate_relative_speed()`: Relative velocity calculation
- `_calculate_convergence_rate()`: Convergence analysis
- `_calculate_cpa_confidence()`: Confidence scoring

**Dependencies:**
- `math`: Trigonometric calculations
- `geodesy`: Aviation mathematics
- `schemas`: Data structures

### 8. Resolution Engines (`src/cdr/resolve.py`)

**Classes:**
- `CommandHistory`: Resolution command tracking

**Functions:**
- `execute_resolution()`: Main resolution execution
- `generate_horizontal_resolution()`: Horizontal conflict resolution
- `generate_vertical_resolution()`: Vertical conflict resolution
- `validate_resolution()`: Safety validation
- `apply_resolution()`: Resolution application
- `to_bluesky_command()`: BlueSky command formatting
- `format_resolution_command()`: Command formatting
- `_validate_resolution_safety()`: Safety checks
- `_generate_fallback_resolution()`: Fallback strategies

**Dependencies:**
- `math`: Resolution calculations
- `geodesy`: Geometric operations
- `nav_utils`: Navigation utilities
- `schemas`: Data structures

### 9. Metrics Collection (`src/cdr/metrics.py`)

**Classes:**
- `MetricsSummary`: Aggregated metrics
- `BaselineMetrics`: Baseline performance
- `ComparisonReport`: Comparative analysis
- `MetricsCollector`: Main metrics collection

**Key Methods:**
- `MetricsCollector.record_conflict()`: Conflict logging
- `MetricsCollector.record_resolution()`: Resolution tracking
- `MetricsCollector.calculate_summary()`: Summary generation
- `MetricsCollector.generate_report()`: Report creation

**Dependencies:**
- `numpy`: Statistical calculations
- `datetime`: Timestamp handling
- `json`: Data serialization

### 10. Main Pipeline Orchestration (`src/cdr/pipeline.py`)

**Functions:**
- `_get_aircraft_id()`: Aircraft identification
- `_get_position()`: Position extraction
- `_get_velocity()`: Velocity extraction
- `_asdict_state()`: State serialization

**Dependencies:**
- All core CDR modules
- `collections`: Data structures
- `json`: Serialization
- `pathlib`: File operations

## Executable Scripts (`bin/`)

### CLI Interface (`cli.py`)

**Main Functions:**
- `cmd_health_check()`: System health verification
- `cmd_simulate()`: Unified simulation interface
- `cmd_batch()`: Batch processing operations
- `cmd_metrics()`: Metrics calculation
- `cmd_report()`: Enhanced reporting
- `cmd_verify_llm()`: LLM connectivity testing

**Dependencies:** All core modules plus argparse for command-line interface

### SCAT LLM Simulation (`bin/complete_scat_llm_simulation.py`)

**Functions:**
- `build_route_from_scat()`: Route construction from SCAT data
- `simulate_aircraft_movement()`: Aircraft movement simulation
- `predict_conflict_with_llm()`: LLM-based conflict prediction
- `generate_llm_resolution()`: LLM resolution generation
- `create_aircraft_state()`: Aircraft state creation

**Dependencies:**
- Core CDR pipeline
- SCAT adapter
- BlueSky interface
- LLM client

### SCAT LLM Runner (`bin/scat_llm_run.py`)

**Classes:**
- `SCATLLMRunner`: Real-time SCAT processing

**Dependencies:**
- Full CDR pipeline
- Real-time processing capabilities

### SCAT Baseline Generator (`bin/scat_baseline.py`)

**Classes:**
- `NeighborAircraft`: Neighboring aircraft tracking
- `BaselineReport`: Baseline analysis results
- `SCATBaselineGenerator`: Baseline generation engine

**Dependencies:**
- SCAT adapter
- Geodesy calculations
- Output utilities

### Batch Processors

**Production Batch Processor (`bin/production_batch_processor.py`):**
- Production-ready batch processing
- Safety validation
- Error handling

**Demo Baseline vs LLM (`bin/demo_baseline_vs_llm.py`):**
- Comparative analysis
- Performance benchmarking
- Statistical reporting

## Utility Scripts (`scripts/`)

### Route Conflict Analysis (`scripts/run_route_conflict_full.py`)

**Functions:**
- Full route conflict analysis
- Fast-time simulation configuration
- Enhanced CPA integration

### Unicode Cleaner (`scripts/unicode_cleaner.py`)

**Functions:**
- Data cleaning utilities
- Unicode handling for SCAT files

### Performance Testing (`scripts/test_vicinity_performance.py`)

**Functions:**
- Vicinity query performance testing
- Spatial indexing optimization

## Variable Dependencies and Usage

### Configuration Variables
- **Safety Parameters:** `min_horizontal_separation_nm`, `min_vertical_separation_ft`
- **Performance:** `polling_interval_min`, `lookahead_time_min`
- **LLM Settings:** `llm_model_name`, `llm_temperature`, `llm_max_tokens`
- **BlueSky Config:** `bluesky_host`, `bluesky_port`, `bluesky_timeout_sec`

### State Variables
- **Aircraft State:** `callsign`, `latitude`, `longitude`, `altitude_ft`, `heading_deg`, `speed_kts`
- **Conflict Data:** `time_to_closest_approach_min`, `closest_approach_distance_nm`, `severity_score`
- **Resolution Commands:** `resolution_type`, `target_value`, `execution_time`

### Performance Metrics
- **Success Rates:** `conflicts_detected`, `conflicts_resolved`, `resolution_success_rate`
- **Timing:** `detection_time_ms`, `resolution_time_ms`, `total_processing_time_ms`
- **Safety:** `minimum_separation_achieved`, `safety_margin`, `violation_count`

## Inter-Module Communication Flow

```
CLI → Pipeline → [BlueSky, LLM, SCAT] → Detect → Resolve → Metrics → Reporting
```

1. **CLI** parses user commands and initializes appropriate modules
2. **Pipeline** orchestrates the overall CDR process
3. **BlueSky/SCAT** provides aircraft state data
4. **Detect** identifies potential conflicts
5. **LLM** generates intelligent resolutions
6. **Resolve** validates and executes resolutions
7. **Metrics** tracks performance and safety
8. **Reporting** generates comprehensive analysis

This dependency tree ensures proper separation of concerns while maintaining tight integration for optimal conflict detection and resolution performance.
