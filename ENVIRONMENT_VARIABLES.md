# Environment Variable Configuration

The ATC-LLM system supports configuration via environment variables. All environment variables use the `ATC_LLM_` prefix.

## Usage

```bash
# Set environment variables
export ATC_LLM_LLM_MODEL_NAME=llama3.1:8b
export ATC_LLM_OLLAMA_BASE_URL=http://localhost:11434

# Run with overridden configuration
atc-llm --dump-config
```

## Available Environment Variables

### LLM Configuration
- `ATC_LLM_LLM_ENABLED` - Enable/disable LLM (default: true)
- `ATC_LLM_LLM_MODEL_NAME` - LLM model identifier (default: "llama3.1:8b")
- `ATC_LLM_LLM_TEMPERATURE` - LLM temperature 0-1 (default: 0.1)
- `ATC_LLM_LLM_MAX_TOKENS` - Maximum tokens per request (default: 2048)
- `ATC_LLM_OLLAMA_BASE_URL` - Ollama API base URL (default: "http://localhost:11434")

### Timing Settings
- `ATC_LLM_POLLING_INTERVAL_MIN` - Polling interval in minutes (default: 5.0)
- `ATC_LLM_LOOKAHEAD_TIME_MIN` - Prediction horizon in minutes (default: 10.0)
- `ATC_LLM_SNAPSHOT_INTERVAL_MIN` - Snapshot interval for trend analysis (default: 1.0)

### Separation Standards
- `ATC_LLM_MIN_HORIZONTAL_SEPARATION_NM` - Minimum horizontal separation in NM (default: 5.0)
- `ATC_LLM_MIN_VERTICAL_SEPARATION_FT` - Minimum vertical separation in feet (default: 1000.0)

### Prompt Builder Settings
- `ATC_LLM_MAX_INTRUDERS_IN_PROMPT` - Max intruders in LLM prompt (default: 5)
- `ATC_LLM_INTRUDER_PROXIMITY_NM` - Max distance for intruders in prompt (default: 100.0)
- `ATC_LLM_INTRUDER_ALTITUDE_DIFF_FT` - Max altitude difference for intruders (default: 5000.0)
- `ATC_LLM_TREND_ANALYSIS_WINDOW_MIN` - Time window for trend analysis (default: 2.0)

### Safety Settings
- `ATC_LLM_SAFETY_BUFFER_FACTOR` - Safety margin multiplier (default: 1.2)
- `ATC_LLM_MAX_RESOLUTION_ANGLE_DEG` - Maximum resolution angle (default: 45.0)
- `ATC_LLM_MAX_ALTITUDE_CHANGE_FT` - Maximum altitude change (default: 2000.0)
- `ATC_LLM_MAX_WAYPOINT_DIVERSION_NM` - Max distance for DCT waypoint (default: 80.0)

### Validation Settings
- `ATC_LLM_ENFORCE_OWNSHIP_ONLY` - Enforce ownship-only commands (default: true)
- `ATC_LLM_MAX_CLIMB_RATE_FPM` - Maximum climb rate in FPM (default: 3000.0)
- `ATC_LLM_MAX_DESCENT_RATE_FPM` - Maximum descent rate in FPM (default: 3000.0)
- `ATC_LLM_MIN_FLIGHT_LEVEL` - Minimum flight level (default: 100)
- `ATC_LLM_MAX_FLIGHT_LEVEL` - Maximum flight level (default: 600)
- `ATC_LLM_MAX_HEADING_CHANGE_DEG` - Maximum heading change (default: 90.0)

### Dual LLM Engine Settings
- `ATC_LLM_ENABLE_DUAL_LLM` - Enable dual LLM engines (default: true)
- `ATC_LLM_HORIZONTAL_ENGINE_ENABLED` - Enable horizontal engine (default: true)
- `ATC_LLM_VERTICAL_ENGINE_ENABLED` - Enable vertical engine (default: true)
- `ATC_LLM_HORIZONTAL_RETRY_COUNT` - Max retries for horizontal engine (default: 2)
- `ATC_LLM_VERTICAL_RETRY_COUNT` - Max retries for vertical engine (default: 2)

### Simulation Control
- `ATC_LLM_REAL_TIME_MODE` - Enable real-time simulation (default: false)
- `ATC_LLM_FAST_TIME` - Fast-time simulation mode (default: true)
- `ATC_LLM_SIM_ACCEL_FACTOR` - Simulation acceleration factor (default: 1.0)

### BlueSky Integration
- `ATC_LLM_BLUESKY_HOST` - BlueSky host (default: "localhost")
- `ATC_LLM_BLUESKY_PORT` - BlueSky port (default: 1337)
- `ATC_LLM_BLUESKY_TIMEOUT_SEC` - BlueSky timeout in seconds (default: 5.0)

## Examples

### Basic Model Override
```bash
ATC_LLM_LLM_MODEL_NAME=gpt-4 atc-llm simulate basic
```

### Production Configuration
```bash
export ATC_LLM_LLM_MODEL_NAME=llama3.1:70b
export ATC_LLM_OLLAMA_BASE_URL=http://ollama-server:11434
export ATC_LLM_MIN_HORIZONTAL_SEPARATION_NM=8.0
export ATC_LLM_SAFETY_BUFFER_FACTOR=1.5

atc-llm batch --scat-dir /data/scat --max-flights 100
```

### Debug Configuration
```bash
export ATC_LLM_LLM_TEMPERATURE=0.0
export ATC_LLM_MAX_INTRUDERS_IN_PROMPT=3
export ATC_LLM_TREND_ANALYSIS_WINDOW_MIN=1.0

atc-llm simulate --verbose
```

## Configuration File Support

The system also supports `.env` files in the project root:

```bash
# .env file
ATC_LLM_LLM_MODEL_NAME=llama3.1:8b
ATC_LLM_OLLAMA_BASE_URL=http://localhost:11434
ATC_LLM_MIN_HORIZONTAL_SEPARATION_NM=5.0
```

## Verification

To see the effective configuration with environment overrides:

```bash
atc-llm --dump-config
```

This will output the current configuration as JSON, showing which values are overridden by environment variables.
