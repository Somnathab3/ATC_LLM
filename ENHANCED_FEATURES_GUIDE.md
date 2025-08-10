# ATC LLM Enhanced Features Guide

This document provides a quick guide to the enhanced features implemented in the ATC LLM system.

## New Features Overview

### 1. LLM Memory System (Experience Library)
- **Purpose**: Stores past conflict resolution scenarios to improve future decisions
- **Implementation**: Vector-based similarity search with normalized features
- **Storage**: JSONL file with append-only experience records

### 2. Monte-Carlo Intruders with OpenAP Constraints
- **Purpose**: Generate realistic intruder aircraft with performance limits
- **Implementation**: Aircraft type sampling from OpenAP database
- **Features**: Type-specific speed, climb/descend, and turn rate limits

### 3. Failure-Focused Analysis
- **Purpose**: Track and analyze LLM failures when conflicts are missed
- **Implementation**: Comprehensive failure recording with counterfactual analysis
- **Output**: JSONL failure cases and HTML summary reports

### 4. Enhanced Navigation
- **Purpose**: Robust waypoint validation with BlueSky DIRECT commands
- **Implementation**: Navigation database lookup with automatic heading fallback
- **Commands**: DIRECT/DIRTO with distance validation

### 5. Visualization System
- **Purpose**: Real-time display of conflicts, waypoints, and Direct-To legs
- **Implementation**: Pygame-based viewer with conflict cylinders
- **Features**: Aircraft symbols, separation circles, and Direct-To arrows

### 6. Repository Cleanup
- **Purpose**: Automated cleanup of old demos and stale outputs
- **Implementation**: Pattern-based file categorization with dry-run support
- **Safety**: Preserves essential files referenced by CLI and function tree

## Command Line Interface

### Enhanced Health Check
```bash
atc-llm health-check --verbose
```
- Checks enhanced feature dependencies (NumPy, Pygame, OpenAP)
- Validates memory system and visualization components

### Repository Cleanup
```bash
# Dry run to see what would be cleaned
atc-llm repo-clean --dry-run

# Actual cleanup with confirmation
atc-llm repo-clean --confirm

# Clean specific categories
atc-llm repo-clean --confirm --categories old_demos stale_outputs
```

### Memory System Statistics
```bash
# Default memory file location
atc-llm memory-stats

# Custom memory file
atc-llm memory-stats --memory-file path/to/memory.jsonl
```

### Enhanced End-to-End Run
```bash
atc-llm run-e2e \
  --scat-path F:/SCAT_extracted \
  --ownship-limit 1 \
  --vicinity-radius 100 --alt-window 5000 \
  --asof --cdmethod GEOMETRIC --dtlook 600 --tmult 10 \
  --spawn-dynamic --intruders 3 --adaptive-cadence \
  --llm-model llama3.1:8b --confidence-threshold 0.8 \
  --max-diversion-nm 80 \
  --results-dir Output/enhanced_demo \
  --reports-dir reports/enhanced \
  --seed 4242 \
  --enable-visualization \
  --memory-enabled
```

## File Structure

### Memory System
- `Output/llm_memory.jsonl` - Experience library records
- `src/cdr/memory.py` - LLM memory implementation

### OpenAP Integration
- `src/cdr/intruders_openap.py` - Monte-Carlo intruder generator
- `Output/scenarios.json` - Generated intruder scenarios

### Failure Analysis
- `Output/llm_failures.jsonl` - Failure case records
- `reports/enhanced/llm_failures.html` - Failure analysis summary

### Visualization
- `src/cdr/visualization.py` - Pygame-based viewer
- `Output/visualization_*.png` - Screenshot exports

### Navigation
- `src/cdr/nav_utils.py` - Enhanced waypoint validation
- BlueSky DIRECT commands with fallback to heading

## Usage Examples

### Memory-Enhanced Resolution
The LLM client now includes past movements and similar experiences in prompts:

```python
from src.cdr.llm_client import LlamaClient
from pathlib import Path

# Initialize with memory
memory_file = Path("Output/llm_memory.jsonl")
llm_client = LlamaClient(config, memory_file=memory_file)

# Memory is automatically used in detection and resolution prompts
conflicts = llm_client.detect_conflicts(ownship, intruders, config, cpa_hints)
resolution = llm_client.resolve_conflicts(ownship, conflicts, config, intruder_states)
```

### OpenAP Performance Constraints
```python
from src.cdr.intruders_openap import OpenAPIntruderGenerator

generator = OpenAPIntruderGenerator(seed=4242)
intruders = generator.generate_intruders(
    ownship_state, count=3, vicinity_radius_nm=100
)

# Each intruder has performance envelope
for intruder in intruders:
    print(f"{intruder.aircraft_type}: {intruder.performance.max_speed_kt} kt max")
```

### Failure Analysis
```python
from src.cdr.metrics import FailureAnalysisCollector

failure_collector = FailureAnalysisCollector("Output/failures.jsonl")

# Record failure when LLM misses conflict
failure_collector.record_failure(
    ownship_state, intruder_states, detection_prompt, detection_output,
    resolution_prompt, resolution_output, command_issued,
    cpa_before, cpa_after, bluesky_cd_flags, "missed_conflict"
)

# Export analysis
failure_collector.export_summary("reports/failure_analysis.json")
```

### Visualization
```python
from src.cdr.visualization import VisualizationManager, VisualizationConfig

config = VisualizationConfig()
viz = VisualizationManager(config, enable_pygame=True)

# Update scenario data
viz.update_scenario_data(aircraft_states, conflicts, waypoints)

# Add Direct-To command
viz.add_direct_to_command("OWNSHIP", "FIX_A", 51.5, -0.1)

# Render frame
viz.render_frame()
```

## Dependencies

### Required
- `numpy` - For memory system vector operations
- `pydantic>=2.0` - For data validation

### Optional
- `pygame` - For visualization (fallback if not available)
- `openap` - For aircraft performance data (fallback data included)

### Installation
```bash
# Install core dependencies
pip install numpy pydantic

# Install optional dependencies
pip install pygame openap

# Or install all at once
pip install -r requirements.txt
```

## Performance Considerations

### Memory System
- Default maximum: 10,000 experience records
- Automatic cleanup of oldest records when limit reached
- Vector similarity search with cosine distance

### OpenAP Integration
- Performance envelope caching for repeated lookups
- Fallback to default performance data if OpenAP unavailable
- Type-specific constraints for realistic behavior

### Visualization
- 30 FPS rendering with configurable zoom levels
- Off-screen culling for performance
- Conflict cylinder display with 5 NM separation standards

## Troubleshooting

### Memory System Issues
- Check NumPy installation: `python -c "import numpy; print('OK')"`
- Verify memory file permissions in Output directory
- Use `atc-llm memory-stats` to check memory file status

### Visualization Issues
- Check Pygame installation: `python -c "import pygame; print('OK')"`
- Run with `--enable-visualization` flag in run-e2e command
- Screenshots saved automatically on exit

### OpenAP Issues
- OpenAP library is optional - fallback data is used if unavailable
- Check aircraft type availability with OpenAP if installed
- Performance constraints logged in verbose mode

### Cleanup Issues
- Always run `atc-llm repo-clean --dry-run` first
- Preserved files listed in repo_cleanup.py
- Use `--categories` to clean specific file types only

## Integration with Existing Code

All enhanced features are designed for backward compatibility:

1. **Memory system** - Optional, gracefully disabled if NumPy unavailable
2. **OpenAP integration** - Fallback performance data if library not installed
3. **Visualization** - Optional, runs headless if Pygame unavailable
4. **Failure analysis** - Always enabled but non-intrusive
5. **Enhanced navigation** - Backward compatible with existing waypoint logic

The existing CLI commands and API remain unchanged while new features add capabilities.