# LLM-BlueSky Conflict Detection & Resolution

An LLM-driven conflict detection and resolution system built on top of BlueSky air traffic simulator using SCAT trajectories.

## Overview

This system implements a real-time conflict detection and resolution loop that:
- Polls every 5 simulation minutes
- Predicts conflicts 10 minutes ahead
- Issues horizontal or vertical resolutions for ownship only
- Uses Llama 3.1 8B for detection and resolution reasoning
- Benchmarks against BlueSky baseline with Wolfgang (2011) KPIs

## Key Features

- **Safety-First**: Hard validation before executing any LLM commands (minimum separation ≥ 5 NM / 1000 ft)
- **Modular Design**: Clean separation of concerns with typed Python interfaces
- **Reproducible**: Fixed random seeds and versioned dependencies
- **Test-Driven**: Comprehensive unit and integration test coverage
- **CI/CD Ready**: Automated testing and validation pipeline

## Architecture

```
src/cdr/          # Core conflict detection & resolution
├─ geodesy.py     # Haversine, CPA, cross-track calculations
├─ detect.py      # 10-minute lookahead conflict prediction
├─ resolve.py     # Horizontal/vertical resolution algorithms
├─ llm_client.py  # Local Llama 3.1 8B integration
├─ schemas.py     # Pydantic validation models
├─ pipeline.py    # Main 5-minute polling loop
├─ bluesky_io.py  # BlueSky simulator interface
└─ metrics.py     # Wolfgang (2011) KPI calculations
```

## Setup

### Prerequisites
- Python 3.11+
- BlueSky simulator
- Llama 3.1 8B model (local)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd llm-bluesky-cdr

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Start development server (optional)
uvicorn src.api.service:app --reload
```

### Running Simulations

```bash
# Basic conflict detection test
python -m src.cdr.pipeline --scenario scenarios/sample_scat_ownship.json

# With BlueSky integration
python -m src.cdr.pipeline --bluesky --scenario scenarios/blue_sky_init.txt
```

## KPIs & Metrics

Following Wolfgang (2011) standards:
- **TBAS**: Time-Based Alerting Score
- **LAT**: Loss of Alerting Time  
- **PA**: Predicted Alerts
- **PI**: Predicted Intrusions
- **DAT**: Delay in Alert Time
- **DFA**: Delay in First Alert
- **RE**: Resolution Efficiency
- **RI**: Resolution Intrusiveness
- **RAT**: Resolution Alert Time

## Development

### Code Quality
- **Formatting**: Black + Ruff
- **Type Checking**: mypy
- **Testing**: pytest with coverage
- **Documentation**: Comprehensive docstrings

### Sprint Reports
Each development sprint generates artifacts in `reports/sprint_##/` with:
- Performance metrics
- Test coverage reports  
- Visualization plots
- Sprint retrospective

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
