# Test Data

This directory contains test fixtures and sample data for the CDR system test suite.

## Contents

- `sample_aircraft_states.json` - Sample aircraft state data for testing
- `test_conflicts.json` - Sample conflict predictions for validation
- `test_scenarios/` - Small test scenarios for unit tests

## Usage

Test fixtures can be loaded using:

```python
import json
import os

def load_test_data(filename):
    test_data_dir = os.path.dirname(__file__)
    with open(os.path.join(test_data_dir, filename), 'r') as f:
        return json.load(f)

# Example usage
aircraft_states = load_test_data('sample_aircraft_states.json')
```
