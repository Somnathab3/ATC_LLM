import types
import sys
import os
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timezone

# Set environment variables for isolated testing
os.environ["LLM_DISABLED"] = "1"
os.environ["BLUESKY_HEADLESS"] = "1" 
os.environ["PYTHONHASHSEED"] = "0"

class FakeBlueSky:
    """Fake BlueSky client that doesn't require actual BlueSky connection."""
    
    def __init__(self, config=None):
        self.config = config
        self.connected = True
        self.aircraft_states = []
        self.commands_executed = []
        
        # Mock traffic manager with proper id list
        self.traf = Mock()
        self.traf.id = []  # Empty list for aircraft IDs
        
        # Mock BlueSky stack
        self.bs = Mock()
        self.bs.stack = Mock()
        
        # Mock simulation
        self.sim = Mock()
        self.sim.step = Mock()
    
    def get_aircraft_states(self):
        """Return mock aircraft states."""
        return self.aircraft_states
    
    def execute_command(self, command):
        """Mock command execution."""
        self.commands_executed.append(command)
        return {"success": True, "command": command}
    
    def stack(self, command):
        """Mock BlueSky stack command."""
        self.commands_executed.append(command)
        if self.bs and hasattr(self.bs, 'stack'):
            self.bs.stack(command)
        return True
    
    def connect(self):
        """Mock connection."""
        self.connected = True
        return True
    
    def disconnect(self):
        """Mock disconnection."""
        self.connected = False
    
    def is_connected(self):
        """Check if connected."""
        return self.connected
    
    def add_mock_aircraft(self, aircraft_id, lat=40.0, lon=-74.0, alt=35000):
        """Add a mock aircraft for testing."""
        from src.cdr.schemas import AircraftState
        aircraft = AircraftState(
            aircraft_id=aircraft_id,
            callsign=aircraft_id,
            latitude=lat,
            longitude=lon,
            altitude_ft=alt,
            heading_deg=90,
            ground_speed_kt=450,
            vertical_speed_fpm=0,
            timestamp=datetime.now(timezone.utc)
        )
        self.aircraft_states.append(aircraft)
        # Also add to traf.id for compatibility
        if aircraft_id not in self.traf.id:
            self.traf.id.append(aircraft_id)
        return aircraft


class FakeOllama:
    """Fake Ollama/LLM client that doesn't require actual LLM connection."""
    
    def __init__(self, model_name="test-model", host="localhost", **kwargs):
        self.model_name = model_name
        self.host = host
        self.responses = []
        self.requests_made = []
    
    def post(self, url, json=None, **kwargs):
        """Mock HTTP post request."""
        self.requests_made.append({"url": url, "json": json})
        
        # Return a mock response with a simple resolution
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "HDG TEST123 120\nReason: Turn right to avoid conflict",
            "done": True
        }
        return mock_response
    
    def generate_resolution(self, conflict_data, config=None):
        """Generate a mock resolution."""
        return {
            "command": "HDG TEST123 120",
            "reasoning": "Turn right to avoid conflict",
            "confidence": 0.85
        }
    
    def detect_conflicts(self, input_data, use_enhanced=False):
        """Mock conflict detection."""
        return {
            "conflict": False,
            "intruders": [],
            "mock": True
        }
    
    def get_resolution(self, conflict_data, config=None):
        """Mock resolution generation."""
        return self.generate_resolution(conflict_data, config)


class FakeMetrics:
    """Fake metrics collector that doesn't require actual data processing."""
    
    def __init__(self):
        self.metrics = {}
        self.events = []
    
    def record_conflict(self, conflict):
        """Record a conflict event."""
        self.events.append({"type": "conflict", "data": conflict})
    
    def record_resolution(self, resolution):
        """Record a resolution event."""
        self.events.append({"type": "resolution", "data": resolution})
    
    def get_metrics(self):
        """Return mock metrics."""
        return {
            "conflicts_detected": len([e for e in self.events if e["type"] == "conflict"]),
            "resolutions_issued": len([e for e in self.events if e["type"] == "resolution"]),
            "success_rate": 0.95
        }


class FakeReportingSystem:
    """Fake enhanced reporting system."""
    
    def __init__(self):
        self.reports = []
    
    def add_report(self, report):
        self.reports.append(report)


class FakePromptBuilder:
    """Fake prompt builder."""
    
    def __init__(self, config):
        self.config = config
    
    def build_prompt(self, *args, **kwargs):
        return "Mock prompt"


class FakeResolutionAgent:
    """Fake resolution agent."""
    
    def __init__(self, llm_client, config):
        self.llm_client = llm_client
        self.config = config
    
    def generate_resolution(self, *args, **kwargs):
        return {"command": "HDG TEST123 120", "mock": True}


class FakeValidator:
    """Fake enhanced resolution validator."""
    
    def __init__(self, config):
        self.config = config
    
    def validate(self, resolution):
        return True


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically set up test environment for all tests."""
    # Ensure isolated environment
    monkeypatch.setenv("LLM_DISABLED", "1")
    monkeypatch.setenv("BLUESKY_HEADLESS", "1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    
    # Mock network imports to prevent real connections
    fake_requests = Mock()
    fake_requests.post = FakeOllama().post
    monkeypatch.setattr("requests.post", fake_requests.post)
    
    # Mock BlueSky imports
    try:
        # Try to mock bluesky if it exists
        monkeypatch.setattr("bluesky.sim", Mock())
    except ImportError:
        pass
    
    # Mock all pipeline dependencies
    monkeypatch.setattr("src.cdr.pipeline.MetricsCollector", lambda: FakeMetrics())
    monkeypatch.setattr("src.cdr.pipeline.EnhancedReportingSystem", lambda: FakeReportingSystem())
    monkeypatch.setattr("src.cdr.pipeline.PromptBuilderV2", lambda config: FakePromptBuilder(config))
    monkeypatch.setattr("src.cdr.pipeline.HorizontalResolutionAgent", lambda llm, config: FakeResolutionAgent(llm, config))
    monkeypatch.setattr("src.cdr.pipeline.VerticalResolutionAgent", lambda llm, config: FakeResolutionAgent(llm, config))
    monkeypatch.setattr("src.cdr.pipeline.EnhancedResolutionValidator", lambda config: FakeValidator(config))


@pytest.fixture
def fake_bluesky():
    """Provide a fake BlueSky client for testing."""
    return FakeBlueSky()


@pytest.fixture  
def fake_ollama():
    """Provide a fake Ollama client for testing."""
    return FakeOllama()


@pytest.fixture
def mock_bluesky_client(monkeypatch):
    """Replace BlueSkyClient with fake implementation."""
    def fake_bluesky_factory(config):
        fake = FakeBlueSky(config)
        return fake
    
    monkeypatch.setattr("src.cdr.bluesky_io.BlueSkyClient", fake_bluesky_factory)
    monkeypatch.setattr("src.cdr.pipeline.BlueSkyClient", fake_bluesky_factory)
    # Return a default instance for direct access in tests
    return FakeBlueSky()


@pytest.fixture
def mock_llm_client(monkeypatch):
    """Replace LLM clients with fake implementation."""
    def fake_llm_factory(*args, **kwargs):
        fake = FakeOllama()
        return fake
    
    monkeypatch.setattr("src.cdr.llm_client.LlamaClient", fake_llm_factory)
    monkeypatch.setattr("src.cdr.pipeline.LlamaClient", fake_llm_factory)
    # Return a default instance for direct access in tests
    return FakeOllama()


@pytest.fixture
def dummy_main_factory():
    """
    Returns a factory to create stub 'main' callables that record calls.
    Usage:
        main, calls = dummy_main_factory(returncode=0)
        main(); assert calls["count"] == 1
    """
    def _factory(returncode=0, side_effect=None):
        calls = {"count": 0, "argv_at_call": None}
        def _main():
            calls["count"] += 1
            import sys as _sys
            calls["argv_at_call"] = list(_sys.argv)
            if side_effect:
                side_effect()
            return returncode
        return _main, calls
    return _factory

@pytest.fixture
def inject_module(monkeypatch):
    """
    Insert a dummy module into sys.modules under a given name.
    Returns the module so tests can populate attributes.
    """
    def _inject(name: str):
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod
    return _inject

@pytest.fixture
def tmp_file(tmp_path: Path):
    f = tmp_path / "data.json"
    f.write_text('{"ok": true}')
    return f

@pytest.fixture
def tmp_dir(tmp_path: Path):
    d = tmp_path / "scat_dir"
    d.mkdir()
    return d
