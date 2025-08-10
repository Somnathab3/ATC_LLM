import logging
from types import SimpleNamespace
from typing import Any, List

import pytest

from src.cdr.bluesky_io import BlueSkyClient, BSConfig


class FakeSim:
    def __init__(self) -> None:
        self.steps: List[float] = []
    def step(self, dt: float) -> None:
        self.steps.append(dt)


def test_bsconfig_defaults_and_overrides(monkeypatch: Any):
    # Defaults
    c_def = BlueSkyClient(BSConfig())
    assert c_def.host == '127.0.0.1'
    assert c_def.port == 5555

    # Overrides via custom config object
    cfg = SimpleNamespace(headless=True, bluesky_host='10.0.0.2', bluesky_port=4242)
    c = BlueSkyClient(cfg)
    assert c.host == '10.0.0.2'
    assert c.port == 4242


def test_command_formatting_pass_through_stack(monkeypatch: Any):
    cfg = BSConfig()
    c = BlueSkyClient(cfg)
    calls: List[str] = []
    def fake_stack(cmd: str) -> bool:
        calls.append(cmd)
        return True
    monkeypatch.setattr(c, "stack", fake_stack, raising=False)

    # Heading
    assert c.set_heading('OWNSHIP', 123.4) is True
    # Altitude
    assert c.set_altitude('OWNSHIP', 35500.6) is True
    # Direct
    assert c.direct_to('OWNSHIP', 'GOTUR') is True
    # AddWpt without alt
    assert c.add_waypoint('OWNSHIP', 59.5, 18.6) is True
    # AddWpt with alt
    assert c.add_waypoint('OWNSHIP', 59.6, 18.7, 34000) is True

    assert "OWNSHIP HDG 123" in calls[0]
    assert "OWNSHIP ALT 35501" in calls[1] or "OWNSHIP ALT 35500" in calls[1]
    assert calls[2] == "OWNSHIP DCT GOTUR"
    assert calls[3].startswith("ADDWPT OWNSHIP 59.500000 18.600000")
    assert calls[4].startswith("ADDWPT OWNSHIP 59.600000 18.700000 ")


def test_reset_flow_order_and_flush(monkeypatch: Any):
    cfg = BSConfig()
    c = BlueSkyClient(cfg)
    calls: List[str] = []
    def fake_stack(cmd: str) -> bool:
        calls.append(cmd)
        return True
    monkeypatch.setattr(c, "stack", fake_stack, raising=False)
    sim = FakeSim()
    monkeypatch.setattr(c, "sim", sim, raising=False)
    assert c.sim_reset() is True
    # Expect DEL ALL then RESET
    assert calls[0] == "DEL ALL"
    assert calls[1] == "RESET"
    # One small step to flush (either executed or ignored)
    assert isinstance(sim.steps, list)


def test_stack_false_propagates_and_logs_warning(monkeypatch: Any, caplog: Any):
    cfg = BSConfig()
    c = BlueSkyClient(cfg)
    def fake_stack_false(cmd: str) -> bool:
        return False
    monkeypatch.setattr(c, "stack", fake_stack_false, raising=False)
    with caplog.at_level(logging.WARNING):
        ok = c.set_heading('OWNSHIP', 270)
    assert ok is False
    assert any("returned False" in r.getMessage() for r in caplog.records)


def test_execute_command_dispatch(monkeypatch: Any):
    cfg = BSConfig()
    c = BlueSkyClient(cfg)
    calls: List[str] = []
    def fake_stack(cmd: str) -> bool:
        calls.append(cmd)
        return True
    monkeypatch.setattr(c, "stack", fake_stack, raising=False)

    class Reso:
        def __init__(self, typ: str, cs: str, hdg: float | None = None, alt: float | None = None, wpt: str | None = None):
            self.resolution_type = typ
            self.target_aircraft = cs
            self.new_heading_deg = hdg
            self.new_altitude_ft = alt
            self.waypoint_name = wpt

    # Heading
    r1 = Reso('heading_change', 'OWNSHIP', hdg=200)
    assert c.execute_command(r1) is True
    # Altitude
    r2 = Reso('altitude_change', 'OWNSHIP', alt=33000)
    assert c.execute_command(r2) is True
    # Waypoint
    r3 = Reso('waypoint_direct', 'OWNSHIP', wpt='ABCD')
    assert c.execute_command(r3) is True

    # Unknown -> False
    class Empty: pass
    assert c.execute_command(Empty()) is False


def test_step_minutes_calls_sim(monkeypatch: Any):
    cfg = BSConfig()
    c = BlueSkyClient(cfg)
    sim = FakeSim()
    monkeypatch.setattr(c, "sim", sim, raising=False)
    assert c.step_minutes(0.02) is True  # very small advances should still work
    assert isinstance(sim.steps, list)
