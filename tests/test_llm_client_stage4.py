"""Stage 4 — LLM client prompt & parser branches.

Covers:
- Prompt build: presence of JSON schema hints and nearest_fixes inclusion.
- JSON extraction: plain, wrapped, malformed triggers fallback.
- HTTP: happy path, timeout, non-200, partial JSON in response string.
- Mock mode: LLM_DISABLED=1 deterministic behavior and validation shape.
- Paths: generate_resolution and resolve_conflict success and failure.
- No network usage: all requests are monkeypatched with dummies.
"""

from __future__ import annotations

import os
import types
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, cast
import pytest

from src.cdr.llm_client import LlamaClient
from src.cdr.schemas import (
    AircraftState,
    ConflictPrediction,
    LLMDetectionInput,
    ConfigurationSettings,
)


@pytest.fixture
def client():
    # Default client with mock disabled by env unless test overrides
    # Keep deterministic model/host values implied by class defaults
    return LlamaClient()


def make_aircraft(idx: str = "OWN", heading: float = 90.0) -> AircraftState:
    return AircraftState(
        aircraft_id=idx,
        timestamp=datetime.now(),
        latitude=59.30,
        longitude=18.10,
        altitude_ft=35000,
        ground_speed_kt=450,
        heading_deg=heading,
    vertical_speed_fpm=0,
    spawn_offset_min=0.0,
    )


def make_conflict(own: str = "OWN", intr: str = "TFC001") -> ConflictPrediction:
    return ConflictPrediction(
        ownship_id=own,
        intruder_id=intr,
        time_to_cpa_min=4.5,
        distance_at_cpa_nm=3.0,
        altitude_diff_ft=500.0,
        is_conflict=True,
        severity_score=0.8,
        conflict_type="both",
        prediction_time=datetime.now(),
        confidence=1.0,
    )


def make_config_full() -> ConfigurationSettings:
    return ConfigurationSettings(
        polling_interval_min=1.0,
        lookahead_time_min=10.0,
        snapshot_interval_min=1.0,
        max_intruders_in_prompt=5,
        intruder_proximity_nm=100.0,
        intruder_altitude_diff_ft=5000.0,
        trend_analysis_window_min=2.0,
        min_horizontal_separation_nm=5.0,
        min_vertical_separation_ft=1000.0,
        llm_enabled=True,
        llm_model_name="llama3.1:8b",
        llm_temperature=0.1,
        llm_max_tokens=512,
        safety_buffer_factor=1.2,
        max_resolution_angle_deg=30.0,
        max_altitude_change_ft=2000.0,
        enforce_ownship_only=True,
        max_climb_rate_fpm=3000.0,
        max_descent_rate_fpm=3000.0,
        min_flight_level=100,
        max_flight_level=600,
        max_heading_change_deg=90.0,
        enable_dual_llm=True,
        horizontal_retry_count=2,
        vertical_retry_count=2,
    bluesky_host="127.0.0.1",
        bluesky_port=5555,
        bluesky_timeout_sec=5.0,
    max_waypoint_diversion_nm=80.0,
        fast_time=True,
        sim_accel_factor=1.0,
    )


class DummyResp:
    def __init__(self, status_code: int = 200, payload: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self._payload: Dict[str, Any] = payload or {"response": "{}"}

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class DummyRequests:
    class RequestException(Exception):
        pass

    class exceptions:  # noqa: N801 (match requests.exceptions)
        class Timeout(Exception):
            pass

    def __init__(self):
        self.calls: List[Tuple[str, Dict[str, Any], float]] = []

    def post(self, url: str, json: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None):  # noqa: A003
        self.calls.append((url, json or {}, float(timeout or 30)))
        return DummyResp(200, {"response": "{}"})


def install_dummy_requests(monkeypatch: Any, payload_text: Optional[str] = None, status: int = 200, raise_timeout: bool = False):
    """Install a dummy requests module into llm_client namespace and set REQUESTS_AVAILABLE.

    Optionally tweak return payload/behaviour per test.
    """
    import src.cdr.llm_client as llm

    dummy = DummyRequests()

    def fake_post(url: str, json: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> DummyResp:  # noqa: ANN001
        if raise_timeout:
            raise dummy.exceptions.Timeout("timeout")
        text = payload_text if payload_text is not None else "{}"
        return DummyResp(status, {"response": text})

    monkeypatch.setattr(llm, "requests", types.SimpleNamespace(
        post=fake_post,
        RequestException=DummyRequests.RequestException,
        exceptions=DummyRequests.exceptions,
    ), raising=True)
    monkeypatch.setattr(llm, "REQUESTS_AVAILABLE", True, raising=True)
    # Ensure not disabled in env for HTTP-path tests
    monkeypatch.setenv("LLM_DISABLED", "0")


def test_build_enhanced_resolve_prompt_includes_schema_and_fixes(monkeypatch: Any, client: LlamaClient) -> None:
    # Patch nearest_fixes to ensure names appear
    # no-op but keeps importers isolated per test
    monkeypatch.setenv("LLM_DISABLED", "1")  # irrelevant here, just isolation

    # Monkeypatch function used internally
    import src.cdr.nav_utils as nav

    def fake_nearest_fixes(lat: float, lon: float, k: int = 3, max_dist_nm: float = 80.0) -> List[Dict[str, Any]]:
        return [
            {"name": "FIXA", "dist_nm": 10.0},
            {"name": "FIXB", "dist_nm": 20.0},
        ]

    monkeypatch.setattr(nav, "nearest_fixes", fake_nearest_fixes)

    own = make_aircraft("OWN", 135)
    conflicts: List[Dict[str, Any]] = [
        {"intruder_id": "TFC001", "time_to_cpa_min": 5.0, "distance_at_cpa_nm": 3.0, "altitude_diff_ft": 500.0}
    ]
    prompt = client.build_enhanced_resolve_prompt(own, conflicts, {"max_resolution_angle_deg": 30})  # type: ignore

    assert "OUTPUT FORMAT (JSON only" in prompt
    assert "Nearby fixes you may choose" in prompt
    assert "FIXA" in prompt and "FIXB" in prompt


def test_build_detect_prompt_contains_schema(client: LlamaClient) -> None:
    prompt = client.build_detect_prompt({"aircraft_id": "OWN"}, [], {"lookahead_time_min": 10})
    assert "Return strict JSON" in prompt or "Only JSON" in prompt
    assert "Ownship:" in prompt and "Intruders:" in prompt


def test_extract_json_variants(client: LlamaClient) -> None:
    # Plain
    assert client.extract_json_from_text("{\"a\":1}") == {"a": 1}
    # Wrapped text
    mixed = "prefix text {\"conflict\": true, \"reason\": \"ok\"} suffix"
    out = client.extract_json_from_text(mixed)
    assert out.get("conflict") is True and out.get("reason") == "ok"
    # Malformed -> {}
    assert client.extract_json_from_text("{bad json: 1,") == {}


def test_parse_enhanced_detection_fallback_on_malformed(client: LlamaClient) -> None:
    fb = client.parse_enhanced_detection_response("<<< nonsense >>>")
    assert fb.get("conflict") is False and fb.get("intruders") == []
    assert "Fallback" in fb.get("reason", "")


def test_http_happy_path_via_generate_resolution(monkeypatch: Any) -> None:
    payload = "Here: {\n  \"action\": \"turn\", \n  \"params\": {\"heading_delta_deg\": 10}, \n  \"rationale\": \"ok\"\n} end"
    install_dummy_requests(monkeypatch, payload_text=payload)
    monkeypatch.setenv("LLM_DISABLED", "0")
    # Create client after env patching to ensure HTTP path
    client = LlamaClient()
    monkeypatch.setattr(client, "use_mock", False, raising=True)
    detect_out: Dict[str, Any] = {
        "ownship": {"aircraft_id": "OWN", "hdg_deg": 100, "lat": 1, "lon": 1, "alt_ft": 30000, "spd_kt": 400},
        "conflicts": [{"intruder_id": "T1", "time_to_cpa_min": 5.0, "distance_at_cpa_nm": 3.0, "altitude_diff_ft": 500.0}],
    }
    out = client.generate_resolution(detect_out, {"max_resolution_angle_deg": 30})  # type: ignore
    data = out if isinstance(out, dict) else out.__dict__
    assert data["action"] == "turn"
    assert data["params"]["heading_delta_deg"] == 10


def test_http_timeout_via_generate_resolution(monkeypatch: Any) -> None:
    install_dummy_requests(monkeypatch, raise_timeout=True)
    monkeypatch.setenv("LLM_DISABLED", "0")
    client = LlamaClient()
    monkeypatch.setattr(client, "use_mock", False, raising=True)
    detect_out: Dict[str, Any] = {
        "ownship": {"aircraft_id": "OWN", "hdg_deg": 100, "lat": 1, "lon": 1, "alt_ft": 30000, "spd_kt": 400},
        "conflicts": [{"intruder_id": "T1", "time_to_cpa_min": 5.0, "distance_at_cpa_nm": 3.0, "altitude_diff_ft": 500.0}],
    }
    with pytest.raises(RuntimeError):
        client.generate_resolution(detect_out, {"max_resolution_angle_deg": 30})  # type: ignore


def test_http_non_200_via_generate_resolution(monkeypatch: Any) -> None:
    install_dummy_requests(monkeypatch, payload_text="{}", status=500)
    monkeypatch.setenv("LLM_DISABLED", "0")
    client = LlamaClient()
    monkeypatch.setattr(client, "use_mock", False, raising=True)
    detect_out: Dict[str, Any] = {
        "ownship": {"aircraft_id": "OWN", "hdg_deg": 100, "lat": 1, "lon": 1, "alt_ft": 30000, "spd_kt": 400},
        "conflicts": [{"intruder_id": "T1", "time_to_cpa_min": 5.0, "distance_at_cpa_nm": 3.0, "altitude_diff_ft": 500.0}],
    }
    with pytest.raises(RuntimeError):
        client.generate_resolution(detect_out, {"max_resolution_angle_deg": 30})  # type: ignore


def test_call_json_extracts_from_mixed_text(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_DISABLED", "0")
    client = LlamaClient()
    # Patch legacy _post to return mixed text with valid JSON inside
    def _fake_post(prompt: str) -> str:
        return "prefix {\"x\": 1} suffix"
    monkeypatch.setattr(client, "_post", _fake_post, raising=True)
    out = client.call_json("any prompt")
    assert out.get("x") == 1


def test_generate_resolution_success_mock_and_failure_http(monkeypatch: Any) -> None:
    # Success via mock mode
    monkeypatch.setenv("LLM_DISABLED", "1")
    client = LlamaClient()
    detect_out: Dict[str, Any] = {
        "ownship": {"aircraft_id": "OWN", "hdg_deg": 100, "lat": 1, "lon": 1, "alt_ft": 30000, "spd_kt": 400},
        "conflicts": [{"intruder_id": "T1", "time_to_cpa_min": 5.0, "distance_at_cpa_nm": 3.0, "altitude_diff_ft": 500.0}],
    }
    res = client.generate_resolution(detect_out, {"max_resolution_angle_deg": 30})  # type: ignore
    # Mock path returns an object via _create_mock_result
    assert hasattr(res, "action") and hasattr(res, "params")

    # Failure via HTTP empty -> raises RuntimeError
    monkeypatch.setenv("LLM_DISABLED", "0")
    client2 = LlamaClient()

    # Force REQUESTS path and empty dict back from _post_ollama
    import src.cdr.llm_client as llm
    monkeypatch.setattr(llm, "REQUESTS_AVAILABLE", True, raising=True)
    # Replace _post_ollama to ensure it returns {}
    def _fake_post_ollama(prompt: str) -> Dict[str, Any]:
        return {}
    monkeypatch.setattr(client2, "_post_ollama", _fake_post_ollama, raising=True)

    with pytest.raises(RuntimeError):
        client2.generate_resolution(detect_out, {"max_resolution_angle_deg": 30})  # type: ignore


def test_resolve_conflict_success_and_fallback(monkeypatch: Any) -> None:
    own = make_aircraft("OWN", 180)
    intr = make_aircraft("TFC001", 270)
    conf = make_conflict()

    # Success via mock mode (returns SimpleNamespace)
    monkeypatch.setenv("LLM_DISABLED", "1")
    client = LlamaClient()
    result = client.resolve_conflict(conf, [own, intr], make_config_full())  # type: ignore
    result_any = cast(Any, result)
    assert result_any.action
    assert result_any.target_aircraft == "OWN"

    # Force failure path: generate_resolution error -> fallback SimpleNamespace
    monkeypatch.setenv("LLM_DISABLED", "0")
    client2 = LlamaClient()
    def _raise(*args: Any, **kwargs: Any):
        raise Exception("boom")
    monkeypatch.setattr(client2, "generate_resolution", _raise, raising=True)
    fb = client2.resolve_conflict(conf, [own, intr], make_config_full())  # type: ignore
    fb_any = cast(Any, fb)
    assert fb_any.action == "HEADING_CHANGE"
    assert fb_any.params.get("heading_delta_deg") == 20


def test_mock_mode_detect_conflicts_deterministic(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_DISABLED", "1")
    cfg = make_config_full()
    client = LlamaClient(cfg)

    own = make_aircraft("OWN")
    # No intruders -> conflict False
    det_in = LLMDetectionInput(ownship=own, traffic=[], lookahead_minutes=10.0, current_time=datetime.now())
    out = client.detect_conflicts(det_in)  # type: ignore
    assert hasattr(out, "assessment")
    assert getattr(out, "conflict", False) is False

    # One intruder -> conflict True
    intr = make_aircraft("I1")
    det_in2 = LLMDetectionInput(ownship=own, traffic=[intr], lookahead_minutes=10.0, current_time=datetime.now())
    out2 = client.detect_conflicts(det_in2)  # type: ignore
    assert getattr(out2, "conflict", False) is True


def test_no_network_in_mock_mode(monkeypatch: Any) -> None:
    # Install a requests that would raise if used
    import src.cdr.llm_client as llm
    def boom(*a: Any, **k: Any):  # noqa: ANN001, D401
        raise AssertionError("Network call attempted")

    monkeypatch.setattr(llm, "requests", types.SimpleNamespace(
        post=boom,
        RequestException=Exception,
        exceptions=types.SimpleNamespace(Timeout=Exception),
    ), raising=True)
    monkeypatch.setattr(llm, "REQUESTS_AVAILABLE", True, raising=True)
    monkeypatch.setenv("LLM_DISABLED", "1")

    client = LlamaClient()
    own = make_aircraft("OWN")
    det_in = LLMDetectionInput(ownship=own, traffic=[], lookahead_minutes=10.0, current_time=datetime.now())
    _ = client.detect_conflicts(det_in)  # type: ignore
    # If no AssertionError, then network was not used


def test_call_json_mock_routes(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_DISABLED", "1")
    client = LlamaClient()
    d1 = client.call_json("Detect conflicts for scenario")
    r1 = client.call_json("Please resolve the situation")
    d2 = client.call_json("generic prompt")
    assert isinstance(d1, dict) and isinstance(r1, dict) and isinstance(d2, dict)
    assert d1.get("horizon_min") is not None
    assert r1.get("action") in {"turn", "climb", "descend"}


def test_resolution_parser_sanitizes_bluesky_command(client: LlamaClient) -> None:
    # Verify public parser normalizes a non-standard heading command
    txt = '{"action":"HEADING_CHANGE","params":{},"rationale":"x","bluesky_command":"OWN HEADING 370"}'
    out = client.parse_enhanced_resolution_response(txt)
    assert out["bluesky_command"].startswith("OWN HDG ")
    assert out["bluesky_command"].endswith("359")


def test_validate_llm_connection_success_and_failure(monkeypatch: Any) -> None:
    client = LlamaClient()
    # Success
    def _ok_post_ollama(p: str) -> Dict[str, Any]:
        return {"status": "connected"}
    monkeypatch.setattr(client, "_post_ollama", _ok_post_ollama, raising=True)
    assert client.validate_llm_connection() is True
    # Failure
    def _empty_post_ollama(p: str) -> Dict[str, Any]:
        return {}
    def _empty_post_ollama_local(p: str) -> Dict[str, Any]:
        return {}
    monkeypatch.setattr(client, "_post_ollama", _empty_post_ollama_local, raising=True)
    assert client.validate_llm_connection() is False


def test_parse_enhanced_resolution_defaults_and_invalid_action(client: LlamaClient) -> None:
    txt = '{"action":"TURN","params":{},"rationale":"r"}'
    out = client.parse_enhanced_resolution_response(txt)
    # Invalid action coerced to HEADING_CHANGE and defaults present
    assert out["action"] == "HEADING_CHANGE"
    assert "confidence" in out and "backup_actions" in out


def test_extract_json_code_fences(client: LlamaClient) -> None:
    fenced = "```json\n{\n \"a\": 2\n}\n```"
    out = client.extract_json_from_text(fenced)
    assert out.get("a") == 2
    fenced2 = "```\n{\n \"b\": 3\n}\n```"
    out2 = client.extract_json_from_text(fenced2)
    assert out2.get("b") == 3


def test_parse_detect_and_resolve_response_on_strings(client: LlamaClient) -> None:
    det = client.parse_detect_response('{"conflict": true, "intruders": ["A1"], "horizon_min": 12, "reason": "x"}')
    assert det["conflict"] is True and det["intruders"] == ["A1"]
    res = client.parse_resolve_response('{"action": "climb", "params": {"altitude_change_ft": 1000}, "ratio": 0.9, "reason": "ok"}')
    assert res["action"] == "climb" and res["params"]["altitude_change_ft"] == 1000


def test_detect_conflicts_http_enhanced_success_and_none(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_DISABLED", "0")
    client = LlamaClient()
    monkeypatch.setattr(client, "use_mock", False, raising=True)

    # Minimal schema-like wrapper with model_dump expected by detect_conflicts
    class MockDetectInput:
        def __init__(self) -> None:
            self.ownship: Dict[str, Any] = {"aircraft_id": "OWN", "lat": 1, "lon": 1, "alt_ft": 30000, "hdg_deg": 90, "spd_kt": 400}
            self.traffic: List[Dict[str, Any]] = [{"aircraft_id": "I1"}]
            self.lookahead_minutes = 10
            self.current_time = datetime.now()

        def model_dump(self, mode: Optional[str] = None) -> Dict[str, Any]:
            return {"ownship": self.ownship, "traffic": self.traffic, "lookahead_minutes": self.lookahead_minutes, "current_time": self.current_time}

    # Success path via _call_llm
    def _good_call_llm(p: str) -> str:
        return '{"conflict": true, "intruders": ["I1"], "horizon_min": 10, "reason": "ok"}'
    monkeypatch.setattr(client, "_call_llm", _good_call_llm, raising=True)
    out = client.detect_conflicts(MockDetectInput())  # type: ignore[arg-type]
    assert hasattr(out, "conflict") and getattr(out, "conflict") is True

    # Invalid JSON -> None path
    def _bad_call_llm(p: str) -> str:
        return 'not json'
    monkeypatch.setattr(client, "_call_llm", _bad_call_llm, raising=True)
    # Ensure _post_ollama returns {} to trigger None path
    def _empty_post_ollama_none(p: str) -> Dict[str, Any]:
        return {}
    monkeypatch.setattr(client, "_post_ollama", _empty_post_ollama_none, raising=True)
    out2 = client.detect_conflicts(MockDetectInput())  # type: ignore[arg-type]
    assert out2 is None


def test_call_json_retry_failure(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_DISABLED", "0")
    client = LlamaClient()
    monkeypatch.setattr(client, "use_mock", False, raising=True)
    # Always raise in _post; with retries=0 should raise RuntimeError
    def boom(prompt: str) -> str:
        raise Exception("net down")
    monkeypatch.setattr(client, "_post", boom, raising=True)
    with pytest.raises(RuntimeError):
        client.call_json("please json", retries=0)


def test_properties_and_wrapper_class() -> None:
    client = LlamaClient()
    # Properties on LlamaClient
    assert client.llama_client is client and client.llm_client is client
    # Wrapper class delegates
    from src.cdr.llm_client import LLMClient
    wrapper = LLMClient()
    assert isinstance(wrapper.llama_client, LlamaClient)
    _ = wrapper.model
    _ = wrapper.host


def test_extract_json_math_and_fence_and_detect_dict_path(monkeypatch: Any) -> None:
    client = LlamaClient()
    # Math expression evaluation
    mixed = 'xx {"alt": 35000 + 500, "x": 1} yy'
    out = client.extract_json_from_text(mixed)
    assert out.get("alt") == 35500
    # Detect with dict input hitting _post_ollama fallback -> mock detect
    monkeypatch.setenv("LLM_DISABLED", "0")
    monkeypatch.setattr(client, "use_mock", False, raising=True)
    # Make _post_ollama return {} to trigger fallback path
    def _empty_post_ollama2(p: str) -> Dict[str, Any]:
        return {}
    monkeypatch.setattr(client, "_post_ollama", _empty_post_ollama2, raising=True)
    det = client.detect_conflicts({"ownship": {}}, use_enhanced=False)  # type: ignore[arg-type]
    assert isinstance(det, dict) or hasattr(det, "conflict")


def test_generate_resolution_legacy_string_mode(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_DISABLED", "0")
    client = LlamaClient()
    monkeypatch.setattr(client, "use_mock", False, raising=True)
    # Patch _post_ollama to return something empty so legacy path uses mock fallback
    def _empty_post_ollama3(p: str) -> Dict[str, Any]:
        return {}
    monkeypatch.setattr(client, "_post_ollama", _empty_post_ollama3, raising=True)
    out = client.generate_resolution("legacy input", {"max_resolution_angle_deg": 30})  # type: ignore[arg-type]
    # Legacy path returns dict
    assert isinstance(out, dict) and "action" in out


def test_call_json_uses_legacy_post_when_requests_unavailable(monkeypatch: Any) -> None:
    import src.cdr.llm_client as llm
    monkeypatch.setenv("LLM_DISABLED", "0")
    client = LlamaClient()
    monkeypatch.setattr(client, "use_mock", False, raising=True)
    # Simulate requests not installed
    monkeypatch.setattr(llm, "REQUESTS_AVAILABLE", False, raising=True)
    res = client.call_json("resolution please")
    assert isinstance(res, dict)


def test_ask_detect_and_ask_resolve_wrappers(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_DISABLED", "1")
    client = LlamaClient()
    det = client.ask_detect('{"ownship": {"aircraft_id":"X"}, "traffic": []}')
    # Should be ResolveOut/DetectOut models if available; assert attributes generically
    assert hasattr(det, "conflict")
    res = client.ask_resolve("data", {"x": 1})
    assert res is None or hasattr(res, "action") or isinstance(res, dict)


def test_build_enhanced_detect_prompt_with_objects() -> None:
    own = make_aircraft("OWN", 90)
    cfg = make_config_full()
    client = LlamaClient()
    prompt = client.build_enhanced_detect_prompt(own, [make_aircraft("I1", 180)], cfg)
    assert "OUTPUT FORMAT (JSON only" in prompt and "Conflict threshold" in prompt


def test_enhanced_resolution_parser_sanitizes_alt_spd_dct(client: LlamaClient) -> None:
    alt_txt = '{"action":"ALTITUDE_CHANGE","params":{},"rationale":"x","bluesky_command":"own altitude 700"}'
    spd_txt = '{"action":"SPEED_CHANGE","params":{},"rationale":"x","bluesky_command":"own speed 700"}'
    dct_txt = '{"action":"DIRECT_TO_WAYPOINT","params":{},"rationale":"x","bluesky_command":"own direct ABCD"}'
    out_alt = client.parse_enhanced_resolution_response(alt_txt)
    out_spd = client.parse_enhanced_resolution_response(spd_txt)
    out_dct = client.parse_enhanced_resolution_response(dct_txt)
    assert out_alt["bluesky_command"].startswith("OWN ALT ")
    assert out_spd["bluesky_command"].startswith("OWN SPD ")
    # Current sanitizer leaves DIRECT as is
    assert out_dct["bluesky_command"].startswith("OWN DIRECT ")


def test_call_llm_subprocess_and_http_paths(monkeypatch: Any) -> None:
    client = LlamaClient()
    # Subprocess success
    class R:  # simple return object
        def __init__(self, rc: int, out: str):
            self.returncode = rc
            self.stdout = out
    monkeypatch.setenv("LLM_DISABLED", "0")
    def _run_ok(*a: Any, **k: Any):
        return R(0, '{"conflict": true}')
    monkeypatch.setattr("src.cdr.llm_client.subprocess.run", _run_ok, raising=True)
    # Use public detect path that internally calls _call_llm
    class DI:
        def __init__(self) -> None:
            self.ownship: Dict[str, Any] = {}
            self.traffic: List[Dict[str, Any]] = []
            self.lookahead_minutes = 10
            self.current_time = datetime.now()
        def model_dump(self, mode: Optional[str] = None) -> Dict[str, Any]:
            return {"ownship": self.ownship, "traffic": self.traffic, "lookahead_minutes": self.lookahead_minutes, "current_time": self.current_time}
    s = client.detect_conflicts(DI())  # type: ignore[arg-type]
    assert hasattr(s, "conflict") or (isinstance(s, dict) and "conflict" in s)
    # FileNotFoundError path
    def _raise_fnf(*a: Any, **k: Any):
        raise FileNotFoundError()
    monkeypatch.setattr("src.cdr.llm_client.subprocess.run", _raise_fnf, raising=True)
    s2 = client.detect_conflicts(DI())  # type: ignore[arg-type]
    assert hasattr(s2, "conflict") or (isinstance(s2, dict) and "conflict" in s2)
    # HTTP fallback path when subprocess returns non-zero
    def _bad_run(*a: Any, **k: Any):
        return R(1, "")
    monkeypatch.setattr("src.cdr.llm_client.subprocess.run", _bad_run, raising=True)
    # Install dummy requests
    import src.cdr.llm_client as llm
    def _ok_post(url: str, json: Dict[str, Any] | None = None, timeout: float | None = None):
        class D:
            def json(self):
                return {"response": '{"status":"ok"}'}
        return D()
    monkeypatch.setattr(llm, "requests", types.SimpleNamespace(post=_ok_post), raising=True)
    monkeypatch.setattr(llm, "REQUESTS_AVAILABLE", True, raising=True)
    s3 = client.detect_conflicts(DI())  # type: ignore[arg-type]
    # HTTP path returns text, detect_conflicts parses into object/dict
    assert s3 is not None


def test_post_ollama_branches(monkeypatch: Any) -> None:
    client = LlamaClient()
    # Mock mode
    monkeypatch.setenv("LLM_DISABLED", "1")
    # Mock mode path: call_json goes via detect mock
    out = client.call_json("detect something")
    assert isinstance(out, dict)
    # Non-200 returns {}
    monkeypatch.setenv("LLM_DISABLED", "0")
    install_dummy_requests(monkeypatch, payload_text="{}", status=500)
    # Non-200 simulated via generate_resolution which calls _post_ollama
    monkeypatch.setattr(client, "use_mock", False, raising=True)
    with pytest.raises(RuntimeError):
        client.generate_resolution({"ownship": {}, "conflicts": []}, {"max_resolution_angle_deg": 30})  # type: ignore[arg-type]


def test_parse_enhanced_detection_filters_conflicts(client: LlamaClient) -> None:
    import json as _json
    txt = _json.dumps({
        "conflict": True,
        "intruders": ["I1"],
        "horizon_min": 10,
        "reason": "x",
        "conflicts_detected": [
            {"intruder_id": "I1", "time_to_cpa_min": 5, "distance_at_cpa_nm": 3, "altitude_diff_ft": 500},
            {"intruder_id": "I2", "time_to_cpa_min": -1, "distance_at_cpa_nm": 999, "altitude_diff_ft": 20000}
        ]
    })
    out = client.parse_enhanced_detection_response(txt)
    assert len(out.get("conflicts_detected", [])) == 1


def test_call_json_retry_success(monkeypatch: Any) -> None:
    client = LlamaClient()
    monkeypatch.setenv("LLM_DISABLED", "0")
    monkeypatch.setattr(client, "use_mock", False, raising=True)
    # First call raises, retry returns JSON based on prompt content
    def _post(prompt: str) -> str:
        if "ONLY valid JSON" in prompt:
            return "{\"ok\":1}"
        raise Exception("fail")
    monkeypatch.setattr(client, "_post", _post, raising=True)
    out = client.call_json("x", retries=1)
    assert out.get("ok") == 1


def test_wrapper_delegation(monkeypatch: Any) -> None:
    from src.cdr.llm_client import LLMClient
    w = LLMClient()
    prompt = w.build_detect_prompt({"aircraft_id": "OWN"}, [], {"lookahead_time_min": 5})  # type: ignore[arg-type]
    assert "Only JSON" in prompt or "Return strict JSON" in prompt


def test_parse_resolve_response_invalid_action_defaults(client: LlamaClient) -> None:
    out = client.parse_resolve_response('{"action": "foo", "params": {}, "ratio": 0.5, "reason": "x"}')
    assert out["action"] == "turn"
