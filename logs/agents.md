# Sprint 0 — Inventory & findings

**Prompt to Codex:**

> **Task:** Produce a precise repo-wide health report.
>
> 1. Parse the tree; list modules and their dependency graph; flag dead code or circulars.
> 2. Run: `ruff check`, `black --check`, `mypy`, `pytest -q`. Capture errors/warnings.
> 3. Map tests → modules; highlight **uncovered** safety-critical paths:
>    `geodesy.cpa_nm`, `detect.predict_conflicts`, `resolve._validate_resolution_safety`, `llm_client.generate_resolution`, `scat_adapter.SCATAdapter`.
> 4. Emit `report/healthcheck.json` with: lints, types, test results, coverage per file, and a **ranked buglist** with file\:line and fix sketch.
>    **Output:** Health summary + `DIFF:` if you add tooling, `TESTS:` skeletons for missing areas.

(Modules/roles from your dependency matrix.)&#x20;

---

# Sprint 1 — Static defects & type safety

**Prompt to Codex:**

> **Task:** Fix high-confidence static issues.
>
> * Add/adjust type hints; eliminate `Optional` misuse and float/degree/radian mixups in `geodesy.py`.
> * Add **units docstrings** and `assert`/`pydantic` validators in `schemas.py` for altitudes (ft), speed (kts), headings (deg 0–360).
> * Ensure `detect.is_conflict` uses **absolute altitude separation** and consistent NM vs m calculations; add tests around boundary conditions (4.9/5.0 NM, 950/1000 ft, CPA at 9.9/10.1 min).
> * Make CI fail on lint/type errors.
>   **Output:** `DIFF:` for code + `tests/test_geodesy.py`, `tests/test_detect_thresholds.py`, `tests/test_schemas_validation.py`.

(Conflict logic & math modules are the safety-critical path.)&#x20;

---

# Sprint 2 — LLM hallucination guards

**Prompt to Codex:**

> **Task:** Harden `llm_client.py` + `resolve.py`.
>
> 1. Wrap LLM outputs in **Pydantic models** already defined in `schemas.py`; reject if any of:
>
>    * Missing maneuver type, out-of-range headings/ROCs, or predicted separation < 5 NM/1000 ft.
> 2. Add `ResolutionSafetyGate`: recompute CPA post-LLM **before** executing; if unsafe or non-parsable → call `generate_horizontal_resolution()` / `generate_vertical_resolution()` fallback.
> 3. Log `safety_gate_rejections` and `fallback_reason`.
> 4. Tests:
>
>    * **Adversarial LLM output** (nonsense JSON, unsafe turn, altitude bust) → gate rejects + fallback path used.
>    * **Valid LLM output** passes and improves severity score.
>      **Output:** `DIFF:` + `tests/test_llm_safety_gate.py`, `tests/test_resolve_fallback.py`.

(LLM I/O schemas and resolution safety hooks are part of your core modules.)&#x20;

---

# Sprint 3 — Pipeline timing guarantees

**Prompt to Codex:**

> **Task:** Verify and enforce timing in `pipeline.py`.
>
> * Ensure **5-minute** cycles with tolerance ±5 s; mock clock in tests.
> * Ensure **10-minute look-ahead** is actually passed into detection and used in CPA; add regression test.
> * Add idempotent start/stop and a `/health` FastAPI check returning cycle, conflicts, and uptime.
>   **Output:** `DIFF:` + `tests/test_pipeline_timing.py`, `tests/test_api_health.py`.

(Cycle & API endpoints are defined in your docs.) &#x20;

---

# Sprint 4 — SCAT ingestion & validation

**Prompt to Codex:**

> **Task:** Add robust SCAT data loader validation.
>
> * In `scat_adapter.py`, implement a schema check for ASTERIX-062-like fields; log and skip malformed records; provide a summary (count, missing fields, time gaps).
> * CLI: `python -m src.cdr.scat_adapter validate <path-to-scat.json>`
> * Tests with tiny fixtures (good + malformed).
>   **Output:** `DIFF:` + `tests/test_scat_adapter_validation.py`.

(SCAT integration is a project requirement; the MA brief references this dataset explicitly.)&#x20;

> **Note:** Download SCAT from your Mendeley link manually if auth is required; point the validator to a local JSON sample.

---

# Sprint 5 — Metrics & reporting for the project

**Prompt to Codex:**

> **Task:** Ensure KPIs (Wolfgang 2011) compute and render.
>
> * In `metrics.py`, add unit tests covering **TBAS, LAT, DAT, DFA, RE, RI, RAT** happy-path and edge-cases (no conflicts, late alerts).
> * In `reporting.py`, add a simple `Sprint5Reporter` that writes `reports/summary.json` and a PNG chart of KPI trends.
>   **Output:** `DIFF:` + `tests/test_metrics_kpis.py`, `tests/test_reporting_outputs.py` and a demo artifact under `reports/`.

(Metrics layer and reporter are present in your architecture docs.)&#x20;

---

# Continuous Integration (drop-in)

**Prompt to Codex:**

> **Task:** Add `.github/workflows/ci.yml`:
>
> * Steps: setup Python 3.11, install, run `ruff`, `black --check`, `mypy`, `pytest --cov=src --cov-report=xml`.
> * Upload `coverage.xml` and `reports/*.json` as artifacts.
>   **Output:** `DIFF:` with CI file.

(README already positions testing/quality tooling.)&#x20;

---

# Example “fix a bug” micro-prompt (use repeatedly)

> **Task:** Fix the highest-impact bug from `report/healthcheck.json`.
>
> * Show failing test(s) or minimal repro.
> * Provide the **unified diff** patch.
> * Explain the root cause (1–2 lines) and why the fix is safe (bounds, units, time).
> * Add/extend tests to prove the fix and guard against regressions.
> * Re-run `pytest -q`; paste the summary.

---

# Hallucination-focused guardrails (checklist)

* **Strict Pydantic** models for LLM outputs; `regex` for maneuver tokens; bound heading/ROC/rate values.&#x20;
* **Safety gate** re-checks CPA/separation before execution; **fallback** if unsafe.&#x20;
* **Telemetry**: counters for `llm_parse_error`, `llm_unsafe`, `fallback_used`; emit into metrics for project figures.&#x20;
* **Determinism**: set LLM temperature low (e.g., 0.1) and cap tokens; expose via config.&#x20;

---

# Minimal commands to demonstrate end-to-end

```bash
# Run unit + integration tests
pytest -v

# Start API and one demo sim (ownship + intruder)
uvicorn src.api.service:app --port 8000 &
python scripts/complete_scat_llm_simulation.py  # per README

# Generate KPI report
python - <<'PY'
from src.cdr.reporting import Sprint5Reporter
Sprint5Reporter().run(output_dir="reports")
print("Wrote reports/")
PY
```

(Usage aligns with the documented scripts and API layout.)&#x20;

---

# What you’ll hand in

* `reports/summary.json` + KPI plots demonstrating **baseline vs LLM** and counts of **hallucination gate rejections**.&#x20;
* Test coverage report (≥ 85%).&#x20;
* CI logs proving reproducibility.
* Short **Methods** paragraph referencing the MA brief’s objectives (distribution shift, FP/FN, safety margins).&#x20;
