"""End-to-end check that starting a run does not block the HTTP request.

Verifies the contract the browser relies on:
  1. POST /api/run returns quickly with 202, instead of holding the connection
     open for the length of the run.
  2. GET /api/progress reports state transitions and an errors_by_type map.
  3. GET /api/result serves the finished run once progress says "done".

Run against an already-running server:
    python scripts/_e2e_run_lifecycle.py http://localhost:8877 user pass
"""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8877"
USER = sys.argv[2] if len(sys.argv) > 2 else ""
PASSWORD = sys.argv[3] if len(sys.argv) > 3 else ""

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def call(path: str, method: str = "GET", body: dict | None = None):
    """Return (status, parsed_json_or_raw_text)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if PASSWORD:
        token = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
        req.add_header("Authorization", "Basic " + token)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        status = exc.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


print(f"testing {BASE}")

status, health = call("/api/health")
check("/api/health reachable", status == 200, f"http {status}")
if isinstance(health, dict):
    print(f"        provider={health.get('provider')} simulated={health.get('simulated')}")

# 1. Starting a run must return immediately. This is the whole point of the fix:
#    a long synchronous POST is what the platform proxy was killing.
t0 = time.monotonic()
status, started = call(
    "/api/run",
    method="POST",
    body={
        "model_a": "openai-gpt-oss-120b",
        "model_b": "mistral-3-14B",
        "concurrency": 4,
    },
)
elapsed = time.monotonic() - t0
check("POST /api/run accepted", status == 202, f"http {status}")
check("POST /api/run returns promptly", elapsed < 5.0, f"{elapsed:.2f}s")
check(
    "POST /api/run body carries a run_id",
    isinstance(started, dict) and bool(started.get("run_id")),
    json.dumps(started)[:120] if not isinstance(started, dict) else started.get("run_id", ""),
)

# 2. Progress must move and must break down errors by type.
saw_running = False
saw_errors_by_type = False
final: dict = {}
deadline = time.monotonic() + 240
while time.monotonic() < deadline:
    status, prog = call("/api/progress")
    if not isinstance(prog, dict):
        check("/api/progress returns JSON", False, str(prog)[:120])
        break
    if prog.get("state") == "running":
        saw_running = True
    if isinstance(prog.get("errors_by_type"), dict):
        saw_errors_by_type = True
    if prog.get("state") in {"done", "failed"}:
        final = prog
        break
    time.sleep(0.4)

check("run reached a terminal state", final.get("state") in {"done", "failed"}, str(final.get("state")))
check("progress reported a running state", saw_running)
check("progress exposes errors_by_type", saw_errors_by_type)
if final.get("state") == "failed":
    check("run did not fail", False, str(final.get("message"))[:200])

# 3. The finished result must be fetchable separately from the POST.
status, result = call("/api/result")
check("/api/result serves the finished run", status == 200, f"http {status}")
if isinstance(result, dict) and "operational" in result:
    ops = result["operational"]
    print(
        f"        run_id={result.get('run_id')} "
        f"requests={ops.get('total_requests')} "
        f"errors={ops.get('error_count', 'n/a')}"
    )
    check(
        "result run_id matches the started run",
        isinstance(started, dict) and result.get("run_id") == started.get("run_id"),
        f"{result.get('run_id')} vs {started.get('run_id') if isinstance(started, dict) else '?'}",
    )
else:
    check("/api/result has an operational block", False, str(result)[:160])

print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("all lifecycle checks passed")
