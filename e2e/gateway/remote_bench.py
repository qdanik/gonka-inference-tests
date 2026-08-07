"""Run the throughput burst ON the gateway box — `--on-server`.

Every other harness here reaches the gateway through an SSH forward tunnel, which
is fine for correctness work but puts a single multiplexed TCP connection in the
measurement path. Measured on this setup, the tunnel carries 1,000 concurrent
connections without errors but inflates median latency from 0.7 s to 8.0 s and
peaks in connection throughput around 200 in flight. For a token-rate benchmark
that is not an acceptable confound.

So the burst runs on the box, straight against the gateway's loopback, and only
the raw per-request records come home. The split is deliberate:

  remote   collects — send N requests at C concurrency, record status, latency
           and token counts. No analysis, no dependencies beyond `requests`.
  local    analyses — the same `build_report` used by the tunnelled path, so
           both routes produce identical, equally tested reports.

The admin key never appears in argv — a command line is readable in the remote
process list. It goes into a mode-600 file that the collector unlinks as its
first action.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .config import GatewayTarget
from .load import RequestOutcome

# Collector shipped to the box. Deliberately dependency-light and self-contained:
# it must run under whatever Python the box has, with no repo on the far side.
REMOTE_COLLECTOR = '''
import json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
import requests

config_path = sys.argv[1]
with open(config_path) as handle:
    config = json.load(handle)
os.unlink(config_path)  # the admin key must not sit on disk any longer than this
base_url = config["base_url"]
headers = {"Authorization": "Bearer " + config["admin_key"],
           "Content-Type": "application/json"}

FILLER = config["filler"]

def send(body, index):
    started = time.monotonic()
    try:
        response = requests.post(base_url + "/v1/chat/completions", headers=headers,
                                 json=body, timeout=config["timeout_s"])
    except requests.RequestException as error:
        return {"index": index, "status": 0, "latency_s": round(time.monotonic() - started, 2),
                "transport_error": type(error).__name__ + ": " + str(error)[:120]}
    latency = round(time.monotonic() - started, 2)
    record = {"index": index, "status": response.status_code, "latency_s": latency}
    try:
        payload = response.json()
    except ValueError:
        record["error_message"] = "non-JSON body"
        return record
    if response.status_code != 200:
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else str(error)
        record["error_message"] = (message or "")[:200]
        return record
    usage = payload.get("usage") or {}
    record["completion_tokens"] = usage.get("completion_tokens")
    record["prompt_tokens"] = usage.get("prompt_tokens")
    choices = payload.get("choices") or []
    if choices:
        record["finish_reason"] = choices[0].get("finish_reason")
    return record

RETRYABLE = (429, 502, 503)

def send_with_retry(body, index, seed):
    """Retry while the gateway sheds, mirroring the local harness exactly.

    Without this the two modes are not comparable: the tunnelled path rides out
    transient shedding and the remote one would report it as permanent failure.
    """
    import random
    rng = random.Random(seed)
    shed = []
    waited = 0.0
    record = {}
    for attempt in range(1, config["max_attempts"] + 1):
        record = send(body, index)
        record["attempts"] = attempt
        if record.get("status") not in RETRYABLE:
            break
        shed.append(record["status"])
        if attempt == config["max_attempts"]:
            break
        window = min(config["backoff_cap_s"],
                     config["backoff_base_s"] * (2 ** (attempt - 1)))
        pause = rng.uniform(0, window)
        time.sleep(pause)
        waited += pause
    record["shed_statuses"] = shed
    record["waited_s"] = round(waited, 2)
    return record

def make_body(prompt, seed, output_tokens=None):
    body = {"model": config["model"],
            "messages": [{"role": "user",
                          "content": prompt + "\\n\\nContinue the passage above. Do not stop early."}],
            "max_tokens": output_tokens or config["output_tokens"],
            "min_tokens": output_tokens or config["output_tokens"],
            "stream": False, "seed": seed}
    if config["thinking_budget"] is not None:
        body["thinking_token_budget"] = config["thinking_budget"]
    return body

# Calibrate the tokenizer before sizing the real prompt. Only prompt_tokens is
# needed from the reply, so generate the minimum the route allows — sizing this
# like a real request cost ~200s and 4,096 wasted tokens before the burst began.
sample = FILLER * 8
probe = send(make_body(sample, config["seed_base"], output_tokens=16), -1)
if probe.get("status") != 200 or not probe.get("prompt_tokens"):
    with open(config["result_path"], "w") as handle:
        json.dump({"error": "calibration failed", "probe": probe}, handle)
    sys.exit(1)
chars_per_token = len(sample) / probe["prompt_tokens"]
target_chars = max(1, int(config["prompt_tokens"] * chars_per_token))
prompt = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]

concurrency = config["concurrency"]
count = config["requests"]
gate = threading.Barrier(concurrency, timeout=300) if count >= concurrency else None

def fire(index):
    if gate is not None:
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
    seed = config["seed_base"] + 1 + index
    return send_with_retry(make_body(prompt, seed), index, seed)

# Each record is appended as it completes, so an aborted run leaves the work
# that did finish instead of nothing, and progress can be watched from outside
# by counting lines. Holding everything in memory until the end cost a full
# hour of a previous run when it had to be stopped.
progress_path = config["progress_path"]
progress_lock = threading.Lock()
completed = [0]

def record_done(record):
    with progress_lock:
        completed[0] += 1
        with open(progress_path, "a") as handle:
            handle.write(json.dumps(record) + chr(10))
        if completed[0] % 10 == 0 or completed[0] == count:
            sys.stderr.write("done %d/%d%s" % (completed[0], count, chr(10)))
            sys.stderr.flush()
    return record

started = time.monotonic()
with ThreadPoolExecutor(max_workers=concurrency) as pool:
    records = list(pool.map(lambda index: record_done(fire(index)), range(count)))
wall = round(time.monotonic() - started, 2)

with open(config["result_path"], "w") as handle:
    json.dump({"chars_per_token": chars_per_token, "wall_clock_s": wall,
               "records": records}, handle)
'''


def collect_on_server(target: GatewayTarget, model: str, prompt_tokens: int,
                      output_tokens: int, requests_count: int, concurrency: int,
                      seed_base: int, timeout_s: int, thinking_budget: int | None,
                      filler: str, max_attempts: int = 5, backoff_base_s: float = 0.5,
                      backoff_cap_s: float = 30.0,
                      poll_seconds: int = 20) -> tuple[list[RequestOutcome], float, float]:
    """Start the burst on the box detached, then poll until its result file appears.

    Detached on purpose. An earlier design piped the payload through the stdout
    of a long-lived `ssh` session, and twice the remote command finished while
    the channel stayed open — leaving the local side blocked forever on a run
    that had actually completed. Nothing now depends on that session surviving:
    the collector is started with `nohup setsid`, writes its result to a file,
    and every later interaction is a short independent ssh.

    Timing comes from the far side: local timing would fold in SSH round-trips,
    which is the whole reason this mode exists.
    """
    prefix = f"/tmp/gonka-bench-{seed_base}"
    script_path, config_path = f"{prefix}.py", f"{prefix}.json"
    result_path, progress_path, log_path = f"{prefix}.result", f"{prefix}.jsonl", f"{prefix}.log"
    config = {
        "base_url": target.gateway_url,
        "admin_key": target.admin_key,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "requests": requests_count,
        "concurrency": concurrency,
        "seed_base": seed_base,
        "timeout_s": timeout_s,
        "thinking_budget": thinking_budget,
        "filler": filler,
        "max_attempts": max_attempts,
        "backoff_base_s": backoff_base_s,
        "backoff_cap_s": backoff_cap_s,
        "progress_path": progress_path,
        "result_path": result_path,
    }
    ssh_base = ["ssh", "-p", str(target.ssh_port), target.ssh_host]

    def remote(command: str, stdin: str | None = None, timeout: int = 120):
        return subprocess.run(ssh_base + [command], input=stdin, text=True,
                              capture_output=True, timeout=timeout)

    upload = remote(f"cat > {shlex.quote(script_path)}", stdin=REMOTE_COLLECTOR)
    if upload.returncode != 0:
        raise SystemExit(f"could not upload collector: {upload.stderr[:300]}")
    # The config carries the admin key, so it is written with a tight mode and
    # the collector unlinks it as its first action.
    written = remote(f"umask 077 && cat > {shlex.quote(config_path)}",
                     stdin=json.dumps(config))
    if written.returncode != 0:
        raise SystemExit(f"could not upload config: {written.stderr[:300]}")

    started = remote(
        f"nohup setsid python3 {shlex.quote(script_path)} {shlex.quote(config_path)} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo started"
    )
    if started.returncode != 0:
        raise SystemExit(f"could not start collector: {started.stderr[:300]}")

    deadline = time.monotonic() + timeout_s + 900
    last_done = -1
    while time.monotonic() < deadline:
        time.sleep(poll_seconds)
        probe = remote(
            f"if [ -f {shlex.quote(result_path)} ]; then echo READY; else "
            f"echo RUNNING $(wc -l < {shlex.quote(progress_path)} 2>/dev/null || echo 0); fi"
        )
        status = (probe.stdout or "").strip()
        if status.startswith("READY"):
            break
        done = int(status.split()[-1]) if status.split()[-1].isdigit() else 0
        if done != last_done:
            print(f"[bench] {done}/{requests_count} requests complete")
            last_done = done
    else:
        raise SystemExit(f"collector did not finish within the deadline; "
                         f"partial records remain at {progress_path} on the box")

    fetched = remote(f"cat {shlex.quote(result_path)}", timeout=600)
    if fetched.returncode != 0:
        raise SystemExit(f"could not fetch result: {fetched.stderr[:300]}")
    try:
        payload = json.loads(fetched.stdout)
    except ValueError as error:
        raise SystemExit(f"could not parse result: {error}; "
                         f"got {fetched.stdout[:300]!r}") from error
    if "error" in payload:
        raise SystemExit(f"remote collector: {payload['error']} — {payload.get('probe')}")

    remote(f"rm -f {prefix}.py {prefix}.json {prefix}.result {prefix}.log")
    print(f"[bench] per-request records left at {progress_path} on the box")

    outcomes = [
        RequestOutcome(
            index=record["index"], seed=seed_base + 1 + record["index"],
            status=record.get("status", 0), latency_s=record.get("latency_s", 0.0),
            completion_tokens=record.get("completion_tokens"),
            prompt_tokens=record.get("prompt_tokens"),
            finish_reason=record.get("finish_reason"),
            error_message=record.get("error_message"),
            transport_error=record.get("transport_error"),
            attempts=record.get("attempts", 1),
            shed_statuses=record.get("shed_statuses") or [],
            waited_s=record.get("waited_s", 0.0),
        )
        for record in payload["records"]
    ]
    return outcomes, payload["wall_clock_s"], payload["chars_per_token"]


def describe_placement(on_server: bool) -> str:
    """One line for the artifact recording where the load was generated from."""
    return ("generated on the gateway box, no SSH tunnel in the measurement path"
            if on_server else
            "generated locally through an SSH forward tunnel, which is in the latency path")
