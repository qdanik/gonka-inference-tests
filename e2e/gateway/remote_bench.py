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
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
                "finished_at": round(time.time(), 2),
                "transport_error": type(error).__name__ + ": " + str(error)[:120]}
    latency = round(time.monotonic() - started, 2)
    record = {"index": index, "status": response.status_code, "latency_s": latency,
              "finished_at": round(time.time(), 2)}
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
    # The response id names the devshard that served the request
    # ("devshard-48087-4203"), which is the only way to attribute results to a
    # host. Always recorded — it costs nothing and cannot be recovered later.
    record["response_id"] = payload.get("id")
    record["system_fingerprint"] = payload.get("system_fingerprint")
    choices = payload.get("choices") or []
    if choices:
        record["finish_reason"] = choices[0].get("finish_reason")
        # MiniMax puts its chain of thought in a separate `reasoning` field and
        # leaves `content` empty until it has finished thinking. Recording both
        # lengths is what separates "the model said nothing" from "the model was
        # still reasoning" — a distinction every earlier run got wrong, because
        # only `content` was read and such replies were filed as empty.
        message = choices[0].get("message") or {}
        record["content_chars"] = len(message.get("content") or "")
        record["reasoning_chars"] = len(message.get("reasoning") or "")
    if config.get("save_content"):
        # The whole body, not just the text: id, usage, fingerprint and any
        # field the gateway adds later are all worth having after the fact.
        record["response"] = payload
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
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_tokens or config["output_tokens"],
            "min_tokens": output_tokens or config["output_tokens"],
            "stream": False, "seed": seed}
    if config["thinking_budget"] is not None:
        body["thinking_token_budget"] = config["thinking_budget"]
    if config.get("logprobs"):
        body["logprobs"] = True
        body["top_logprobs"] = config.get("top_logprobs", 5)
    return body

# Calibrate the tokenizer before sizing the real prompt. Only prompt_tokens is
# needed from the reply, so generate the minimum the route allows — sizing this
# like a real request cost ~200s and 4,096 wasted tokens before the burst began.
sample = FILLER * 8
# Through send_with_retry, not send: this one probe gates the whole run, and a
# single transient 503 here once threw away a started run — and the 90 MB corpus
# upload that preceded it — while the gateway was perfectly healthy a second later.
probe = send_with_retry(make_body(sample, config["seed_base"], output_tokens=16),
                        -1, config["seed_base"])
if probe.get("status") != 200 or not probe.get("prompt_tokens"):
    with open(config["result_path"], "w") as handle:
        json.dump({"error": "calibration failed", "probe": probe}, handle)
    sys.exit(1)
chars_per_token = len(sample) / probe["prompt_tokens"]
target_chars = max(1, int(config["prompt_tokens"] * chars_per_token))

# Build the context out of varied prose rather than one sentence repeated to
# length. A degenerate prompt is not a benchmark of production inference: the
# model falls into repetition loops (earlier runs came back full of "(.) (.)
# (.)"), and highly redundant context does not exercise attention or KV cache
# the way real input does.
TOPICS = ["consensus", "sharding", "latency budgets", "cache eviction", "quorum reads",
          "backpressure", "replication lag", "idempotency", "circuit breaking",
          "token accounting", "batch scheduling", "memory bandwidth", "speculative decoding",
          "prefix caching", "load shedding", "rate limiting", "escrow settlement",
          "proof verification", "nonce rotation", "capacity planning"]
SUBJECTS = ["The scheduler", "A participant node", "The routing layer", "Each replica",
            "The verification pass", "A settlement batch", "The admission controller",
            "The token accountant", "An idle worker", "The retry policy"]
VERBS = ["reconciles", "defers", "amortises", "rejects", "buffers", "replays",
         "partitions", "throttles", "coalesces", "invalidates"]
OBJECTS = ["pending writes", "in-flight requests", "stale entries", "unacknowledged batches",
           "overdue receipts", "speculative branches", "queued completions",
           "orphaned sessions", "partial results", "expired leases"]

def build_context(rng, target):
    parts, size = [], 0
    section = 1
    while size < target:
        topic = rng.choice(TOPICS)
        parts.append("Section %d: notes on %s." % (section, topic))
        for _ in range(rng.randint(4, 9)):
            sentence = "%s %s %s under %s, holding %d of them for %d ms before the %s stage." % (
                rng.choice(SUBJECTS), rng.choice(VERBS), rng.choice(OBJECTS),
                rng.choice(TOPICS), rng.randint(2, 9999), rng.randint(5, 4000),
                rng.choice(TOPICS))
            parts.append(sentence)
        size = sum(len(p) + 1 for p in parts)
        section += 1
    return " ".join(parts)[:target]

SYNTHETIC_TASK = (chr(10) * 2 + "Write the report now, directly, with no preamble and no "
        "planning out loud. Start immediately with the heading for Section 1 and work through "
        "the sections in order. For each one give a short description, the failure modes it "
        "invites, and how you would test for them. Keep writing until every section is "
        "covered.")

CORPUS_TASK = (chr(10) * 2 + "You have just read the opening of a book. Write a close "
        "reading of it, directly, with no preamble. Cover, in this order: (1) what happens, "
        "scene by scene; (2) each character who speaks or acts, and what drives them; "
        "(3) four passages worth quoting, each quoted exactly and then explained; "
        "(4) the themes the text develops and how its prose builds them; "
        "(5) anything left unresolved where the text breaks off. Be specific to this book "
        "and quote from it rather than generalising.")

count_needed = config["requests"]
CORPUS_PATH = config.get("corpus_path") or ""
DOCUMENTS = []
if CORPUS_PATH:
    with open(CORPUS_PATH) as handle:
        DOCUMENTS = json.load(handle)["documents"]

import random as _random

def document_for(index):
    """One real book per request, read from its first page.

    Not a window cut from the middle of a concatenated corpus: the model gets a
    document that actually begins where it begins. Distinct books also mean
    distinct prefixes, so the gateway's prefix cache cannot serve one request
    from another — which a repeated prompt would let it do, turning a decode
    benchmark into a cache benchmark.
    """
    document = DOCUMENTS[index % len(DOCUMENTS)]
    text = document["text"]
    # First pass through the pool reads every book from its first page. A soak
    # runs far past that, so later passes step deeper into the same books rather
    # than resending a prompt the gateway's prefix cache already holds — which
    # would measure the cache instead of the fleet.
    room = max(len(text) - target_chars, 1)
    start = (index // len(DOCUMENTS)) * target_chars % room
    if start:
        paragraph = text.find(chr(10) * 2, start, start + 4000)
        start = paragraph + 2 if paragraph != -1 else start
    return (text[start:start + target_chars],
            {"id": document["id"], "title": document["title"], "offset": start})

def build_prompt_for(seed, index=0):
    """The prompt for one request, and which book it came from."""
    if DOCUMENTS:
        text, source = document_for(index)
        return text + CORPUS_TASK, source
    return build_context(_random.Random(seed), target_chars) + SYNTHETIC_TASK, None

prompt = build_prompt_for(config["seed_base"], 0)[0]
if DOCUMENTS:
    windows = sum(max(len(d["text"]) // target_chars, 1) for d in DOCUMENTS)
    if count_needed > windows:
        sys.stderr.write("warning: %d requests but the corpus holds %d distinct "
                         "windows — prompts repeat past that%s"
                         % (count_needed, windows, chr(10)))
    else:
        sys.stderr.write("corpus: %d books, %d distinct windows of %d chars%s"
                         % (len(DOCUMENTS), windows, target_chars, chr(10)))

concurrency = config["concurrency"]
count = config["requests"]
gate = threading.Barrier(concurrency, timeout=300) if count >= concurrency else None

def dispatch(index):
    """Send one request, keeping the body beside the reply.

    The prompt is built here on the box, so without this the local side would
    have to reconstruct it to show what was asked — two implementations of one
    rule, free to drift apart. Recorded only under --save-content: a prompt is
    half a megabyte.
    """
    seed = config["seed_base"] + 1 + index
    prompt, source = build_prompt_for(seed, index)
    body = make_body(prompt, seed)
    record = send_with_retry(body, index, seed)
    record["seed"] = seed
    if config.get("save_requests"):
        record["request"] = body
    if config.get("save_content") or config.get("save_requests"):
        if source:
            record["document"] = source
    return record

def fire(index):
    if gate is not None:
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
    return dispatch(index)

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

duration_s = config.get("duration_s") or 0
started = time.monotonic()

if duration_s:
    # Each worker replaces its own finished request immediately, which keeps
    # exactly `concurrency` requests in flight for the whole window rather than
    # letting the burst drain.
    deadline = started + duration_s
    counter = [0]
    counter_lock = threading.Lock()

    def take_index():
        with counter_lock:
            counter[0] += 1
            return counter[0] - 1

    def worker(_slot):
        while time.monotonic() < deadline:
            record_done(dispatch(take_index()))
        return None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(concurrency)))
    records = []
else:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(lambda index: record_done(fire(index)), range(count)))
wall = round(time.monotonic() - started, 2)

with open(config["result_path"], "w") as handle:
    json.dump({"chars_per_token": chars_per_token, "wall_clock_s": wall,
               "records": records, "completed": completed[0],
               "soak": bool(duration_s),
               "prompt": prompt if config.get("save_content") else ""}, handle)
'''


def fetch_remote_file(target: GatewayTarget, remote_path: str, local_path: Path,
                      label: str, attempts: int = 4) -> None:
    """Copy a file off the box, retrying transient drops.

    A finished run's payload is worth up to an hour of GPU time, so a single
    dropped connection must never be what loses it — one already did. The file
    stays on the box on failure so the fetch can be repeated by hand.
    """
    last_error = ""
    for attempt in range(1, attempts + 1):
        copied = subprocess.run(
            ["scp", "-C", "-P", str(target.ssh_port), "-q",
             "-o", "ConnectTimeout=30",
             "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8",
             f"{target.ssh_host}:{remote_path}", str(local_path)],
            capture_output=True, text=True, timeout=3600)
        if copied.returncode == 0:
            return
        last_error = (copied.stderr or "").strip()[:300]
        print(f"[bench] fetching {label} failed "
              f"(attempt {attempt}/{attempts}): {last_error}")
    raise SystemExit(f"could not fetch {label} after {attempts} attempts: "
                     f"{last_error}\nit is still on the box at {remote_path}")


def collect_on_server(target: GatewayTarget, model: str, prompt_tokens: int,
                      output_tokens: int, requests_count: int, concurrency: int,
                      seed_base: int, timeout_s: int, thinking_budget: int | None,
                      filler: str, max_attempts: int = 5, backoff_base_s: float = 0.5,
                      backoff_cap_s: float = 30.0, duration_s: int = 0,
                      save_content: bool = False, save_requests: bool = True,
                      logprobs: bool = False,
                      top_logprobs: int = 5, corpus_path: Path | None = None,
                      poll_seconds: int = 20) -> tuple[list[RequestOutcome], float, float, str,
                                                       dict[int, Any], dict[int, Any]]:
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
    corpus_remote = (f"/tmp/gonka-corpus-{corpus_path.stem}-{corpus_path.stat().st_size}.json"
                     if corpus_path else "")
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
        "duration_s": duration_s,
        "save_content": save_content,
        "save_requests": save_content and save_requests,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "corpus_path": corpus_remote,
    }
    ssh_base = ["ssh", "-p", str(target.ssh_port), target.ssh_host]

    def remote(command: str, stdin: str | None = None, timeout: int = 120):
        return subprocess.run(ssh_base + [command], input=stdin, text=True,
                              capture_output=True, timeout=timeout)

    upload = remote(f"cat > {shlex.quote(script_path)}", stdin=REMOTE_COLLECTOR)
    if upload.returncode != 0:
        raise SystemExit(f"could not upload collector: {upload.stderr[:300]}")
    if corpus_path:
        # Named for its content rather than this run, and left in place
        # afterwards: the pool is tens of megabytes and identical between runs,
        # so re-uploading it every time is minutes of nothing for no gain.
        size = corpus_path.stat().st_size
        already = remote(f"stat -c %s {shlex.quote(corpus_remote)} 2>/dev/null || echo 0")
        if (already.stdout or "").strip() == str(size):
            print(f"[bench] corpus already on the box at {corpus_remote} "
                  f"({size / 1e6:.1f} MB), skipping upload")
        else:
            # Tens of megabytes of prose, so scp rather than an ssh heredoc.
            uploaded = subprocess.run(
                ["scp", "-C", "-P", str(target.ssh_port), "-q",
                 "-o", "ConnectTimeout=30",
                 str(corpus_path), f"{target.ssh_host}:{corpus_remote}"],
                capture_output=True, text=True, timeout=1800)
            if uploaded.returncode != 0:
                raise SystemExit(f"could not upload corpus: {uploaded.stderr[:300]}")
            print(f"[bench] corpus {corpus_path.name} ({size / 1e6:.1f} MB) "
                  f"uploaded to {corpus_remote}")
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

    # Give up on being stuck, not on being slow. A fixed deadline of
    # "one request timeout plus slack" assumed the whole burst was one wave; on
    # a slow host 100 requests at 34 concurrent take three waves and ran past
    # it, abandoning a finished run at 99/100 while the collector was still
    # serialising its result. The run now ends only when nothing has completed
    # for longer than a single request could legitimately take.
    stall_limit = duration_s + timeout_s + 900 if duration_s else timeout_s + 900
    last_progress = time.monotonic()
    last_done = -1
    while time.monotonic() - last_progress < stall_limit:
        time.sleep(poll_seconds)
        probe = remote(
            f"if [ -f {shlex.quote(result_path)} ]; then echo READY; else "
            f"echo RUNNING $(wc -l < {shlex.quote(progress_path)} 2>/dev/null || echo 0); fi"
        )
        status = (probe.stdout or "").strip()
        if status.startswith("READY"):
            break
        # An ssh probe can come back empty on a hiccup. Treating that as fatal
        # once cost the poller a six-hour soak — the collector kept running on
        # the box, but nothing was here to collect it.
        fields = status.split()
        done = int(fields[-1]) if fields and fields[-1].isdigit() else last_done
        if done != last_done:
            scope = (f" in the {duration_s // 60} min window" if duration_s
                     else f"/{requests_count}")
            print(f"[bench] {done}{scope} requests complete")
            last_done = done
            last_progress = time.monotonic()
    else:
        raise SystemExit(
            f"collector made no progress for {stall_limit / 60:.0f} min "
            f"(last seen {last_done}"
            f"{' completed' if duration_s else '/' + str(requests_count)}); records remain at "
            f"{progress_path} on the box — recover them with "
            f"scripts/recover_run.py rather than rerunning")

    # Copied to disk rather than piped through `cat`: with logprobs enabled a
    # run's payload reaches a gigabyte, and capturing that as a subprocess
    # string would hold it in memory twice over before parsing even starts.
    local_result = Path(tempfile.gettempdir()) / f"gonka-bench-{seed_base}.result"
    fetch_remote_file(target, result_path, local_result, "result")
    try:
        with local_result.open() as handle:
            payload = json.load(handle)
    except ValueError as error:
        raise SystemExit(f"could not parse result: {error}") from error
    finally:
        local_result.unlink(missing_ok=True)
    if "error" in payload:
        raise SystemExit(f"remote collector: {payload['error']} — {payload.get('probe')}")

    if payload.get("soak") and not payload.get("records"):
        local_records = Path(tempfile.gettempdir()) / f"gonka-bench-{seed_base}.jsonl"
        fetch_remote_file(target, progress_path, local_records, "soak records")
        with local_records.open() as handle:
            payload["records"] = [json.loads(line) for line in handle if line.strip()]
        local_records.unlink(missing_ok=True)

    remote(f"rm -f {prefix}.py {prefix}.json {prefix}.log")
    print(f"[bench] per-request records left at {progress_path} on the box")
    print(f"[bench] raw result kept at {result_path} on the box")

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
            finished_at=record.get("finished_at", 0.0),
            response_id=record.get("response_id"),
            system_fingerprint=record.get("system_fingerprint"),
            content_chars=record.get("content_chars", 0),
            reasoning_chars=record.get("reasoning_chars", 0),
        )
        for record in payload["records"]
    ]
    contents = {r["index"]: r["response"] for r in payload["records"] if "response" in r}
    sent = {r["index"]: {key: r[key] for key in ("seed", "document", "request") if key in r}
            for r in payload["records"] if "request" in r}
    return (outcomes, payload["wall_clock_s"], payload["chars_per_token"],
            payload.get("prompt", ""), contents, sent)


def describe_placement(on_server: bool) -> str:
    """One line for the artifact recording where the load was generated from."""
    return ("generated on the gateway box, no SSH tunnel in the measurement path"
            if on_server else
            "generated locally through an SSH forward tunnel, which is in the latency path")
