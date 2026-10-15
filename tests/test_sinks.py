"""Tests for the notification sinks in gradecrossing.alerting: one JSON serialiser with an
event id, the ASCII helper, close() on every sink, and reliable webhook delivery (retries
with back-off, per-operation timeouts inside a wall-clock deadline per event, aborted
attempts that never send late, ALARM priority, overflow policy).

Everything runs against local HTTP servers on 127.0.0.1; no external network.
"""
from __future__ import annotations

import dataclasses
import http.server
import io
import json
import socket
import threading
import time
import urllib.error
from typing import List

import pytest

from gradecrossing.alerting import (
    CallbackSink,
    JsonlSink,
    StdoutSink,
    WebhookSink,
    _retryable,
    ascii_safe,
    build_sinks,
    event_id,
    event_to_json,
)
from gradecrossing.config import AlertConfig
from gradecrossing.types import Event


def ev(etype: str = "ALARM_START", t: float = 1.0, **extra) -> Event:
    return Event(etype, t, int(t * 30), "ALARM" if etype.startswith("ALARM") else "CLEAR",
                 ["vehicle 7 (car) stationary in zone 8.0s"], [7], dict(extra))


# --------------------------------------------------------------------------------------
# local HTTP servers
# --------------------------------------------------------------------------------------
class Server:
    """Threaded local endpoint. ``plan`` is a list of per-request behaviours, consumed in
    order (the last one repeats): an int HTTP status, ("delay", seconds, status),
    ("trickle", seconds_per_byte) or ("slow", gap): a 200 answer in two parts, each sent
    ``gap`` seconds after the previous one (a slow but steady uplink)."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.bodies: List[dict] = []
        self.started = threading.Event()  # set when the first request arrives
        self.lock = threading.Lock()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (http.server naming)
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                with server.lock:
                    step = server.plan.pop(0) if len(server.plan) > 1 else server.plan[0]
                server.started.set()
                try:
                    if isinstance(step, tuple) and step[0] == "trickle":
                        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
                        for i in range(len(reply)):
                            self.wfile.write(reply[i:i + 1])
                            self.wfile.flush()
                            time.sleep(step[1])
                        return
                    if isinstance(step, tuple) and step[0] == "slow":
                        with server.lock:
                            server.bodies.append(body)
                        for part in (b"HTTP/1.1 200 OK\r\n", b"Content-Length: 2\r\nConnection: close\r\n\r\nok"):
                            time.sleep(step[1])
                            self.wfile.write(part)
                            self.wfile.flush()
                        return
                    status = step
                    if isinstance(step, tuple) and step[0] == "delay":
                        time.sleep(step[1])
                        status = step[2]
                    with server.lock:
                        server.bodies.append(body)
                    self.send_response(status)
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"ok")
                except OSError:  # the client gave up (deadline): nothing to answer
                    pass

            def log_message(self, *args):  # keep pytest output quiet
                return

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/hook"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server_factory():
    servers = []

    def make(plan):
        s = Server(plan)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for_post_threads(seconds: float = 5.0) -> bool:
    """Wait until no POST helper thread is alive; True when they all exited."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not any(t.name == "gradecrossing-webhook-post" for t in threading.enumerate()):
            return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------------------
# serialisation, event ids, ascii
# --------------------------------------------------------------------------------------
def test_event_to_json_carries_a_stable_event_id_and_jsonl_keeps_its_format(tmp_path):
    a, b = ev("ALARM_START", 1.0), ev("ALARM_END", 2.0)
    da, db = json.loads(event_to_json(a)), json.loads(event_to_json(b))
    assert da["event_id"] and da["event_id"] != db["event_id"]
    assert json.loads(event_to_json(a))["event_id"] == da["event_id"] == event_id(a)  # cached per event
    # the id is not a dataclass field: equality, asdict and the JSONL line are untouched
    assert {k: v for k, v in da.items() if k != "event_id"} == dataclasses.asdict(a)
    assert json.loads(event_to_json(a, include_id=False)) == dataclasses.asdict(a)
    assert a == ev("ALARM_START", 1.0)
    # a producer-assigned id wins
    assert event_id(ev(event_id="run7-000042")) == "run7-000042"
    path = tmp_path / "events.jsonl"
    with JsonlSink(str(path)) as sink:
        sink.emit(a)
    line = json.loads(path.read_text(encoding="utf-8"))
    assert "event_id" not in line and line == dataclasses.asdict(a)


def test_ascii_safe_is_the_one_console_helper():
    assert ascii_safe("café → ok") == "caf? ? ok"
    assert ascii_safe(12.5) == "12.5"
    buf = io.StringIO()
    StdoutSink(stream=buf).emit(Event("HEALTH", 1.0, 30, "CLEAR", ["zone écart"], [], {}))
    assert "zone ?cart" in buf.getvalue()


def test_every_sink_has_close(tmp_path):
    buf = io.StringIO()
    out = StdoutSink(stream=buf)
    out.emit(ev())
    out.close()
    assert not buf.closed  # flushes, never closes the console stream
    StdoutSink().close()
    got = []
    cb = CallbackSink(got.append)
    cb.close()
    jsonl = JsonlSink(str(tmp_path / "e.jsonl"))
    jsonl.close()
    jsonl.close()  # idempotent
    hook = WebhookSink(f"http://127.0.0.1:{free_port()}/x", timeout=0.5)
    assert hook.close(timeout=1.0) is True  # nothing queued
    for sink in (out, cb, jsonl, hook):
        assert callable(getattr(sink, "close"))


def test_retryable_classification():
    def http_error(code):
        return urllib.error.HTTPError("http://x", code, "x", {}, None)

    assert all(_retryable(http_error(c)) for c in (500, 502, 503, 504, 408, 429))
    assert not any(_retryable(http_error(c)) for c in (400, 401, 403, 404, 410))
    assert _retryable(urllib.error.URLError("refused")) and _retryable(TimeoutError()) and _retryable(OSError())
    assert not _retryable(ValueError("unknown url type"))


# --------------------------------------------------------------------------------------
# webhook delivery
# --------------------------------------------------------------------------------------
def test_webhook_retries_a_failed_post_with_the_same_event_id(server_factory):
    # MISS-02 / F14: an endpoint that answers 503 to the first POST lost the ALARM_START.
    srv = server_factory([503, 200])
    sink = WebhookSink(srv.url, timeout=2.0, backoff_seconds=0.05)
    alarm = ev("ALARM_START", 8.0)
    sink.emit(alarm)
    assert sink.flush(timeout=10.0)
    assert (sink.delivered, sink.failures, sink.retries) == (1, 0, 1)
    assert [b["type"] for b in srv.bodies] == ["ALARM_START", "ALARM_START"]
    assert srv.bodies[0]["event_id"] == srv.bodies[1]["event_id"] == event_id(alarm)
    assert srv.bodies[1] == json.loads(event_to_json(alarm))


def test_webhook_gives_up_after_its_attempts_and_skips_client_errors(server_factory):
    srv = server_factory([503])
    sink = WebhookSink(srv.url, timeout=2.0, attempts=3, backoff_seconds=0.05)
    t0 = time.monotonic()
    sink.emit(ev())
    assert sink.flush(timeout=10.0)
    assert (sink.delivered, sink.failures, sink.retries) == (0, 1, 2)
    assert len(srv.bodies) == 3
    assert time.monotonic() - t0 >= 0.05 + 0.1  # waited 0.05 s, then 0.1 s (doubling)
    bad_request = server_factory([400])
    sink2 = WebhookSink(bad_request.url, timeout=2.0, backoff_seconds=0.05)
    sink2.emit(ev())
    assert sink2.flush(timeout=10.0)
    assert (sink2.failures, sink2.retries, len(bad_request.bodies)) == (1, 0, 1)  # 4xx: no retry


def test_webhook_deadline_bounds_a_trickling_endpoint(server_factory):
    # F14: an endpoint trickling one byte at a time kept the single worker busy for 16 s
    # (urllib's timeout bounds each read, not the request). The attempt still running at
    # the event's deadline is now aborted and the worker moves on to the next event.
    srv = server_factory([("trickle", 0.25), 200])
    sink = WebhookSink(srv.url, timeout=1.0, attempts=2, backoff_seconds=0.05, deadline_seconds=2.0)
    t0 = time.monotonic()
    sink.emit(ev("ALARM_START", 1.0))
    sink.emit(ev("ALARM_END", 2.0))
    assert sink.flush(timeout=10.0)
    elapsed = time.monotonic() - t0
    assert (sink.delivered, sink.failures, sink.retries) == (1, 1, 0)  # no time left for a retry
    assert 2.0 <= elapsed < 3.5, elapsed  # the trickle would have taken 10 s
    assert [b["type"] for b in srv.bodies] == ["ALARM_END"]
    assert wait_for_post_threads(3.0)  # the aborted helper thread exits too


def test_webhook_timeout_is_per_operation_so_a_slow_steady_endpoint_is_delivered_once(server_factory):
    # The answer takes 2.4 s in all, longer than the 2 s timeout, but no single read waits
    # more than 1.2 s: one POST, delivered. (A wall-clock cap of one timeout per attempt
    # failed it three times while the receiver got three copies.)
    srv = server_factory([("slow", 1.2)])
    sink = WebhookSink(srv.url, timeout=2.0)
    t0 = time.monotonic()
    sink.emit(ev("ALARM_START", 8.0))
    assert sink.flush(timeout=20.0)
    assert (sink.delivered, sink.failures, sink.retries) == (1, 0, 0)
    assert len(srv.bodies) == 1 and time.monotonic() - t0 < 5.0


def test_webhook_aborted_attempt_never_sends_a_stale_copy(server_factory, monkeypatch):
    # The first name lookup hangs past the event's deadline: the worker gives up and
    # delivers the next event. When the lookup finally returns, the abandoned attempt must
    # not connect and POST the old ALARM_START after the newer ALARM_END.
    srv = server_factory([200])
    port = srv.httpd.server_address[1]
    real = socket.getaddrinfo
    calls = []

    def slow_first_lookup(host, p, *args, **kwargs):
        if str(p) == str(port):
            calls.append(host)
            if len(calls) == 1:
                time.sleep(1.5)
        return real(host, p, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow_first_lookup)
    sink = WebhookSink(srv.url, timeout=0.5, attempts=1, deadline_seconds=0.6)
    sink.emit(ev("ALARM_START", 8.0))
    sink.emit(ev("ALARM_END", 20.0))
    assert sink.flush(timeout=10.0)
    assert (sink.delivered, sink.failures) == (1, 1)
    assert wait_for_post_threads(5.0)  # the stuck lookup returned and its thread finished
    assert len(calls) >= 2
    assert [b["type"] for b in srv.bodies] == ["ALARM_END"]


def test_webhook_deadline_covers_all_attempts():
    sink = WebhookSink(f"http://127.0.0.1:{free_port()}/x", timeout=0.4, attempts=10,
                       backoff_seconds=0.05, deadline_seconds=1.0)
    t0 = time.monotonic()
    sink.emit(ev())
    assert sink.flush(timeout=10.0)
    assert sink.failures == 1 and sink.delivered == 0
    assert time.monotonic() - t0 < 2.0  # 10 attempts would take far longer
    assert sink.retries < 9
    # default: every attempt may time out once, plus the back-off waits (0.5 + 1 + 2 s)
    assert WebhookSink("http://x/y", timeout=2.0, attempts=4).deadline_seconds == pytest.approx(8.0 + 3.5)
    assert WebhookSink("http://x/y", timeout=2.0, attempts=1).deadline_seconds == pytest.approx(2.0)


def test_webhook_delivers_alarm_events_before_queued_others(server_factory):
    srv = server_factory([("delay", 0.5, 200), 200])
    sink = WebhookSink(srv.url, timeout=3.0)
    sink.emit(ev("HEALTH", 0.5))  # in flight while the rest is queued
    assert srv.started.wait(5.0)
    for etype, t in (("WARNING_START", 1.0), ("HEALTH", 1.5), ("ALARM_START", 2.0),
                     ("ALARM_UPDATE", 2.5), ("ALARM_END", 3.0), ("WARNING_END", 3.5)):
        sink.emit(ev(etype, t))
    assert sink.flush(timeout=10.0)
    assert [(b["type"], b["timestamp"]) for b in srv.bodies] == [
        ("HEALTH", 0.5), ("ALARM_START", 2.0), ("ALARM_UPDATE", 2.5), ("ALARM_END", 3.0),
        ("WARNING_START", 1.0), ("HEALTH", 1.5), ("WARNING_END", 3.5)]


def _event(etype, t, status):
    return Event(etype, t, int(t * 30), status, [], [7], {})


def test_webhook_sends_the_alarm_start_of_a_warning_to_alarm_transition_first(server_factory):
    # WARNING -> ALARM emits WARNING_END (status ALARM) and then ALARM_START. The ALARM_START
    # goes first and the WARNING_END right behind it, ahead of the older queued notices: it
    # arrives while the alarm is on, never after the ALARM_END with a stale status ALARM.
    srv = server_factory([("delay", 0.5, 200), 200])
    sink = WebhookSink(srv.url, timeout=3.0)
    sink.emit(_event("HEALTH", 0.5, "CLEAR"))  # in flight while the rest is queued
    assert srv.started.wait(5.0)
    for etype, t, status in (("WARNING_START", 1.0, "WARNING"), ("HEALTH", 1.5, "WARNING"),
                             ("WARNING_END", 2.0, "ALARM"), ("ALARM_START", 2.0, "ALARM"),
                             ("ALARM_END", 3.0, "CLEAR"), ("WARNING_START", 4.0, "WARNING"),
                             ("WARNING_END", 5.0, "CLEAR")):
        sink.emit(_event(etype, t, status))
    assert sink.flush(timeout=10.0)
    assert [(b["type"], b["timestamp"], b["status"]) for b in srv.bodies] == [
        ("HEALTH", 0.5, "CLEAR"), ("ALARM_START", 2.0, "ALARM"), ("WARNING_END", 2.0, "ALARM"),
        ("ALARM_END", 3.0, "CLEAR"), ("WARNING_START", 1.0, "WARNING"), ("HEALTH", 1.5, "WARNING"),
        ("WARNING_START", 4.0, "WARNING"), ("WARNING_END", 5.0, "CLEAR")]
    # the sequence numbers of the ids still give the emit order, for receivers that track
    # the current status: the late WARNING_START is older than the ALARM_END before it
    seqs = [int(b["event_id"].rsplit("-", 1)[1]) for b in srv.bodies]
    emit_order = [(srv.bodies[i]["type"], srv.bodies[i]["timestamp"])
                  for i in sorted(range(len(seqs)), key=lambda i: seqs[i])]
    assert emit_order == [("HEALTH", 0.5), ("WARNING_START", 1.0), ("HEALTH", 1.5), ("WARNING_END", 2.0),
                          ("ALARM_START", 2.0), ("ALARM_END", 3.0), ("WARNING_START", 4.0), ("WARNING_END", 5.0)]
    assert seqs[4] < seqs[3]


def test_webhook_idle_worker_waits_briefly_for_the_alarm_start_of_a_transition(server_factory):
    # The worker is idle and picks up the WARNING_END before the engine's ALARM_START has
    # reached the webhook (the console and JSONL sinks run first): on a slow endpoint the
    # alarm would wait a whole delivery behind it.
    srv = server_factory([("delay", 0.3, 200)])
    sink = WebhookSink(srv.url, timeout=3.0)
    sink.emit(_event("WARNING_END", 2.0, "ALARM"))
    time.sleep(0.05)
    sink.emit(_event("ALARM_START", 2.0, "ALARM"))
    assert sink.flush(timeout=10.0)
    assert [b["type"] for b in srv.bodies] == ["ALARM_START", "WARNING_END"]
    # without an ALARM_START behind it (another producer), the WARNING_END still goes out
    lone = WebhookSink(srv.url, timeout=3.0)
    t0 = time.monotonic()
    lone.emit(_event("WARNING_END", 9.0, "ALARM"))
    assert lone.flush(timeout=10.0) and lone.delivered == 1
    assert time.monotonic() - t0 < 2.0


def test_webhook_full_queue_drops_the_oldest_non_alarm_event(server_factory, caplog):
    srv = server_factory([("delay", 0.5, 200), 200])
    sink = WebhookSink(srv.url, timeout=3.0, max_queue=3)
    sink.emit(ev("HEALTH", 0.0))  # in flight
    assert srv.started.wait(5.0)
    sink.emit(ev("HEALTH", 1.0))
    sink.emit(ev("WARNING_START", 2.0))
    sink.emit(ev("ALARM_START", 3.0))  # queue full now: [HEALTH 1, WARNING 2] + [ALARM 3]
    sink.emit(ev("ALARM_END", 4.0))  # evicts HEALTH 1 (oldest non-ALARM)
    sink.emit(ev("HEALTH", 5.0))  # evicts WARNING 2
    sink.emit(ev("ALARM_START", 6.0, repeat=True))  # evicts HEALTH 5
    sink.emit(ev("ALARM_START", 7.0))  # only ALARMs left: evicts the repeat reminder
    sink.emit(ev("HEALTH", 8.0))  # nothing it may evict: the incoming HEALTH is dropped
    sink.emit(ev("ALARM_END", 9.0))  # three ALARM transitions queued: the incoming one is dropped
    assert sink.flush(timeout=10.0)
    assert [(b["type"], b["timestamp"]) for b in srv.bodies] == [
        ("HEALTH", 0.0), ("ALARM_START", 3.0), ("ALARM_END", 4.0), ("ALARM_START", 7.0)]
    assert sink.dropped == 6 and sink.delivered == 4
    assert any("dropping queued HEALTH" in r.getMessage() for r in caplog.records)


def test_webhook_full_queue_evicts_the_oldest_non_alarm_event_not_the_newest(server_factory):
    # The scenario above ends with every non-ALARM event dropped, so it cannot tell which one
    # went first: here the newer WARNING_START survives the eviction.
    srv = server_factory([("delay", 0.5, 200), 200])
    sink = WebhookSink(srv.url, timeout=3.0, max_queue=2)
    sink.emit(ev("HEALTH", 0.0))  # in flight
    assert srv.started.wait(5.0)
    sink.emit(ev("HEALTH", 1.0))
    sink.emit(ev("WARNING_START", 2.0))
    sink.emit(ev("ALARM_START", 3.0))  # full: evicts HEALTH 1, the oldest non-ALARM event
    assert sink.flush(timeout=10.0)
    assert [(b["type"], b["timestamp"]) for b in srv.bodies] == [
        ("HEALTH", 0.0), ("ALARM_START", 3.0), ("WARNING_START", 2.0)]
    assert sink.dropped == 1 and sink.delivered == 3


def test_webhook_full_queue_evicts_a_transition_warning_end_before_an_alarm(server_factory):
    srv = server_factory([("delay", 0.5, 200), 200])
    sink = WebhookSink(srv.url, timeout=3.0, max_queue=2)
    sink.emit(ev("HEALTH", 0.0))  # in flight
    assert srv.started.wait(5.0)
    sink.emit(_event("WARNING_END", 1.0, "ALARM"))  # a transition WARNING_END: queued with the ALARMs
    sink.emit(ev("ALARM_START", 1.0))
    sink.emit(ev("ALARM_END", 2.0))  # full: the WARNING_END goes, not an ALARM event
    assert sink.flush(timeout=10.0)
    assert [(b["type"], b["timestamp"]) for b in srv.bodies] == [
        ("HEALTH", 0.0), ("ALARM_START", 1.0), ("ALARM_END", 2.0)]
    assert sink.dropped == 1


def test_webhook_close_flushes_then_refuses_new_events(server_factory, caplog):
    srv = server_factory([("delay", 0.2, 200), 200])
    sink = WebhookSink(srv.url, timeout=3.0)
    sink.emit(ev("ALARM_START", 1.0))
    sink.emit(ev("ALARM_END", 2.0))
    assert sink.close(timeout=10.0) is True
    assert [b["type"] for b in srv.bodies] == ["ALARM_START", "ALARM_END"]
    sink.emit(ev("HEALTH", 3.0))
    assert sink.dropped == 1 and sink.delivered == 2
    assert any("closed" in r.getMessage() for r in caplog.records)
    assert sink.close(timeout=1.0) is True  # idempotent


def test_webhook_close_sends_the_closing_events_before_queued_health_notices(server_factory):
    # FO5: on a slow endpoint the start-up HEALTH notices filled the queue, and close() waited
    # one event's deadline (0.5 s here) in all: WARNING_END and RUN_END never arrived.
    srv = server_factory([("delay", 0.15, 200)])
    sink = WebhookSink(srv.url, timeout=0.5, attempts=1)
    assert sink.deadline_seconds == pytest.approx(0.5)
    sink.emit(ev("HEALTH", 0.0))  # in flight
    assert srv.started.wait(5.0)
    for etype, t in (("HEALTH", 0.1), ("HEALTH", 0.2), ("HEALTH", 0.3), ("WARNING_START", 1.0),
                     ("WARNING_END", 2.0), ("RUN_END", 3.0)):
        sink.emit(ev(etype, t))
    assert sink.close() is True  # default: 0.5 s per queued event (7 here), not one deadline
    assert [(b["type"], b["timestamp"]) for b in srv.bodies] == [
        ("HEALTH", 0.0), ("WARNING_START", 1.0), ("WARNING_END", 2.0), ("RUN_END", 3.0),
        ("HEALTH", 0.1), ("HEALTH", 0.2), ("HEALTH", 0.3)]


def test_webhook_close_waits_one_timeout_per_queued_event_within_bounds(monkeypatch):
    def budget(n: int, **kwargs) -> float:
        sink = WebhookSink(f"http://127.0.0.1:{free_port()}/x", **kwargs)
        monkeypatch.setattr(sink, "_ensure_worker", lambda: None)  # no worker: everything stays queued
        waited = []
        monkeypatch.setattr(sink, "_flush_until_failing", lambda b: waited.append(b) or False)
        for i in range(n):
            sink.emit(ev("HEALTH", float(i)))
        assert sink.close() is False
        return waited[0]

    assert WebhookSink("http://x/y", timeout=3.0).deadline_seconds == pytest.approx(10.5)
    assert budget(2, timeout=3.0) == pytest.approx(10.5)  # at least one event's deadline
    assert budget(5, timeout=3.0) == pytest.approx(15.0)  # one timeout per queued event
    assert budget(50, timeout=3.0) == pytest.approx(30.0)  # at most 30 s ...
    assert budget(50, timeout=3.0, deadline_seconds=40.0) == pytest.approx(40.0)  # ... or one deadline


def test_webhook_close_gives_a_failing_endpoint_one_deadline_not_one_per_event():
    # Connections are accepted but never answered: every POST times out. close() stops
    # waiting one deadline after it started once a delivery has failed (one timeout per
    # queued event would be 3 s here), like the single-deadline close did.
    hole = socket.socket()
    hole.bind(("127.0.0.1", 0))
    hole.listen(16)
    try:
        sink = WebhookSink(f"http://127.0.0.1:{hole.getsockname()[1]}/x", timeout=0.5, attempts=1)
        for i in range(6):
            sink.emit(ev("HEALTH", float(i)))
        t0 = time.monotonic()
        assert sink.close() is False
        assert time.monotonic() - t0 < 1.5
        assert sink.flush(timeout=10.0)  # the worker still attempts the rest (quietly, in this test)
        assert sink.failures == 6
    finally:
        hole.close()


def test_webhook_close_times_out_on_a_dead_endpoint():
    sink = WebhookSink(f"http://127.0.0.1:{free_port()}/x", timeout=0.5, attempts=3, backoff_seconds=0.05)
    sink.emit(ev())
    sink.emit(ev("ALARM_END", 2.0))
    t0 = time.monotonic()
    assert sink.close(timeout=0.3) is False  # returns at its timeout, events still pending
    assert time.monotonic() - t0 < 1.5
    assert sink.flush(timeout=15.0)  # the worker still finishes the queued events (quietly, in this test)
    assert sink.failures == 2


def test_build_sinks_passes_the_webhook_delivery_settings(tmp_path):
    cfg = AlertConfig(stdout=False, webhook_url="http://127.0.0.1:9/x", webhook_timeout_seconds=2.0,
                      webhook_attempts=5, webhook_backoff_seconds=0.25, webhook_deadline_seconds=7.0,
                      webhook_max_queue=9)
    (hook,) = build_sinks(cfg)
    assert isinstance(hook, WebhookSink)
    assert (hook.timeout, hook.attempts, hook.backoff_seconds, hook.deadline_seconds, hook.max_queue) == (
        2.0, 5, 0.25, 7.0, 9)
    (default_hook,) = build_sinks(AlertConfig(stdout=False, webhook_url="http://127.0.0.1:9/x"))
    assert (default_hook.attempts, default_hook.backoff_seconds, default_hook.max_queue) == (3, 0.5, 256)
    assert default_hook.timeout == AlertConfig().webhook_timeout_seconds == 3.0
    assert default_hook.deadline_seconds == pytest.approx(3 * 3.0 + 0.5 + 1.0)  # 10.5 s
