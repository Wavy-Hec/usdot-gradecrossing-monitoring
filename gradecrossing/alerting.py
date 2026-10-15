"""Alert decision logic and notification sinks.

This module turns the per-object measurements produced by ``StuckVehicleTracker``,
the ``DebrisResult`` from ``DebrisDetector`` and the ``GateState`` from the
infrastructure detector into a single safety ``Assessment`` (CLEAR / WARNING / ALARM)
per frame, and emits ``Event`` objects on status transitions.

Design rules that matter for safety (see docs/ARCHITECTURE.md):

* Decisions are made **per tracked object** using the counters the tracker already
  maintains (``stuck_seconds``, ``dwell_seconds``, ``near_stuck_seconds``). Nothing here
  looks at raw boxes.
* Only vehicles feed the stuck/dwell logic and only persons feed the pedestrian logic.
  Trains are ignored entirely: a train inside the crossing is normal. ALARM means "on the
  tracks"; a vehicle stopped *too close* to them raises a WARNING (``close_policy``).
* An alarmed vehicle hidden behind a passing train (``ObjectState.occluded``) keeps the
  ALARM: the hazard is still there, only the camera cannot see it.
* When the crossing gates are reported **down**, the ``gates_down_*`` thresholds apply.
  They only escalate when they are shorter than the normal ones; with the 3 s stuck
  ALARM the default gates-down alarm (3 s) no longer shortens it, only the warning (1 s)
  and the dwell alarm (6 s) do, and the engine logs a warning at construction for every
  gates-down threshold that is not shorter. The gate model is not validated on the team
  clips, so the infrastructure detector is off by default.
* Alarms **latch**: once raised, ALARM is held for ``latch_seconds`` after the last
  alarm condition. Without a latch, a tracker dropping an id for a single frame
  would flap the alarm and a downstream consumer could miss it. WARNING has a shorter
  hold (``warning_hold_seconds``) while its objects are still in or near the zone.
* A new cause (a new track id, or debris) joining an active ALARM is announced at once
  with an ``ALARM_UPDATE`` event; without it a second incident would only surface with
  the next repeat notification, up to ``repeat_notify_seconds`` later.
* Notification sinks must never be able to crash or stall the frame loop. The engine
  wraps every sink call, and the webhook sink hands network I/O to a daemon thread.

The module deliberately imports nothing heavier than the standard library and the
package's own types so it can be unit-tested without OpenCV, torch or ultralytics.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import logging
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .config import AlertConfig, DebrisConfig, StuckConfig
from .types import Assessment, DebrisResult, Event, GateState, ObjectState, Status

logger = logging.getLogger(__name__)

_CLOSE_POLICIES = ("off", "warning", "alarm")


# --------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------
def _json_fallback(value: Any) -> Any:
    """``json.dumps`` ``default`` hook: turn numpy scalars/arrays into plain Python.

    Track ids coming out of ultralytics are numpy integers and dwell/stuck counters can
    be numpy floats; the standard encoder refuses them. Rather than crash the monitor
    on the first alarm (the worst possible moment), convert them here.
    """
    if hasattr(value, "tolist"):  # numpy scalar or ndarray
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


# Delivery ids are "<session>-<sequence>": a random session per process (one monitor run)
# and a counter in the order events are first serialised. See event_id().
_EVENT_SESSION = uuid.uuid4().hex[:12]
_EVENT_SEQ_LOCK = threading.Lock()
_event_seq = 0


def event_id(event: Event) -> str:
    """Return the delivery id of ``event``, assigning one the first time it is asked for.

    A producer may set its own id as ``event.extra["event_id"]``. Otherwise the id is
    ``"<session>-<n>"``: random per process, then a sequence number that grows in the
    order events are first serialised (for the webhook, the order they were emitted in).
    It is cached on the event object, outside the dataclass fields, so every sink that
    reports ids reports the same one for the same event and a retried delivery repeats
    it. A receiver that sees an id twice (an attempt that timed out on our side but did
    arrive) can drop the duplicate, and one that receives events out of order (see
    ``WebhookSink``) can order them by ``n``.
    """
    extra = event.extra if isinstance(event.extra, dict) else {}
    if extra.get("event_id"):
        return str(extra["event_id"])
    cached = getattr(event, "_delivery_id", None)
    if cached is not None:
        return cached
    global _event_seq
    with _EVENT_SEQ_LOCK:
        cached = getattr(event, "_delivery_id", None)
        if cached is None:
            _event_seq += 1
            cached = f"{_EVENT_SESSION}-{_event_seq:06d}"
            try:
                setattr(event, "_delivery_id", cached)  # not a field: asdict() ignores it
            except AttributeError:  # pragma: no cover - frozen/slotted look-alike: not cached
                pass
    return cached


def event_to_json(event: Event, include_id: bool = True) -> str:
    """Serialise an ``Event`` as one JSON object (one line, no newline); used by every sink.

    The object is ``dataclasses.asdict(event)`` with numpy values converted, plus a
    top-level ``"event_id"`` (:func:`event_id`) unless ``include_id`` is false. The JSONL
    log uses ``include_id=False``: its lines stay exactly the event fields that
    ``gradecrossing.evaluate`` and external tools read, and a local file is never
    re-delivered, so it needs no de-duplication key.
    """
    data = dataclasses.asdict(event)
    if include_id:
        data["event_id"] = event_id(event)
    return json.dumps(data, default=_json_fallback)


def ascii_safe(text: Any) -> str:
    """Make ``text`` safe for a cp1252/ASCII console (Windows terminals, serial logs).

    Every non-ASCII character becomes ``?``. Use it for anything printed to a console:
    sink lines, reports, overlay labels.
    """
    return str(text).encode("ascii", "replace").decode("ascii")


_ascii = ascii_safe  # former private name, kept for callers that still use it


# --------------------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------------------
class StdoutSink:
    """Print one human-readable ASCII line per event to stdout.

    This is the sink an operator watching the console relies on, so it is kept trivially
    simple: no colours, no unicode, flushed after every line so the message is visible
    even when stdout is a pipe.
    """

    def __init__(self, stream=None) -> None:
        # ``None`` means "resolve sys.stdout at emit time" so tests that capture stdout
        # (pytest capsys) and processes that redirect stdout after construction both work.
        self._stream = stream

    def emit(self, event: Event) -> None:
        stream = self._stream if self._stream is not None else sys.stdout
        reasons = "; ".join(event.reasons) if event.reasons else "-"
        ids = ",".join(str(i) for i in event.track_ids) if event.track_ids else "-"
        extra = " repeat" if event.extra.get("repeat") else ""
        line = (
            f"[t={event.timestamp:9.3f}s f={event.frame_index:7d}] "
            f"{event.type:<13} {event.status:<7} ids={ids} {reasons}{extra}"
        )
        print(ascii_safe(line), file=stream, flush=True)

    def close(self) -> None:
        """Flush the stream. Never closes it (it is usually ``sys.stdout``)."""
        stream = self._stream if self._stream is not None else sys.stdout
        try:
            stream.flush()
        except Exception:  # a closed or broken stream has nothing left to flush
            pass


class JsonlSink:
    """Append every event as one JSON object per line to ``path``.

    The line is exactly ``json.dumps(dataclasses.asdict(event))`` (``event_to_json`` with
    ``include_id=False``, so numpy scalars are converted) so
    ``gradecrossing.evaluate.load_alarm_intervals`` and any external tool can replay the
    run. The file is flushed after every event: if the process dies mid-run the log still
    contains everything up to the crash, which is what you want for an incident review.

    Appending (the default) never destroys the log of an earlier run, and
    ``gradecrossing.evaluate`` scores the last run of the file. A producer marks each run
    with a RUN_START event (the event-log contract in ``gradecrossing.evaluate``); in a log
    without one, timestamps going back to the start mark the re-run.
    """

    def __init__(self, path: str, append: bool = True) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a" if append else "w", encoding="utf-8")
        self.count = 0

    def emit(self, event: Event) -> None:
        if self._fh is None or self._fh.closed:
            logger.warning("JsonlSink %s is closed; dropping %s", self.path, event.type)
            return
        self._fh.write(event_to_json(event, include_id=False) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _is_alarm_type(etype: str) -> bool:
    """ALARM_START / ALARM_UPDATE / ALARM_END: the events a dispatcher must not miss."""
    return str(etype).startswith("ALARM")


def _is_hand_over(event: Event) -> bool:
    """The WARNING_END of a WARNING -> ALARM transition (the engine gives it status ALARM
    and emits the ALARM_START of the same transition right after it)."""
    return str(event.type) == "WARNING_END" and str(event.status) == "ALARM"


# How long the webhook worker waits for the ALARM_START that follows the WARNING_END of a
# WARNING -> ALARM transition (it is emitted microseconds later, after the other sinks ran).
_HAND_OVER_WAIT_SECONDS = 0.25

# WebhookSink.close() waits one socket timeout per queued event by default, at least one
# event's deadline and at most this long (unless that deadline is longer): a slow endpoint
# still gets the closing events, a dead one cannot hold the shutdown up for minutes.
_CLOSE_MAX_SECONDS = 30.0


def _retryable(exc: BaseException) -> bool:
    """Is a failed POST worth another attempt?

    Refused or reset connections, timeouts and HTTP 5xx/408/429 can be transient. Other
    HTTP 4xx answers and malformed URLs fail the same way every time.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code in (408, 429)
    return not isinstance(exc, ValueError)


class _Attempt:
    """The connections of one POST attempt, so that the worker can abort it.

    The attempt runs in a helper thread, which registers each socket as soon as it is
    connected. Once the worker gives up (``abandon``), an open socket is shut down, which
    wakes a blocked send/recv, and a socket that only finishes connecting later (a slow
    DNS answer, a slow TCP or TLS handshake) is closed before the request is written. An
    abandoned attempt therefore never sends a stale copy after newer events.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._socks: List[Any] = []
        self._abandoned = False

    def register(self, sock: Any) -> None:
        """Called by the helper thread once connected; raises if the attempt was abandoned."""
        with self._lock:
            if not self._abandoned:
                self._socks.append(sock)
                return
        try:
            sock.close()
        except OSError:
            pass
        raise TimeoutError("POST attempt abandoned before the request was sent")

    def abandon(self) -> None:
        with self._lock:
            self._abandoned = True
            socks = list(self._socks)
        for sock in socks:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class _SocketTracking:
    """Mixin for urllib's HTTP(S)Handler that registers every connection with ``attempt``.

    ``urlopen``'s timeout bounds each socket operation, not the request: a server that
    trickles one byte per second keeps a POST alive indefinitely. Registering the socket
    lets ``WebhookSink._post`` abort the attempt when the event's deadline passes.
    """

    attempt: _Attempt

    def do_open(self, http_class: Any, req: Any, **http_conn_args: Any) -> Any:
        attempt = self.attempt

        def factory(*args: Any, **kwargs: Any) -> Any:
            conn = http_class(*args, **kwargs)
            connect = conn.connect

            def tracked_connect() -> None:
                connect()
                attempt.register(conn.sock)

            conn.connect = tracked_connect
            return conn

        return super().do_open(factory, req, **http_conn_args)  # type: ignore[misc]


class _TrackingHTTPHandler(_SocketTracking, urllib.request.HTTPHandler):
    pass


_TRACKING_HANDLERS: List[Any] = [_TrackingHTTPHandler]
if hasattr(urllib.request, "HTTPSHandler"):  # absent when Python is built without ssl
    class _TrackingHTTPSHandler(_SocketTracking, urllib.request.HTTPSHandler):
        pass

    _TRACKING_HANDLERS.append(_TrackingHTTPSHandler)


class _Queued(NamedTuple):
    etype: str
    payload: bytes
    repeat: bool
    hand_over: bool = False  # a WARNING_END that waits for its ALARM_START (see _is_hand_over)


class WebhookSink:
    """POST every event as JSON to ``url`` from a background daemon thread.

    Why a thread: the frame loop must keep evaluating the crossing at frame rate even
    when the network is down. Events go onto a bounded queue and a single long-lived
    daemon worker delivers them; ``emit`` returns immediately and never raises.

    Delivery rules (one network blip must not cost the dispatcher an ALARM):

    * Retries: a failed POST (refused or reset connection, timeout, HTTP 5xx/408/429) is
      tried up to ``attempts`` times in all, waiting ``backoff_seconds`` before the first
      retry and doubling the wait after each failure (0.5, 1, 2 s ...). Other HTTP 4xx
      answers and malformed URLs are not retried; they would fail the same way again.
    * Timeouts: ``timeout`` is urllib's: it bounds each socket operation (connect, send,
      every read), so a slow but steady endpoint is never cut off by it. All attempts and
      waits for one event share ``deadline_seconds`` of wall-clock time (default: every
      attempt may time out once, plus the back-off waits; 10.5 s with the defaults), and
      the attempt still running when it passes is aborted even when the server keeps
      trickling bytes. One bad endpoint therefore delays the next event by at most one
      deadline. An aborted attempt sends nothing afterwards (see ``_Attempt``).
    * Priority: queued ALARM_START / ALARM_UPDATE / ALARM_END events are delivered before
      queued HEALTH, WARNING and RUN events, so a backlog of notices cannot hold up an
      alarm. Order is kept within each group, so a consumer never sees an ALARM_END
      before its ALARM_START. One exception keeps the most common alarm fast and
      consistent: the WARNING_END of a WARNING -> ALARM transition (status ALARM) is sent
      right after the ALARM_START of that transition instead of before it, so the
      ALARM_START never waits for it and it still arrives while the alarm is on.
    * Overflow: when ``max_queue`` events are waiting, the oldest queued non-ALARM event
      is dropped to make room (for an incoming ALARM event, then the oldest repeat ALARM
      reminder or transition WARNING_END). The incoming event is dropped only when
      nothing else can go.
    * For receivers: delivery is at-least-once, and when events back up an ALARM event
      overtakes older WARNING, HEALTH and RUN events. Every payload carries a top-level
      ``"event_id"`` (``event_id``) that a retry repeats, so a duplicate whose first
      attempt timed out on our side but did arrive can be dropped. An event's ``status``
      is the status at its own ``timestamp``: a receiver that tracks the current status
      should skip an event older than the last one it applied (by the sequence number at
      the end of a sink-assigned ``event_id``, or by ``timestamp`` within a run).

    * Shutdown: ``close()`` first moves queued HEALTH notices behind the other queued
      events, so the RUN_END and the WARNING_END / ALARM_END of the shutdown are not stuck
      behind start-up notices on a slow endpoint, and then waits up to one ``timeout`` per
      queued event (at least one deadline, at most 30 s unless the deadline is longer; one
      deadline once a delivery fails, so a dead endpoint does not hold the shutdown up).

    Counters (never raised): ``delivered``, ``failures`` (events given up on),
    ``dropped`` and ``retries``. ``flush(timeout)`` waits for the queue to drain;
    ``close(timeout)`` flushes and then refuses new events.
    """

    def __init__(self, url: str, timeout: float = 3.0, max_queue: int = 256,
                 headers: Optional[Dict[str, str]] = None, attempts: int = 3,
                 backoff_seconds: float = 0.5, deadline_seconds: Optional[float] = None) -> None:
        self.url = str(url)
        self.timeout = max(0.1, float(timeout))
        self.attempts = max(1, int(attempts))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        if deadline_seconds is None or float(deadline_seconds) <= 0:
            waits = sum(self.backoff_seconds * (2 ** i) for i in range(self.attempts - 1))
            deadline_seconds = self.attempts * self.timeout + waits
        self.deadline_seconds = float(deadline_seconds)
        self.max_queue = max(1, int(max_queue))
        self.headers = {"Content-Type": "application/json", "User-Agent": "gradecrossing-monitor"}
        if headers:
            self.headers.update(headers)
        self._alarms: Deque[_Queued] = collections.deque()
        self._others: Deque[_Queued] = collections.deque()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._pending = 0  # queued + in flight
        self._closed = False
        self.delivered = 0
        self.failures = 0
        self.dropped = 0
        self.retries = 0

    # -- public -------------------------------------------------------------------
    def emit(self, event: Event) -> None:
        """Queue ``event`` for delivery. Returns immediately; never raises."""
        etype = str(event.type)
        if self._closed:
            self.dropped += 1
            logger.warning("WebhookSink is closed; dropping %s", etype)
            return
        try:
            payload = event_to_json(event).encode("utf-8")
        except Exception:  # pragma: no cover - defensive; _json_fallback handles known cases
            logger.exception("WebhookSink could not serialise %s", etype)
            return
        extra = event.extra if isinstance(event.extra, dict) else {}
        hand_over = _is_hand_over(event)
        item = _Queued(etype, payload, bool(extra.get("repeat")), hand_over)
        priority = hand_over or _is_alarm_type(etype)
        try:
            self._ensure_worker()
        except Exception:  # pragma: no cover - e.g. thread creation failure
            self.dropped += 1
            logger.exception("WebhookSink could not queue %s", etype)
            return
        with self._cond:
            if len(self._alarms) + len(self._others) >= self.max_queue:
                victim = self._make_room(item)
                self.dropped += 1
                if victim is None:
                    logger.warning("WebhookSink queue full (%d); dropping %s", self.max_queue, etype)
                    return
                self._pending -= 1
                logger.warning("WebhookSink queue full (%d); dropping queued %s to keep %s",
                               self.max_queue, victim.etype, etype)
            (self._alarms if priority else self._others).append(item)
            self._pending += 1
            self._cond.notify_all()

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Block until every queued event has been attempted (or ``timeout`` s). True if drained."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._cond:
            while self._pending > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def close(self, timeout: Optional[float] = None) -> bool:
        """Flush for up to ``timeout`` s, then refuse new events.

        Before waiting, queued HEALTH notices go behind the other queued events (their own
        order kept): at shutdown the RUN_END and the WARNING_END / ALARM_END that close the
        run matter more than start-up notices, which used to fill the queue ahead of them on
        a slow endpoint. ALARM events keep their priority and the other events their order,
        so an END still never overtakes its START. The default ``timeout`` is one socket
        ``timeout`` per queued event, at least ``deadline_seconds`` and at most
        ``_CLOSE_MAX_SECONDS`` (30 s; the deadline when that is longer), cut back to one
        deadline as soon as a delivery fails meanwhile: a slow endpoint gets every event,
        a dead one holds the shutdown up no longer than one deadline.

        Returns True when everything queued was attempted. Events still queued after the
        timeout keep being attempted by the daemon worker until the process exits.
        """
        with self._cond:
            if not self._closed and any(q.etype == "HEALTH" for q in self._others):
                rest = [q for q in self._others if q.etype != "HEALTH"]
                health = [q for q in self._others if q.etype == "HEALTH"]
                self._others.clear()
                self._others.extend(rest + health)
            pending = self._pending
        if timeout is None:
            budget = max(self.deadline_seconds, min(pending * self.timeout, _CLOSE_MAX_SECONDS))
            drained = self._flush_until_failing(budget)
        else:
            drained = self.flush(timeout)
        with self._cond:
            self._closed = True
            left = self._pending
            self._cond.notify_all()
        if not drained:
            logger.warning("WebhookSink closed with %d event(s) not yet delivered to %s", left, self.url)
        return drained

    # -- internals ----------------------------------------------------------------
    def _flush_until_failing(self, budget: float) -> bool:
        """``flush(budget)``, cut back to one deadline once a delivery fails meanwhile."""
        start = time.monotonic()
        failures = self.failures
        with self._cond:
            while self._pending > 0:
                limit = self.deadline_seconds if self.failures > failures else budget
                remaining = start + limit - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def _make_room(self, incoming: _Queued) -> Optional[_Queued]:
        """Evict one queued event for ``incoming`` (lock held). None: drop ``incoming``."""
        if self._others:
            return self._others.popleft()
        if _is_alarm_type(incoming.etype):
            for i, queued in enumerate(self._alarms):
                # a reminder of an alarm already announced, or a transition WARNING_END
                if queued.repeat or not _is_alarm_type(queued.etype):
                    del self._alarms[i]
                    return queued
        return None

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._worker, name="gradecrossing-webhook", daemon=True)
                self._thread.start()

    def _alarm_start_first(self, hand_over: _Queued) -> _Queued:
        """Swap a transition WARNING_END with the ALARM_START emitted right after it (lock held).

        The worker can pick up the WARNING_END before the ALARM_START has been queued (the
        other sinks run in between), so it waits up to ``_HAND_OVER_WAIT_SECONDS`` for it.
        The WARNING_END goes back to the front of the ALARM queue, right behind it.
        """
        until = time.monotonic() + _HAND_OVER_WAIT_SECONDS
        while not self._alarms and not self._closed:
            remaining = until - time.monotonic()
            if remaining <= 0:
                break
            self._cond.wait(remaining)
        if self._alarms and self._alarms[0].etype == "ALARM_START":
            first = self._alarms.popleft()
            self._alarms.appendleft(hand_over._replace(hand_over=False))
            return first
        return hand_over

    def _worker(self) -> None:
        while True:
            with self._cond:
                while not self._alarms and not self._others:
                    if self._closed:
                        return
                    self._cond.wait()
                item = self._alarms.popleft() if self._alarms else self._others.popleft()
                if item.hand_over:
                    item = self._alarm_start_first(item)
            try:
                self._deliver(item)
            except Exception:  # pragma: no cover - _deliver handles delivery errors itself
                self.failures += 1
                logger.exception("WebhookSink could not deliver %s", item.etype)
            finally:
                with self._cond:
                    self._pending -= 1
                    self._cond.notify_all()

    def _deliver(self, item: _Queued) -> None:
        """POST one event, retrying with back-off until it succeeds or its deadline passes."""
        deadline = time.monotonic() + self.deadline_seconds
        error: Optional[BaseException] = None
        for attempt in range(1, self.attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._post(item.payload, remaining)
            except Exception as exc:  # network errors, HTTP errors, bad URL: log and go on
                error = exc
                wait = self.backoff_seconds * (2 ** (attempt - 1))
                if attempt >= self.attempts or not _retryable(exc) or time.monotonic() + wait >= deadline:
                    break
                self.retries += 1
                logger.info("WebhookSink POST of %s failed (%s); retry %d of %d in %.1fs",
                            item.etype, exc, attempt, self.attempts - 1, wait)
                time.sleep(wait)
                continue
            self.delivered += 1
            return
        self.failures += 1
        logger.warning("WebhookSink POST to %s failed: %s", self.url,
                       error if error is not None else "deadline expired")

    def _post(self, payload: bytes, limit: Optional[float] = None) -> None:
        """One POST attempt: ``timeout`` per socket operation, aborted after ``limit``
        seconds of wall-clock time (default: one per-event deadline)."""
        limit = self.deadline_seconds if limit is None else max(0.001, float(limit))
        attempt = _Attempt()
        errors: List[BaseException] = []

        def run() -> None:
            try:
                handlers = []
                for cls in _TRACKING_HANDLERS:
                    handler = cls()
                    handler.attempt = attempt
                    handlers.append(handler)
                opener = urllib.request.build_opener(*handlers)
                req = urllib.request.Request(self.url, data=payload, headers=self.headers, method="POST")
                with opener.open(req, timeout=min(self.timeout, limit)) as resp:
                    resp.read()  # drain so the connection can be closed cleanly
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, name="gradecrossing-webhook-post", daemon=True)
        thread.start()
        thread.join(limit)
        if thread.is_alive():
            attempt.abandon()  # wakes a blocked send/recv; a late connection sends nothing
            raise TimeoutError(f"no complete response within {limit:.1f}s")
        if errors:
            raise errors[0]


class CallbackSink:
    """Call ``fn(event)`` for every event. Handy for tests, GUIs and custom integrations."""

    def __init__(self, fn: Callable[[Event], None]) -> None:
        if not callable(fn):
            raise TypeError("CallbackSink needs a callable")
        self._fn = fn

    def emit(self, event: Event) -> None:
        self._fn(event)

    def close(self) -> None:
        """Nothing to release; present so every sink can be closed the same way."""


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------
def _vehicle_reason(tid: int, label: str, stuck: float, dwell: float, stuck_threshold: float,
                    gate_note: str) -> str:
    """Reason for a vehicle on the tracks: its stop when that reached the tier, else its dwell."""
    if stuck >= stuck_threshold:
        return f"vehicle {tid} ({label}) stationary in zone {stuck:.1f}s{gate_note}"
    return f"vehicle {tid} ({label}) in zone {dwell:.1f}s{gate_note}"


def _object_summary(obj: ObjectState) -> Dict[str, Any]:
    """Plain-JSON description of one tracked object, for an event's ``extra['objects']``."""
    dist = float(obj.zone_distance_bh)
    out: Dict[str, Any] = {
        "id": int(obj.track_id),
        "class": str(obj.cls_name),
        "box": [round(float(v), 1) for v in obj.last_box],
        "in_zone": bool(obj.in_zone),
        "near_zone": bool(obj.near_zone),
        "zone_overlap": round(float(obj.zone_overlap), 3),
        "zone_distance_bh": round(dist, 3) if dist < float("inf") else None,
        "dwell": round(float(obj.dwell_seconds), 2),
        "stuck": round(float(obj.stuck_seconds), 2),
        "still": round(float(obj.still_seconds), 2),
        "speed": round(float(obj.speed), 2),
    }
    if obj.occluded:
        out["occluded"] = True
    return out


def _cause_key(obj: Optional[ObjectState], tid: int) -> Tuple[str, int]:
    """Identity of an alarm cause: the physical track (``ObjectState.uid``) when known, so a
    vehicle the tracker re-identified under a new id is not announced as a new cause."""
    if obj is not None and int(obj.uid) > 0:
        return ("uid", int(obj.uid))
    return ("id", int(tid))


@dataclasses.dataclass
class _Episode:
    """The WARNING or ALARM episode in progress: what its END event reports."""

    status: Status
    start: float
    ids: List[int]  # every track id that was at this severity during the episode
    first_reason: str
    debris: bool = False
    keys: List[Tuple[str, int]] = dataclasses.field(default_factory=list)  # causes seen (see _cause_key)


class AlertEngine:
    """Combine tracker, debris and gate information into a CLEAR/WARNING/ALARM decision.

    Call ``evaluate`` once per processed frame. It returns the ``Assessment`` for that
    frame plus the list of ``Event`` objects generated by status transitions (already
    delivered to every registered sink). The engine is the only place in the package
    that holds alert state (current status, latch expiry, WARNING hold, the episode in
    progress, repeat-notification timer), so ``reset`` must be called when a new video
    starts.

    Vehicle tiers, in order (``warn_s``/``alarm_s`` from the gate state):

    1. in the zone and stuck >= alarm_s, or dwell >= the dwell alarm -> ALARM
    2. ``close_policy == "alarm"`` and stopped on-or-near the zone >= alarm_s -> ALARM
    3. in the zone and stuck >= warn_s, or dwell >= the dwell warning -> WARNING
    4. ``close_policy != "off"`` and stopped on-or-near the zone >= warn_s -> WARNING
       ("stopped near the tracks")

    A hidden hazard (an alarmed vehicle occluded by a train) stays at ALARM.
    """

    def __init__(self, stuck_cfg: StuckConfig, alert_cfg: AlertConfig, *legacy: Any) -> None:
        if legacy:
            # Pre-port signature AlertEngine(stuck_cfg, debris_cfg, alert_cfg): the debris
            # config was stored and never read. Still accepted, with a log line, so code
            # written against it keeps working while it migrates.
            if len(legacy) != 1 or not isinstance(alert_cfg, DebrisConfig):
                raise TypeError("AlertEngine(stuck_cfg, alert_cfg)")
            logger.warning("AlertEngine(stuck_cfg, debris_cfg, alert_cfg) is deprecated (the debris config "
                           "was never used): call AlertEngine(stuck_cfg, alert_cfg)")
            alert_cfg = legacy[0]
        self.stuck_cfg = stuck_cfg
        self.alert_cfg = alert_cfg
        if stuck_cfg.close_policy not in _CLOSE_POLICIES:
            raise ValueError(f"stuck.close_policy must be one of {_CLOSE_POLICIES}, got {stuck_cfg.close_policy!r}")
        if float(alert_cfg.warning_hold_seconds) < 0.0:
            raise ValueError("alerts.warning_hold_seconds must be >= 0")
        if stuck_cfg.close_policy != "off" and stuck_cfg.zone_test == "bottom_center":
            logger.warning("stuck.close_policy=%r has no effect with stuck.zone_test='bottom_center': "
                           "no vehicle is ever 'near' the zone", stuck_cfg.close_policy)
        self._warn_if_gates_down_not_shorter()
        self._sinks: List[Any] = []
        self._status: Status = Status.CLEAR
        self._prev: Optional[Assessment] = None
        self._latched_until: Optional[float] = None
        self._last_alarm_notify: Optional[float] = None
        self._episode: Optional[_Episode] = None
        self._last_warning_t: Optional[float] = None

    def _warn_if_gates_down_not_shorter(self) -> None:
        """Say which gates-down thresholds do not escalate anything (validated nowhere yet)."""
        cfg = self.stuck_cfg
        pairs = (("gates_down_warning_seconds", cfg.gates_down_warning_seconds, "warning_seconds", cfg.warning_seconds),
                 ("gates_down_alarm_seconds", cfg.gates_down_alarm_seconds, "alarm_seconds", cfg.alarm_seconds),
                 ("gates_down_dwell_alarm_seconds", cfg.gates_down_dwell_alarm_seconds,
                  "dwell_alarm_seconds", cfg.dwell_alarm_seconds))
        same = [f"{gd_name} {float(gd):g} s >= {name} {float(v):g} s"
                for gd_name, gd, name, v in pairs if float(gd) >= float(v)]
        if same:
            logger.warning("gates-down thresholds not shorter than the normal ones (%s): gates down does "
                           "not escalate those tiers", "; ".join(same))

    # -- sinks --------------------------------------------------------------------
    def add_sink(self, sink) -> None:
        """Register an object with an ``emit(event)`` method. Events go to every sink."""
        if not hasattr(sink, "emit") or not callable(getattr(sink, "emit")):
            raise TypeError("sink must have an emit(event) method")
        self._sinks.append(sink)

    @property
    def sinks(self) -> Tuple[Any, ...]:
        return tuple(self._sinks)

    def emit(self, event: Event) -> None:
        """Deliver an externally created event (e.g. a HEALTH event) to all sinks.

        A failing sink is logged and skipped: a broken webhook or a full disk must not
        stop the monitor from evaluating the next frame or from reaching the other sinks.
        """
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                logger.exception("sink %s failed on %s", type(sink).__name__, event.type)

    # -- state --------------------------------------------------------------------
    @property
    def status(self) -> Status:
        """Status produced by the most recent ``evaluate`` (CLEAR before the first call)."""
        return self._status

    @property
    def latched_until(self) -> Optional[float]:
        return self._latched_until

    @property
    def alarm_track_ids(self) -> List[int]:
        """Track ids at ALARM on the most recent ``evaluate`` (raw: a latched ALARM has none)."""
        return list(self._prev.alarm_track_ids) if self._prev is not None else []

    @property
    def alarm_episode_track_ids(self) -> List[int]:
        """The ids the tracker treats as alarmed: :attr:`alarm_track_ids`, plus every id of the
        ALARM episode in progress while the status is ALARM (latched included).

        The tracker keeps these behind a train (hidden hazards) and keeps their vehicle label.
        The raw ids alone are not enough: a train usually covers a stopped car gradually, and
        while it does the car's box is cut short, which restarts its still and speed clocks.
        By the time the car is lost under the train only the latch holds the ALARM, and with
        the raw ids the car was dropped and the ALARM ended with the latch while the train was
        still over it (work/rev-missed-alarms/hidden_probe.py 'partial': ALARM_END 18.6 s, CLEAR
        from 20 s with the car under the train until 30 s).
        """
        ids = self.alarm_track_ids
        ep = self._episode
        if self._status is Status.ALARM and ep is not None and ep.status is Status.ALARM:
            ids += [i for i in ep.ids if i not in ids]
        return ids

    def reset(self) -> None:
        """Forget all alert state (new video / reconnect). Emits nothing; see ``close_open_episodes``."""
        self._status = Status.CLEAR
        self._prev = None
        self._latched_until = None
        self._last_alarm_notify = None
        self._episode = None
        self._last_warning_t = None

    def close_open_episodes(self, timestamp: float, frame_index: int = -1,
                            reason: str = "pipeline re-initialised") -> List[Event]:
        """End the WARNING or ALARM episode in progress, if any, then ``reset``.

        For a pipeline re-initialisation (a live stream that changed geometry, a looped
        file): the tracks, the latch and the episode are discarded, so the episode could
        never end on its own and a receiver would keep showing it forever. The END event
        has the usual payload (episode track ids, opening reason, ``episode_start``) with
        status CLEAR, no ``objects`` (nothing is tracked any more) and ``extra['reason']``.
        It is delivered to every sink and returned; nothing is emitted when the status is
        CLEAR. If the hazard is still there, the re-initialised monitor raises it again
        once its clocks have run up.
        """
        events: List[Event] = []
        prev = self._status
        if prev is not Status.CLEAR:
            ep = self._episode
            extra: Dict[str, object] = {"objects": [], "reason": str(reason)}
            if ep is not None:
                extra["episode_start"] = ep.start
            events.append(Event(type="ALARM_END" if prev is Status.ALARM else "WARNING_END",
                                timestamp=float(timestamp), frame_index=int(frame_index),
                                status=Status.CLEAR.value,
                                reasons=[ep.first_reason] if ep is not None and ep.first_reason else [],
                                track_ids=[int(i) for i in ep.ids] if ep is not None else [],
                                extra=extra))
        self.reset()
        for event in events:
            self.emit(event)
        return events

    # -- decision -----------------------------------------------------------------
    def _thresholds(self, gates_down: bool) -> Tuple[float, float, float, float]:
        """Return (warn, alarm, dwell_warning, dwell_alarm) seconds for the gate state.

        With the gates down a train is imminent, so stationary and dwelling vehicles are
        escalated on the ``gates_down_*`` timers (where those are shorter). The dwell
        *warning* has no gate-specific override in the config and is used unchanged.
        """
        cfg = self.stuck_cfg
        if gates_down:
            return (float(cfg.gates_down_warning_seconds), float(cfg.gates_down_alarm_seconds),
                    float(cfg.dwell_warning_seconds), float(cfg.gates_down_dwell_alarm_seconds))
        return (float(cfg.warning_seconds), float(cfg.alarm_seconds),
                float(cfg.dwell_warning_seconds), float(cfg.dwell_alarm_seconds))

    def _vehicle_tier(self, obj: ObjectState, tid: int, label: str, warn_s: float, alarm_s: float,
                      dwell_warn_s: float, dwell_alarm_s: float, gate_note: str) -> Tuple[Optional[Status], str]:
        """The severity one vehicle raises this frame (None: nothing) and its reason."""
        stuck = float(obj.stuck_seconds)
        dwell = float(obj.dwell_seconds)
        if obj.occluded:
            return Status.ALARM, f"vehicle {tid} ({label}) occluded by train, stationary in zone {stuck:.1f}s{gate_note}"
        on = bool(obj.in_zone)
        # near_stuck: stopped while on the zone OR too close to it (the team's "merged"
        # region). close_policy decides what it may raise on its own.
        near_stuck = float(obj.near_stuck_seconds)
        policy = self.stuck_cfg.close_policy
        if on and (stuck >= alarm_s or dwell >= dwell_alarm_s):
            return Status.ALARM, _vehicle_reason(tid, label, stuck, dwell, alarm_s, gate_note)
        if policy == "alarm" and near_stuck >= alarm_s:
            return Status.ALARM, f"vehicle {tid} ({label}) stopped on/near the tracks {near_stuck:.1f}s{gate_note}"
        if on and (stuck >= warn_s or dwell >= dwell_warn_s):
            return Status.WARNING, _vehicle_reason(tid, label, stuck, dwell, warn_s, gate_note)
        if policy != "off" and near_stuck >= warn_s:
            return Status.WARNING, f"vehicle {tid} ({label}) stopped near the tracks {near_stuck:.1f}s{gate_note}"
        return None, ""

    def _episode_present(self, objects: Sequence[ObjectState]) -> bool:
        """True while one of the warning episode's objects is still in, near or hidden in the zone."""
        ep = self._episode
        if ep is None or not ep.ids:
            return False
        ids = set(ep.ids)
        return any(int(o.track_id) in ids and (o.in_zone or o.near_zone or o.occluded) for o in objects)

    def evaluate(self, objects: List[ObjectState], debris: Optional[DebrisResult], gate: GateState,
                 timestamp: float, frame_index: int) -> Tuple[Assessment, List[Event]]:
        """Decide the status for this frame and emit transition events.

        Steps: pick thresholds from the gate state, classify each vehicle/person by its
        counters, fold in the debris trigger, compute the raw status, apply the latch and
        the WARNING hold, then diff against the previous status to produce events.
        """
        t = float(timestamp)
        gates_down = bool(gate is not None and gate.gates_down)
        warn_s, alarm_s, dwell_warn_s, dwell_alarm_s = self._thresholds(gates_down)
        base_alarm = float(self.stuck_cfg.alarm_seconds)
        severity_scale = (alarm_s / base_alarm) if base_alarm > 0 else 1.0
        gate_note = " (gates down)" if gates_down else ""

        alarm_ids: List[int] = []
        warning_ids: List[int] = []
        alarm_reasons: List[str] = []
        warning_reasons: List[str] = []
        reason_by_id: Dict[int, str] = {}
        person_alert = False

        # Sort by id so reasons/ids are deterministic regardless of tracker ordering.
        for obj in sorted(objects, key=lambda o: int(o.track_id)):
            if obj.is_train:
                continue  # a train in the crossing is the normal case, never an alert
            vehicle = bool(obj.is_vehicle)
            if not (obj.in_zone or (vehicle and (obj.near_zone or obj.occluded))):
                # By contract dwell/stuck are 0 outside the zone; this guard additionally
                # protects against a (mis)configured threshold of 0 s alarming on traffic
                # that never came near the crossing. Persons must be in the zone.
                continue
            tid = int(obj.track_id)
            if tid in reason_by_id:
                continue
            label = f"{obj.cls_name}" if obj.cls_name else "object"
            if vehicle:
                level, reason = self._vehicle_tier(obj, tid, label, warn_s, alarm_s, dwell_warn_s, dwell_alarm_s,
                                                   gate_note)
            elif obj.is_person:
                # Pedestrian thresholds are dwell-based: a person standing on the tracks
                # is the concern; they are not expected to move at vehicle speeds.
                dwell = float(obj.dwell_seconds)
                level, reason = None, f"person {tid} in zone {dwell:.1f}s"
                if dwell >= float(self.stuck_cfg.person_alarm_seconds):
                    level = Status.ALARM
                elif dwell >= float(self.stuck_cfg.person_warning_seconds):
                    level = Status.WARNING
                person_alert = person_alert or level is not None
            else:
                continue
            if level is Status.ALARM:
                alarm_ids.append(tid)
                alarm_reasons.append(reason)
            elif level is Status.WARNING:
                warning_ids.append(tid)
                warning_reasons.append(reason)
            else:
                continue
            reason_by_id[tid] = reason

        debris_alarm = bool(debris is not None and debris.triggered)
        debris_reason = ""
        if debris_alarm:
            debris_reason = (f"debris {100.0 * float(debris.area_fraction):.1f}% of zone for "
                             f"{float(debris.persist_seconds):.1f}s")
            alarm_reasons.append(debris_reason)

        # Raw (un-latched) status for this frame.
        if alarm_ids or debris_alarm:
            raw = Status.ALARM
        elif warning_ids:
            raw = Status.WARNING
        else:
            raw = Status.CLEAR
        reasons = alarm_reasons + warning_reasons

        # Latch: an ALARM condition arms/extends the latch; once the condition goes away
        # the status is held at ALARM until the latch expires. This stops a one-frame
        # tracker dropout from clearing an alarm that a human still needs to act on.
        latch_s = float(self.alert_cfg.latch_seconds)
        if raw is Status.ALARM:
            self._latched_until = t + latch_s
            status = Status.ALARM
        elif self._latched_until is not None and t < self._latched_until:
            status = Status.ALARM
            reasons = ["latched"] + reasons
        else:
            self._latched_until = None
            status = raw

        # WARNING hold: a warned vehicle that is still where it was keeps the WARNING when
        # its clock blips (a speed spike, a label flicker) instead of flapping END/START.
        hold_s = float(self.alert_cfg.warning_hold_seconds)
        if raw is Status.WARNING:
            self._last_warning_t = t
        elif (status is Status.CLEAR and self._status is Status.WARNING and hold_s > 0.0
              and self._last_warning_t is not None and t - self._last_warning_t < hold_s
              and self._episode_present(objects)):
            status = Status.WARNING
            reasons = ["held"] + reasons

        assessment = Assessment(
            status=status,
            reasons=reasons,
            warning_track_ids=warning_ids,
            alarm_track_ids=alarm_ids,
            debris_alarm=debris_alarm,
            person_alert=person_alert,
            gates_down=gates_down,
            latched_until=self._latched_until,
            severity_scale=float(severity_scale),
        )

        events = self._transition_events(assessment, t, int(frame_index), objects, alarm_reasons,
                                         warning_reasons, reason_by_id, debris_reason)
        for event in events:
            self.emit(event)
        self._prev = assessment
        self._status = status
        return assessment, events

    # -- events -------------------------------------------------------------------
    @staticmethod
    def _event(etype: str, assessment: Assessment, t: float, frame_index: int, reasons: Sequence[str],
               track_ids: Sequence[int], extra: Optional[Dict[str, object]] = None) -> Event:
        return Event(
            type=etype,
            timestamp=t,
            frame_index=frame_index,
            status=assessment.status.value,
            reasons=list(reasons),
            track_ids=[int(i) for i in track_ids],
            extra=dict(extra) if extra else {},
        )

    @staticmethod
    def _objects(by_id: Dict[int, ObjectState], ids: Sequence[int]) -> List[Dict[str, Any]]:
        return [_object_summary(by_id[i]) for i in ids if i in by_id]

    def _transition_events(self, new: Assessment, t: float, frame_index: int, objects: Sequence[ObjectState],
                           alarm_reasons: Sequence[str], warning_reasons: Sequence[str],
                           reason_by_id: Dict[int, str], debris_reason: str) -> List[Event]:
        """Diff the new status against the previous one and build the event list.

        Transitions:
            CLEAR   -> WARNING : WARNING_START
            CLEAR   -> ALARM   : ALARM_START
            WARNING -> ALARM   : WARNING_END, ALARM_START
            WARNING -> CLEAR   : WARNING_END
            ALARM   -> WARNING : ALARM_END, WARNING_START
            ALARM   -> CLEAR   : ALARM_END
            ALARM   -> ALARM   : ALARM_UPDATE (extra: new_track_ids, new_reasons) when a new
                                 cause - a physical track (a vehicle re-identified under a
                                 new id is not new) or debris - joins the episode, and
                                 ALARM_START with extra={"repeat": True} every
                                 ``repeat_notify_seconds`` (0 disables)

        Payloads. An ALARM event's ``reasons`` are the alarm-level reasons of the frame
        (["latched"] while only the latch holds it); the frame's warning-level reasons go
        to ``extra['warning_reasons']``. START, UPDATE and END events carry
        ``extra['objects']`` for the tracks involved. END events carry the episode's track
        ids (every id at that severity during the episode, even when the latch outlived
        them) and the reason that opened it, plus ``extra['episode_start']``.
        """
        prev_status = self._status
        by_id = {int(o.track_id): o for o in objects}
        events: List[Event] = []
        if new.status is Status.ALARM:
            sev_reasons = list(alarm_reasons) or ["latched"]
            extra_base: Dict[str, object] = {"warning_reasons": list(warning_reasons)} if warning_reasons else {}
        else:
            sev_reasons, extra_base = list(new.reasons), {}

        if new.status is prev_status:
            ep = self._episode
            if new.status is Status.ALARM:
                if ep is not None:
                    new_ids = []
                    for i in new.alarm_track_ids:
                        key = _cause_key(by_id.get(i), i)
                        if key not in ep.keys:
                            ep.keys.append(key)
                            new_ids.append(i)
                        if i not in ep.ids:
                            ep.ids.append(i)  # a re-identified cause is reported at the END, not announced
                    new_debris = bool(new.debris_alarm) and not ep.debris
                    if new_ids or new_debris:
                        ep.debris = ep.debris or new_debris
                        new_reasons = [reason_by_id[i] for i in new_ids] + ([debris_reason] if new_debris else [])
                        events.append(self._event("ALARM_UPDATE", new, t, frame_index, sev_reasons,
                                                  new.alarm_track_ids,
                                                  dict(extra_base, new_track_ids=new_ids, new_reasons=new_reasons,
                                                       objects=self._objects(by_id, new_ids))))
                repeat_s = float(self.alert_cfg.repeat_notify_seconds)
                if repeat_s > 0 and self._last_alarm_notify is not None \
                        and t - self._last_alarm_notify >= repeat_s:
                    events.append(self._event("ALARM_START", new, t, frame_index, sev_reasons,
                                              new.alarm_track_ids, dict(extra_base, repeat=True)))
                    self._last_alarm_notify = t
            elif new.status is Status.WARNING and ep is not None:
                ep.ids.extend(i for i in new.warning_track_ids if i not in ep.ids)
            return events

        # Close whatever was open.
        ep = self._episode
        if prev_status is not Status.CLEAR:
            ids = list(ep.ids) if ep is not None else []
            extra: Dict[str, object] = {"objects": self._objects(by_id, ids)}
            if ep is not None:
                extra["episode_start"] = ep.start
            first = [ep.first_reason] if ep is not None and ep.first_reason else []
            etype = "ALARM_END" if prev_status is Status.ALARM else "WARNING_END"
            events.append(self._event(etype, new, t, frame_index, first, ids, extra))
            if prev_status is Status.ALARM:
                self._last_alarm_notify = None
        self._episode = None

        # Open the new one.
        if new.status is Status.WARNING:
            ids = list(new.warning_track_ids)
            self._episode = _Episode(Status.WARNING, t, ids, sev_reasons[0] if sev_reasons else "")
            events.append(self._event("WARNING_START", new, t, frame_index, sev_reasons, ids,
                                      {"objects": self._objects(by_id, ids)}))
        elif new.status is Status.ALARM:
            ids = list(new.alarm_track_ids)
            self._episode = _Episode(Status.ALARM, t, ids, alarm_reasons[0] if alarm_reasons else "",
                                     debris=bool(new.debris_alarm),
                                     keys=[_cause_key(by_id.get(i), i) for i in ids])
            events.append(self._event("ALARM_START", new, t, frame_index, sev_reasons, ids,
                                      dict(extra_base, objects=self._objects(by_id, ids))))
            self._last_alarm_notify = t
        return events


def build_sinks(alert_cfg: AlertConfig) -> List[Any]:
    """Create the sinks enabled in ``AlertConfig`` (stdout, JSONL file, webhook).

    Convenience for ``Monitor``/CLI code so the mapping from config to sinks lives next
    to the sink classes. Returns an empty list when everything is disabled.
    """
    sinks: List[Any] = []
    if alert_cfg.stdout:
        sinks.append(StdoutSink())
    if alert_cfg.events_jsonl:
        sinks.append(JsonlSink(alert_cfg.events_jsonl))
    if alert_cfg.webhook_url:
        sinks.append(WebhookSink(alert_cfg.webhook_url, timeout=alert_cfg.webhook_timeout_seconds,
                                 max_queue=alert_cfg.webhook_max_queue,
                                 attempts=alert_cfg.webhook_attempts,
                                 backoff_seconds=alert_cfg.webhook_backoff_seconds,
                                 deadline_seconds=alert_cfg.webhook_deadline_seconds))
    return sinks


__all__ = [
    "AlertEngine",
    "StdoutSink",
    "JsonlSink",
    "WebhookSink",
    "CallbackSink",
    "build_sinks",
    "event_to_json",
    "event_id",
    "ascii_safe",
]
