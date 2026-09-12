"""Alert decision logic and notification sinks.

This module turns the per-object measurements produced by ``StuckVehicleTracker``,
the ``DebrisResult`` from ``DebrisDetector`` and the ``GateState`` from the
infrastructure detector into a single safety ``Assessment`` (CLEAR / WARNING / ALARM)
per frame, and emits ``Event`` objects on status transitions.

Design rules that matter for safety (see docs/ARCHITECTURE.md):

* Decisions are made **per tracked object** using the counters the tracker already
  maintains (``stuck_seconds``, ``dwell_seconds``). Nothing here looks at raw boxes.
* Only vehicles feed the stuck/dwell logic and only persons feed the pedestrian logic.
  Trains are ignored entirely: a train inside the crossing is normal.
* When the crossing gates are reported **down**, thresholds are shortened. A vehicle
  that is stationary on the tracks while a train is approaching is the most dangerous
  case the system exists for, so it must be reported quickly.
* Alarms **latch**: once raised, ALARM is held for ``latch_seconds`` after the last
  alarm condition. Without a latch, a tracker dropping an id for a single frame
  would flap the alarm and a downstream consumer could miss it.
* Notification sinks must never be able to crash or stall the frame loop. The engine
  wraps every sink call, and the webhook sink hands network I/O to a daemon thread.

The module deliberately imports nothing heavier than the standard library and the
package's own types so it can be unit-tested without OpenCV, torch or ultralytics.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .config import AlertConfig, DebrisConfig, StuckConfig
from .types import Assessment, DebrisResult, Event, GateState, ObjectState, Status

logger = logging.getLogger(__name__)

# Ordering used for "is this status more severe" comparisons.
_SEVERITY = {Status.CLEAR: 0, Status.WARNING: 1, Status.ALARM: 2}


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


def event_to_json(event: Event) -> str:
    """Serialise an ``Event`` exactly the way the JSONL sink does (one line, no newline)."""
    return json.dumps(dataclasses.asdict(event), default=_json_fallback)


def _ascii(text: str) -> str:
    """Make a string safe for a cp1252/ASCII console (Windows terminals, serial logs)."""
    return text.encode("ascii", "replace").decode("ascii")


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
        print(_ascii(line), file=stream, flush=True)


class JsonlSink:
    """Append every event as one JSON object per line to ``path``.

    The line is exactly ``json.dumps(dataclasses.asdict(event))`` (with a fallback for
    numpy scalars) so ``gradecrossing.evaluate.load_alarm_intervals`` and any external
    tool can replay the run. The file is flushed after every event: if the process dies
    mid-run the log still contains everything up to the crash, which is what you want
    for an incident review.
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
        self._fh.write(json.dumps(dataclasses.asdict(event), default=_json_fallback) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class WebhookSink:
    """POST every event as JSON to ``url`` from a background daemon thread.

    Why a thread: the frame loop must keep evaluating the crossing at frame rate even
    when the network is down. ``urllib`` can block for ``timeout`` seconds per request,
    which at 30 fps would freeze the monitor for ~90 frames. Events are pushed onto a
    bounded queue and a single long-lived daemon worker delivers them in order (so a
    consumer never sees ALARM_END before ALARM_START). If the queue is full the event is
    dropped with a log line rather than blocking.

    Failures are counted and logged, never raised. ``flush(timeout)`` waits for the queue
    to drain (used by tests and by a graceful shutdown).
    """

    def __init__(self, url: str, timeout: float = 3.0, max_queue: int = 256,
                 headers: Optional[Dict[str, str]] = None) -> None:
        self.url = str(url)
        self.timeout = float(timeout)
        self.headers = {"Content-Type": "application/json", "User-Agent": "gradecrossing-monitor"}
        if headers:
            self.headers.update(headers)
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=max(1, int(max_queue)))
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._pending = 0
        self.delivered = 0
        self.failures = 0
        self.dropped = 0

    # -- public -------------------------------------------------------------------
    def emit(self, event: Event) -> None:
        """Queue ``event`` for delivery. Returns immediately; never raises."""
        try:
            payload = json.dumps(dataclasses.asdict(event), default=_json_fallback).encode("utf-8")
        except Exception:  # pragma: no cover - defensive; _json_fallback handles known cases
            logger.exception("WebhookSink could not serialise %s", event.type)
            return
        try:
            self._ensure_worker()
            with self._cond:
                self._queue.put_nowait(payload)
                self._pending += 1
        except queue.Full:
            self.dropped += 1
            logger.warning("WebhookSink queue full (%d); dropping %s", self._queue.maxsize, event.type)
        except Exception:  # pragma: no cover - e.g. thread creation failure
            self.dropped += 1
            logger.exception("WebhookSink could not queue %s", event.type)

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

    # -- internals ----------------------------------------------------------------
    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._worker, name="gradecrossing-webhook", daemon=True)
                self._thread.start()

    def _worker(self) -> None:
        while True:
            payload = self._queue.get()
            try:
                self._post(payload)
                self.delivered += 1
            except Exception as exc:  # network errors, HTTP errors, bad URL: log and go on
                self.failures += 1
                logger.warning("WebhookSink POST to %s failed: %s", self.url, exc)
            finally:
                with self._cond:
                    self._pending -= 1
                    self._cond.notify_all()

    def _post(self, payload: bytes) -> None:
        req = urllib.request.Request(self.url, data=payload, headers=self.headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp.read()  # drain so the connection can be reused/closed cleanly


class CallbackSink:
    """Call ``fn(event)`` for every event. Handy for tests, GUIs and custom integrations."""

    def __init__(self, fn: Callable[[Event], None]) -> None:
        if not callable(fn):
            raise TypeError("CallbackSink needs a callable")
        self._fn = fn

    def emit(self, event: Event) -> None:
        self._fn(event)


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------
class AlertEngine:
    """Combine tracker, debris and gate information into a CLEAR/WARNING/ALARM decision.

    Call ``evaluate`` once per processed frame. It returns the ``Assessment`` for that
    frame plus the list of ``Event`` objects generated by status transitions (already
    delivered to every registered sink). The engine is the only place in the package
    that holds alert state (current status, latch expiry, repeat-notification timer), so
    ``reset`` must be called when a new video starts.
    """

    def __init__(self, stuck_cfg: StuckConfig, debris_cfg: DebrisConfig, alert_cfg: AlertConfig) -> None:
        self.stuck_cfg = stuck_cfg
        self.debris_cfg = debris_cfg
        self.alert_cfg = alert_cfg
        self._sinks: List[Any] = []
        self._status: Status = Status.CLEAR
        self._prev: Optional[Assessment] = None
        self._latched_until: Optional[float] = None
        self._last_alarm_notify: Optional[float] = None

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

    def reset(self) -> None:
        """Forget all alert state (new video / reconnect). Emits nothing."""
        self._status = Status.CLEAR
        self._prev = None
        self._latched_until = None
        self._last_alarm_notify = None

    # -- decision -----------------------------------------------------------------
    def _thresholds(self, gates_down: bool) -> Tuple[float, float, float, float]:
        """Return (warn, alarm, dwell_warning, dwell_alarm) seconds for the gate state.

        With the gates down a train is imminent, so stationary and dwelling vehicles are
        escalated on the much shorter ``gates_down_*`` timers. The dwell *warning* has no
        gate-specific override in the config and is used unchanged.
        """
        cfg = self.stuck_cfg
        if gates_down:
            return (float(cfg.gates_down_warning_seconds), float(cfg.gates_down_alarm_seconds),
                    float(cfg.dwell_warning_seconds), float(cfg.gates_down_dwell_alarm_seconds))
        return (float(cfg.warning_seconds), float(cfg.alarm_seconds),
                float(cfg.dwell_warning_seconds), float(cfg.dwell_alarm_seconds))

    def evaluate(self, objects: List[ObjectState], debris: Optional[DebrisResult], gate: GateState,
                 timestamp: float, frame_index: int) -> Tuple[Assessment, List[Event]]:
        """Decide the status for this frame and emit transition events.

        Steps: pick thresholds from the gate state, classify each vehicle/person by its
        counters, fold in the debris trigger, compute the raw status, apply the latch,
        then diff against the previous status to produce events.
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
        person_alert = False

        # Sort by id so reasons/ids are deterministic regardless of tracker ordering.
        for obj in sorted(objects, key=lambda o: int(o.track_id)):
            if obj.is_train:
                continue  # a train in the crossing is the normal case, never an alert
            if not obj.in_zone:
                # By contract dwell/stuck are 0 outside the zone; this guard additionally
                # protects against a (mis)configured threshold of 0 s alarming on traffic
                # that never entered the crossing.
                continue
            tid = int(obj.track_id)
            if tid in alarm_ids or tid in warning_ids:
                continue
            stuck = float(obj.stuck_seconds)
            dwell = float(obj.dwell_seconds)
            label = f"{obj.cls_name}" if obj.cls_name else "object"

            if obj.is_vehicle:
                if stuck >= alarm_s or dwell >= dwell_alarm_s:
                    alarm_ids.append(tid)
                    if stuck >= alarm_s:
                        alarm_reasons.append(
                            f"vehicle {tid} ({label}) stationary in zone {stuck:.1f}s{gate_note}")
                    else:
                        alarm_reasons.append(f"vehicle {tid} ({label}) in zone {dwell:.1f}s{gate_note}")
                elif stuck >= warn_s or dwell >= dwell_warn_s:
                    warning_ids.append(tid)
                    if stuck >= warn_s:
                        warning_reasons.append(
                            f"vehicle {tid} ({label}) stationary in zone {stuck:.1f}s{gate_note}")
                    else:
                        warning_reasons.append(f"vehicle {tid} ({label}) in zone {dwell:.1f}s{gate_note}")
            elif obj.is_person:
                # Pedestrian thresholds are dwell-based: a person standing on the tracks
                # is the concern; they are not expected to move at vehicle speeds.
                if dwell >= float(self.stuck_cfg.person_alarm_seconds):
                    alarm_ids.append(tid)
                    alarm_reasons.append(f"person {tid} in zone {dwell:.1f}s")
                    person_alert = True
                elif dwell >= float(self.stuck_cfg.person_warning_seconds):
                    warning_ids.append(tid)
                    warning_reasons.append(f"person {tid} in zone {dwell:.1f}s")
                    person_alert = True

        debris_alarm = bool(debris is not None and debris.triggered)
        if debris_alarm:
            alarm_reasons.append(
                f"debris {100.0 * float(debris.area_fraction):.1f}% of zone for {float(debris.persist_seconds):.1f}s")

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

        events = self._transition_events(assessment, t, int(frame_index))
        for event in events:
            self.emit(event)
        self._prev = assessment
        self._status = status
        return assessment, events

    # -- events -------------------------------------------------------------------
    def _make_event(self, etype: str, assessment: Assessment, t: float, frame_index: int,
                    track_ids: Sequence[int], extra: Optional[Dict[str, object]] = None) -> Event:
        return Event(
            type=etype,
            timestamp=t,
            frame_index=frame_index,
            status=assessment.status.value,
            reasons=list(assessment.reasons),
            track_ids=[int(i) for i in track_ids],
            extra=dict(extra) if extra else {},
        )

    def _transition_events(self, new: Assessment, t: float, frame_index: int) -> List[Event]:
        """Diff the new status against the previous one and build the event list.

        Transitions:
            CLEAR   -> WARNING : WARNING_START
            CLEAR   -> ALARM   : ALARM_START
            WARNING -> ALARM   : WARNING_END, ALARM_START
            WARNING -> CLEAR   : WARNING_END
            ALARM   -> WARNING : ALARM_END, WARNING_START
            ALARM   -> CLEAR   : ALARM_END
            ALARM   -> ALARM   : ALARM_START with extra={"repeat": True} every
                                 ``repeat_notify_seconds`` (0 disables)
        END events carry the ids that were in that severity on the previous frame.
        """
        prev_status = self._status
        prev_warn_ids = list(self._prev.warning_track_ids) if self._prev is not None else []
        prev_alarm_ids = list(self._prev.alarm_track_ids) if self._prev is not None else []
        events: List[Event] = []

        if new.status is prev_status:
            if new.status is Status.ALARM:
                repeat_s = float(self.alert_cfg.repeat_notify_seconds)
                if repeat_s > 0 and self._last_alarm_notify is not None \
                        and t - self._last_alarm_notify >= repeat_s:
                    events.append(self._make_event("ALARM_START", new, t, frame_index,
                                                   new.alarm_track_ids, {"repeat": True}))
                    self._last_alarm_notify = t
            return events

        # Close whatever was open.
        if prev_status is Status.WARNING:
            events.append(self._make_event("WARNING_END", new, t, frame_index, prev_warn_ids))
        elif prev_status is Status.ALARM:
            events.append(self._make_event("ALARM_END", new, t, frame_index, prev_alarm_ids))
            self._last_alarm_notify = None

        # Open the new one.
        if new.status is Status.WARNING:
            events.append(self._make_event("WARNING_START", new, t, frame_index, new.warning_track_ids))
        elif new.status is Status.ALARM:
            events.append(self._make_event("ALARM_START", new, t, frame_index, new.alarm_track_ids))
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
        sinks.append(WebhookSink(alert_cfg.webhook_url, timeout=alert_cfg.webhook_timeout_seconds))
    return sinks


__all__ = [
    "AlertEngine",
    "StdoutSink",
    "JsonlSink",
    "WebhookSink",
    "CallbackSink",
    "build_sinks",
    "event_to_json",
]
