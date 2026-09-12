"""The frame loop: wires zone, detector, tracker, gates, debris and alerts together.

``Monitor.process_frame`` is the single place where the modules meet, in this order
(see docs/ARCHITECTURE.md, "gradecrossing/monitor.py"):

1. optional resize to ``video.resize_width`` (the zone polygon is scaled by the same
   factor once, on the first frame, because it was calibrated at native resolution);
2. zone stabilisation, with ``scene_static`` true only when the tracker sees nothing in
   the zone *and* no debris alarm is active (a keyframe refresh must never be anchored on
   a frame with an obstruction in it);
3. the vehicle detector every ``detect_every_n_frames`` frames, otherwise the tracker's
   velocity-propagated boxes;
4. tracker update (per-object dwell / speed / stuck counters);
5. infrastructure (gate) detector, if enabled or injected;
6. debris detector with *every* box of the frame (any class, real or predicted) as an
   exclusion and the tracker's zone-empty time as the refresh gate (withheld while
   candidate debris is accumulating persistence);
7. alert evaluation and event delivery;
8. per-stage timings in milliseconds.

``Monitor.run`` owns the video I/O (files, cameras, RTSP with reconnect), the display
window, the annotated output video, alarm snapshots and the periodic status line;
``main`` is the CLI behind ``python -m gradecrossing.monitor``.

Testability: the detector and the infrastructure detector are constructed lazily on the
first frame, so a ``Monitor`` built with an injected fake detector never imports
ultralytics/torch (``tests/test_monitor.py`` drives the whole loop with scripted
detections and synthetic frames).
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .alerting import AlertEngine, JsonlSink, WebhookSink, build_sinks
from .background import DebrisDetector
from .config import DetectorConfig, MonitorConfig, VideoConfig, load_config
from .detection import (
    CATEGORY_OTHER,
    CATEGORY_PERSON,
    CATEGORY_TRAIN,
    CATEGORY_VEHICLE,
    InfrastructureDetector,
    VehicleDetector,
    _normalise_name,
)
from .tracking import StuckVehicleTracker
from .types import Detection, Event, FrameResult, GateState, ObjectState
from .visualize import draw_overlay
from .zone import ZoneStabilizer

LOGGER = logging.getLogger(__name__)

# URL schemes that mean "live stream": no seeking, wall-clock timestamps, reconnect on
# read failure. Anything else (and any existing path) is treated as a file.
LIVE_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://", "srt://")
# Live-source reconnect policy: attempts and the first back-off delay (doubles each time:
# 1, 2, 4, 8, 16 s = 31 s total by default before giving up).
RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_S = 1.0
# A crossing must keep being watched through a transient failure (one bad decode, one
# inference hiccup, a stream that renegotiated its resolution): the frame is skipped and
# logged instead of ending the run. Only this many *consecutive* failures - which means
# something is systematically broken, not transient - abort the loop.
MAX_CONSECUTIVE_FRAME_ERRORS = 5
# Frame rate assumed when the container/camera reports nothing usable.
DEFAULT_FPS = 30.0
WINDOW_NAME = "gradecrossing"


# --------------------------------------------------------------------------------------
# Class mapping without a model
# --------------------------------------------------------------------------------------
class ConfigClassifier:
    """``cls_name -> "vehicle" | "person" | "train" | "other"`` from ``DetectorConfig`` lists.

    Identical rule to ``VehicleDetector.classify`` (case-insensitive membership), kept
    here so the tracker can be built before the detector exists (the detector is loaded
    lazily) and so injected fake detectors without a ``classify`` method still get the
    configured class mapping.
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        self._vehicle = {_normalise_name(n) for n in cfg.vehicle_classes}
        self._person = {_normalise_name(n) for n in cfg.person_classes}
        self._train = {_normalise_name(n) for n in cfg.train_classes}

    def __call__(self, cls_name: str) -> str:
        key = _normalise_name(cls_name)
        if key in self._vehicle:
            return CATEGORY_VEHICLE
        if key in self._person:
            return CATEGORY_PERSON
        if key in self._train:
            return CATEGORY_TRAIN
        return CATEGORY_OTHER


# --------------------------------------------------------------------------------------
# Video sources
# --------------------------------------------------------------------------------------
def is_live_source(source: str) -> bool:
    """Guess whether ``source`` is a live stream (camera index or streaming URL) or a file.

    Digits mean a camera index, the schemes in ``LIVE_SCHEMES`` mean a network stream,
    ``/dev/video*`` is a V4L2 camera; everything else is a file. ``VideoConfig.live``
    overrides the guess when it is wrong (e.g. an HTTP URL that serves a static clip).
    """
    src = str(source).strip()
    if src.isdigit():
        return True
    low = src.lower()
    if low.startswith(LIVE_SCHEMES) or low.startswith("/dev/video"):
        return True
    return False


def _make_capture(source: str) -> "cv2.VideoCapture":
    """``cv2.VideoCapture`` for a path/URL, or for an integer camera index given as digits."""
    src = str(source).strip()
    if src.isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src)


def open_source(video: VideoConfig) -> Tuple["cv2.VideoCapture", float, bool]:
    """Open ``video.source`` and return ``(capture, fps, is_live)``.

    * ``fps`` is ``video.fps_override`` when set, else what the container reports; a
      missing/NaN/``<= 1`` value (common for RTSP cameras and some MOV files) falls back
      to ``DEFAULT_FPS`` with a warning. For files the frame timestamp is
      ``frame_index / fps``, so a wrong fps would scale every time threshold - the
      warning exists so nobody tunes ``alarm_seconds`` against a mis-timed clip.
    * For files ``video.start_seconds`` seeks before the first read.
    * Raises ``RuntimeError`` when the source cannot be opened (a typo in a path must be
      an error, not a silent zero-frame run).
    """
    source = str(video.source)
    is_live = bool(video.live) if video.live is not None else is_live_source(source)
    cap = _make_capture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video source {source!r}")

    reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video.fps_override is not None and float(video.fps_override) > 0:
        fps = float(video.fps_override)
    elif math.isnan(reported) or reported <= 1.0:
        LOGGER.warning("source %r reports fps=%s; assuming %.1f (set video.fps_override to fix timestamps)",
                       source, reported, DEFAULT_FPS)
        fps = DEFAULT_FPS
    else:
        fps = reported

    if not is_live and float(video.start_seconds) > 0:
        if not cap.set(cv2.CAP_PROP_POS_MSEC, float(video.start_seconds) * 1000.0):
            LOGGER.warning("seek to %.2fs not supported by this container; starting at 0", video.start_seconds)
    LOGGER.info("opened %r (%s, fps=%.2f)", source, "live" if is_live else "file", fps)
    return cap, fps, is_live


# --------------------------------------------------------------------------------------
# Monitor
# --------------------------------------------------------------------------------------
class Monitor:
    """Run the whole safety pipeline on frames; see the module docstring for the order.

    Parameters
    ----------
    cfg:
        Full ``MonitorConfig``.
    detector:
        Object with ``detect(frame) -> List[Detection]`` (and optionally ``classify``
        and ``reset``). ``None`` builds ``VehicleDetector(cfg.detector)`` on the first
        frame so that ultralytics is only imported when it is really used.
    infra_detector:
        Object with ``update(frame, t, i) -> GateState``. ``None`` builds
        ``InfrastructureDetector(cfg.infrastructure)`` on the first frame *only if*
        ``cfg.infrastructure.enabled``; an injected one is always used.
    sinks:
        Extra event sinks (objects with ``emit(event)``) added on top of the ones built
        from ``cfg.alerts`` (stdout / JSONL / webhook).
    """

    def __init__(self, cfg: MonitorConfig, detector: Any = None, infra_detector: Any = None,
                 sinks: Sequence[Any] = ()) -> None:
        self.cfg = cfg
        self._detector = detector
        self._infra = infra_detector
        self._config_classifier = ConfigClassifier(cfg.detector)
        # Built on the first frame: the polygon scale depends on the native frame size.
        self.stabilizer: Optional[ZoneStabilizer] = None
        self.tracker = StuckVehicleTracker(cfg.stuck, self._classify)
        self.debris: Optional[DebrisDetector] = DebrisDetector(cfg.debris) if cfg.debris.enabled else None
        self.engine = AlertEngine(cfg.stuck, cfg.debris, cfg.alerts)
        self._config_sinks = build_sinks(cfg.alerts)
        for sink in list(self._config_sinks) + list(sinks):
            self.engine.add_sink(sink)
        self.fps: Optional[float] = None  # set by run()
        self.last_frame: Optional[np.ndarray] = None  # processing-resolution frame of the last call
        self.last_result: Optional[FrameResult] = None
        self._frames_processed = 0  # detector cadence counter (reset() zeroes it)
        self._last_debris = None
        self._scale = 1.0

    # ------------------------------------------------------------------ helpers
    def _classify(self, cls_name: str) -> str:
        """Class mapping used by the tracker: the detector's own when it has one, else the config's."""
        det = self._detector
        if det is not None and hasattr(det, "classify"):
            return det.classify(cls_name)
        return self._config_classifier(cls_name)

    def _get_detector(self) -> Any:
        if self._detector is None:
            LOGGER.info("loading vehicle detector %s", self.cfg.detector.weights)
            self._detector = VehicleDetector(self.cfg.detector)
        return self._detector

    def _get_infra(self) -> Any:
        if self._infra is None and self.cfg.infrastructure.enabled:
            LOGGER.info("loading infrastructure detector %s", self.cfg.infrastructure.weights)
            self._infra = InfrastructureDetector(self.cfg.infrastructure)
        return self._infra

    def _prepare_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resize to ``video.resize_width`` (aspect preserved) when configured."""
        rw = self.cfg.video.resize_width
        if rw is None or int(rw) <= 0 or frame_bgr.shape[1] == int(rw):
            return frame_bgr
        rw = int(rw)
        scale = rw / float(frame_bgr.shape[1])
        new_h = max(1, int(round(frame_bgr.shape[0] * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(frame_bgr, (rw, new_h), interpolation=interp)

    def _build_stabilizer(self, native_width: int, processed_width: int) -> None:
        """Create the ZoneStabilizer with the polygon scaled from native to processing pixels.

        The polygon in the config is in the coordinates of the calibration frame (native
        resolution). If the monitor downsizes frames, the polygon must shrink by the same
        factor or the zone would sit in the wrong place - silently, which is the failure
        mode this package exists to remove.
        """
        self._scale = processed_width / float(native_width) if native_width > 0 else 1.0
        polygon = [[float(x) * self._scale, float(y) * self._scale] for x, y in self.cfg.zone.polygon]
        zone_cfg = dataclasses.replace(self.cfg.zone, polygon=polygon)
        self.stabilizer = ZoneStabilizer(zone_cfg)
        if self._scale != 1.0:
            LOGGER.info("zone polygon scaled by %.3f for resize_width=%s", self._scale, self.cfg.video.resize_width)

    def _zone_empty_seconds(self, timestamp: float) -> Optional[float]:
        """How long the tracker has seen the zone empty, or None when refresh must not run.

        The debris detector lets the background adapt only while this is large. Two
        cases return None on purpose: the tracker has not seen the zone empty (occupied
        or nothing known yet), and *candidate debris is accumulating persistence*
        (``persist_seconds > 0``). The tracker cannot see debris, so "zone empty" is
        exactly the state in which undetected debris is present; letting MOG2 adapt then
        would absorb the obstruction before ``persist_seconds`` elapses (about 21 frames
        at the default 0.005 rate) and the alarm would never fire. See background.py.
        """
        since = self.tracker.zone_empty_since
        if since is None:
            return None
        if self._last_debris is not None and float(self._last_debris.persist_seconds) > 0.0:
            return None
        return max(0.0, float(timestamp) - float(since))

    @staticmethod
    def _snapshot_objects(objects: List[ObjectState]) -> List[ObjectState]:
        """Copy the tracker's live ObjectStates so a FrameResult does not mutate next frame."""
        return [dataclasses.replace(o, last_box=np.array(o.last_box, copy=True), history=list(o.history))
                for o in objects]

    # ------------------------------------------------------------------ per frame
    def process_frame(self, frame_bgr: np.ndarray, timestamp: float, frame_index: int) -> FrameResult:
        """Run every stage on one frame and return the :class:`FrameResult`.

        ``timestamp`` is seconds (``frame_index / fps`` for files, monotonic for live),
        ``frame_index`` is the source frame number. Detector cadence uses an internal
        counter of processed frames so the detector always runs on the first frame after
        construction/reset, whatever the source index is.
        """
        timings: Dict[str, float] = {}
        t_all = time.perf_counter()
        timestamp = float(timestamp)
        frame_index = int(frame_index)

        # (1) resize
        t0 = time.perf_counter()
        frame = self._prepare_frame(frame_bgr)
        self.last_frame = frame
        timings["resize"] = (time.perf_counter() - t0) * 1000.0

        # (2) zone: refresh is allowed only when nothing tracked is in the zone and no
        # debris alarm is active (an obstruction must never become the new reference).
        t0 = time.perf_counter()
        if self.stabilizer is None:
            self._build_stabilizer(frame_bgr.shape[1], frame.shape[1])
        assert self.stabilizer is not None
        debris_active = self._last_debris is not None and bool(self._last_debris.triggered)
        scene_static = (not self.tracker.zone_occupied) and (not debris_active)
        zone = self.stabilizer.update(frame, timestamp, scene_static=scene_static)
        health_events: List[Event] = []
        if self.cfg.zone.health_events and self.stabilizer.health_changed:
            health = self.stabilizer.health()
            state_txt = "healthy" if zone.healthy else "unhealthy (holding last good polygon)"
            ev = Event(type="HEALTH", timestamp=timestamp, frame_index=frame_index,
                       status=self.engine.status.value,
                       reasons=[f"zone {state_txt}: {zone.method} {zone.inliers} inliers"],
                       track_ids=[], extra=dict(health))
            self.engine.emit(ev)
            health_events.append(ev)
        timings["zone"] = (time.perf_counter() - t0) * 1000.0

        # (3) detector every N processed frames, otherwise velocity-propagated boxes
        t0 = time.perf_counter()
        every = max(1, int(self.cfg.detector.detect_every_n_frames))
        if self._frames_processed % every == 0:
            detections: List[Detection] = list(self._get_detector().detect(frame))
        else:
            detections = list(self.tracker.predict_boxes(timestamp))
        timings["detect"] = (time.perf_counter() - t0) * 1000.0

        # (4) tracker
        t0 = time.perf_counter()
        objects = self.tracker.update(detections, zone, timestamp)
        timings["track"] = (time.perf_counter() - t0) * 1000.0

        # (5) infrastructure / gates
        t0 = time.perf_counter()
        infra = self._get_infra()
        gate = infra.update(frame, timestamp, frame_index) if infra is not None else GateState(available=False)
        timings["infra"] = (time.perf_counter() - t0) * 1000.0

        # (6) debris: every box of the frame (any class) is excluded from the foreground
        t0 = time.perf_counter()
        debris = None
        if self.debris is not None:
            boxes = [np.asarray(d.xyxy, dtype=np.float32).reshape(4) for d in detections]
            debris = self.debris.update(frame, zone.mask, boxes, self._zone_empty_seconds(timestamp), timestamp)
            self._last_debris = debris
            # A global lighting step re-warmed the background model: report it, because
            # debris detection is blind until the new warm-up finishes.
            illumination = getattr(self.debris, "last_illumination_event", None)
            if illumination:
                pct = 100.0 * float(illumination.get("area_fraction", 0.0))
                ev = Event(type="HEALTH", timestamp=timestamp, frame_index=frame_index,
                           status=self.engine.status.value,
                           reasons=[f"background invalid: {pct:.0f}% of the zone changed at once; "
                                    f"re-warming the background model"],
                           track_ids=[], extra=dict(illumination))
                self.engine.emit(ev)
                health_events.append(ev)
        timings["debris"] = (time.perf_counter() - t0) * 1000.0

        # (7) alerts
        t0 = time.perf_counter()
        assessment, events = self.engine.evaluate(objects, debris, gate, timestamp, frame_index)
        timings["alert"] = (time.perf_counter() - t0) * 1000.0

        # (8) bookkeeping
        timings["total"] = (time.perf_counter() - t_all) * 1000.0
        self._frames_processed += 1
        result = FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            assessment=assessment,
            zone=zone,
            detections=detections,
            objects=self._snapshot_objects(objects),
            debris=debris,
            gate=gate,
            timings_ms=timings,
            events=health_events + list(events),
        )
        self.last_result = result
        return result

    def reset(self) -> None:
        """Forget all state (new video, loop restart). The zone re-initialises on the next frame."""
        self.stabilizer = None
        self.tracker.reset()
        if self.debris is not None:
            self.debris.reset()
        self.engine.reset()
        for component in (self._detector, self._infra):
            if component is not None and hasattr(component, "reset"):
                try:
                    component.reset()
                except Exception:  # a detector reset failure must not abort the run
                    LOGGER.exception("%s.reset() failed", type(component).__name__)
        self._frames_processed = 0
        self._last_debris = None
        self.last_result = None

    def close(self) -> None:
        """Flush/close the sinks built from the config (JSONL file, webhook queue)."""
        for sink in self._config_sinks:
            try:
                if isinstance(sink, JsonlSink):
                    sink.close()
                elif isinstance(sink, WebhookSink):
                    sink.flush(timeout=self.cfg.alerts.webhook_timeout_seconds * 2)
            except Exception:
                LOGGER.exception("closing sink %s failed", type(sink).__name__)

    # ------------------------------------------------------------------ run loop
    def _reconnect(self, cap: "cv2.VideoCapture") -> Optional["cv2.VideoCapture"]:
        """Re-open a live source with exponential back-off; None after RECONNECT_ATTEMPTS."""
        source = str(self.cfg.video.source)
        for attempt in range(1, RECONNECT_ATTEMPTS + 1):
            delay = RECONNECT_BACKOFF_S * (2 ** (attempt - 1))
            LOGGER.warning("live source %r read failed; reconnect attempt %d/%d in %.0fs",
                           source, attempt, RECONNECT_ATTEMPTS, delay)
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(delay)
            cap = _make_capture(source)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    LOGGER.info("reconnected to %r", source)
                    return cap
        LOGGER.error("giving up on %r after %d reconnect attempts", source, RECONNECT_ATTEMPTS)
        try:
            cap.release()
        except Exception:
            pass
        return None

    def run(self) -> Dict[str, Any]:
        """Open ``cfg.video.source`` and process it to the end (or ``max_frames``).

        Returns a summary dict with at least ``frames``, ``alarms`` (non-repeat
        ALARM_START events) and ``fps`` (processed frames per wall-clock second), plus
        ``warnings``, ``health_events``, ``elapsed_s``, ``reconnects``, ``errors``,
        ``loops``, ``final_status``, ``interrupted`` and mean per-stage ``timings_ms``.
        Ctrl-C stops the loop cleanly and still returns the summary. ``max_frames``
        of ``None`` or ``<= 0`` means "run to the end / forever".

        Robustness: a frame that makes any stage raise is counted in ``errors``, logged
        with its traceback and skipped; the loop only gives up after
        ``MAX_CONSECUTIVE_FRAME_ERRORS`` failures in a row. A crossing that stops being
        watched because of one bad decoded frame, one transient inference error or a live
        stream that changed resolution is a worse outcome than a gap in the log.

        The config-built sinks are closed when ``run`` returns, so use one ``Monitor``
        per run (construct a new one to run again).
        """
        cfg = self.cfg
        cap, fps, is_live = open_source(cfg.video)
        self.fps = fps
        index0 = int(round(float(cfg.video.start_seconds) * fps)) if (not is_live and cfg.video.start_seconds > 0) else 0
        frame_index = index0
        max_frames = cfg.video.max_frames if (cfg.video.max_frames is not None and int(cfg.video.max_frames) > 0) else None
        processed = 0
        alarms = warnings = health_count = reconnects = loops = 0
        errors = consecutive_errors = 0
        frames_this_pass = 0
        sums: Dict[str, float] = {}
        writer: Optional["cv2.VideoWriter"] = None
        display = bool(cfg.output.display)
        snapshot_dir = str(cfg.output.snapshot_dir or "")
        print_every = int(cfg.output.print_every_n_frames or 0)
        fps_est: Optional[float] = None
        interrupted = False
        final_status = "CLEAR"
        wall_start = time.monotonic()
        t_live0 = wall_start

        try:
            while True:
                if max_frames is not None and processed >= int(max_frames):
                    break
                loop_t0 = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if is_live:
                        new_cap = self._reconnect(cap)
                        if new_cap is None:
                            break
                        cap = new_cap
                        reconnects += 1
                        continue
                    if cfg.video.loop and frames_this_pass > 0:
                        loops += 1
                        LOGGER.info("end of file; looping (pass %d)", loops + 1)
                        cap.release()
                        cap, fps, _ = open_source(cfg.video)
                        self.reset()  # timestamps restart with the file; all state is forgotten
                        frame_index = index0
                        frames_this_pass = 0
                        continue
                    break
                frames_this_pass += 1
                timestamp = (time.monotonic() - t_live0) if is_live else frame_index / fps

                try:
                    result = self.process_frame(frame, timestamp, frame_index)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the monitor
                    errors += 1
                    consecutive_errors += 1
                    LOGGER.exception("frame %d failed (%d consecutive error(s)); continuing",
                                     frame_index, consecutive_errors)
                    if is_live and isinstance(exc, ValueError):
                        # Almost always the ZoneStabilizer refusing a frame whose size no
                        # longer matches the reference keyframe: an RTSP source that
                        # renegotiated its resolution. Re-initialise on the new geometry
                        # rather than giving up on the crossing (pin the geometry with
                        # video.resize_width to avoid the re-initialisation entirely).
                        prior = self.engine.status.value
                        LOGGER.warning("re-initialising the pipeline on the new frame geometry "
                                       "(status was %s; tracks, background model and alarm latch "
                                       "are discarded)", prior)
                        self.reset()
                        ev = Event(type="HEALTH", timestamp=timestamp, frame_index=frame_index,
                                   status=self.engine.status.value,
                                   reasons=[f"frame rejected ({exc}); pipeline re-initialised, "
                                            f"previous status {prior} discarded"],
                                   track_ids=[],
                                   extra={"error": str(exc), "reset": True, "previous_status": prior})
                        self.engine.emit(ev)
                        health_count += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
                        LOGGER.error("aborting after %d consecutive frame errors", consecutive_errors)
                        break
                    frame_index += 1
                    continue
                consecutive_errors = 0
                processed += 1
                final_status = result.assessment.status.value
                for k, v in result.timings_ms.items():
                    sums[k] = sums.get(k, 0.0) + v

                alarm_started = False
                for ev in result.events:
                    if ev.type == "ALARM_START" and not ev.extra.get("repeat"):
                        alarms += 1
                        alarm_started = True
                    elif ev.type == "WARNING_START":
                        warnings += 1
                    elif ev.type == "HEALTH":
                        health_count += 1

                need_overlay = display or bool(cfg.output.save_video) or (alarm_started and bool(snapshot_dir))
                if need_overlay:
                    overlay = draw_overlay(self.last_frame, result, fps_est)
                    if alarm_started and snapshot_dir:
                        path = Path(snapshot_dir) / f"alarm_f{frame_index:07d}_t{timestamp:08.2f}s.jpg"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if cv2.imwrite(str(path), overlay):
                            LOGGER.info("snapshot %s", path)
                        else:
                            LOGGER.warning("could not write snapshot %s", path)
                    if cfg.output.save_video:
                        if writer is None:
                            h, w = overlay.shape[:2]
                            Path(cfg.output.save_video).parent.mkdir(parents=True, exist_ok=True)
                            writer = cv2.VideoWriter(str(cfg.output.save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     float(fps), (w, h))
                            if not writer.isOpened():
                                LOGGER.error("cannot open video writer %s; disabling save_video", cfg.output.save_video)
                                writer.release()
                                writer = None
                                cfg.output.save_video = ""
                        if writer is not None:
                            writer.write(overlay)
                    if display:
                        try:
                            cv2.imshow(WINDOW_NAME, overlay)
                            key = cv2.waitKey(1) & 0xFF
                            if key in (27, ord("q")):
                                LOGGER.info("display window closed by user")
                                break
                        except cv2.error:
                            # opencv-python-headless (CI, Jetson without X): keep processing.
                            LOGGER.warning("OpenCV has no GUI support here; disabling display")
                            display = False

                dt = time.perf_counter() - loop_t0
                inst = 1.0 / dt if dt > 0 else 0.0
                fps_est = inst if fps_est is None else 0.9 * fps_est + 0.1 * inst
                if print_every > 0 and processed % print_every == 0:
                    print(self._status_line(result, fps_est), flush=True)
                frame_index += 1
        except KeyboardInterrupt:
            interrupted = True
            LOGGER.info("interrupted by user")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            if writer is not None:
                writer.release()
            if display:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            self.close()

        elapsed = time.monotonic() - wall_start
        summary: Dict[str, Any] = {
            "frames": processed,
            "alarms": alarms,
            "fps": (processed / elapsed) if elapsed > 0 else 0.0,
            "warnings": warnings,
            "health_events": health_count,
            "elapsed_s": elapsed,
            "reconnects": reconnects,
            "errors": errors,
            "loops": loops,
            "final_status": final_status,
            "interrupted": interrupted,
            "timings_ms": {k: v / processed for k, v in sums.items()} if processed else {},
        }
        LOGGER.info("run finished: %d frames, %d alarms, %.1f fps", processed, alarms, summary["fps"])
        return summary

    @staticmethod
    def _status_line(result: FrameResult, fps_est: Optional[float]) -> str:
        """One ASCII status line for the periodic console print."""
        in_zone = sum(1 for o in result.objects if o.in_zone)
        debris_txt = f"{100.0 * result.debris.area_fraction:.1f}%" if result.debris is not None else "off"
        zone = result.zone
        total = result.timings_ms.get("total", 0.0)
        fps_txt = f" {fps_est:.1f}fps" if fps_est else ""
        return (f"[f={result.frame_index} t={result.timestamp:.1f}s] {result.assessment.status.value:<7} "
                f"objs={len(result.objects)} inzone={in_zone} debris={debris_txt} "
                f"zone={zone.method}/{zone.inliers}{'' if zone.healthy else ' HELD'} "
                f"{total:.1f}ms{fps_txt}").encode("ascii", "replace").decode("ascii")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Argument parser for ``python -m gradecrossing.monitor``; every flag overrides the YAML."""
    p = argparse.ArgumentParser(
        prog="python -m gradecrossing.monitor",
        description="AI rail grade-crossing monitor: stuck-vehicle, pedestrian and debris alerts "
                    "on a video file, camera or RTSP stream.",
        epilog="Example: python -m gradecrossing.monitor --config configs/stresstest.yaml --events events.jsonl",
    )
    p.add_argument("--config", metavar="YAML", help="config file (default: built-in defaults; see configs/default.yaml)")
    p.add_argument("--source", help="video file, camera index or stream URL (overrides video.source)")
    p.add_argument("--weights", help="vehicle detector weights .pt/.engine (overrides detector.weights)")
    p.add_argument("--display", action="store_true", help="show an annotated window (needs an OpenCV GUI build)")
    p.add_argument("--save", metavar="OUT.mp4", help="write the annotated video (overrides output.save_video)")
    p.add_argument("--events", metavar="events.jsonl", help="append events as JSON lines (overrides alerts.events_jsonl)")
    p.add_argument("--max-frames", type=int, metavar="N", help="stop after N frames (overrides video.max_frames)")
    p.add_argument("--infra-weights", metavar="W", help="enable the gate detector with these weights")
    p.add_argument("--infra", action="store_true", help="enable the gate detector with the configured weights")
    p.add_argument("--no-stabilize", action="store_true", help="disable zone stabilisation (zone.stabilize=false)")
    p.add_argument("--print-every", type=int, metavar="N", help="print a status line every N frames")
    p.add_argument("--fps", type=float, help="force the source frame rate (overrides video.fps_override)")
    p.add_argument("--resize-width", type=int, metavar="W", help="process at this width (zone polygon is scaled)")
    p.add_argument("--start-seconds", type=float, help="seek before processing (files only)")
    p.add_argument("--loop", action="store_true", help="loop a file (soak test)")
    p.add_argument("--live", action="store_true", help="treat the source as live (wall-clock timestamps, reconnect)")
    p.add_argument("--detect-every", type=int, metavar="N", help="run the detector every N frames")
    p.add_argument("--device", help="inference device for the detectors ('' auto, '0' CUDA, 'cpu')")
    p.add_argument("--snapshot-dir", metavar="DIR", help="save an annotated JPEG on every ALARM_START")
    p.add_argument("--webhook", metavar="URL", help="POST events to this URL (overrides alerts.webhook_url)")
    p.add_argument("--no-stdout", action="store_true", help="do not print events to stdout")
    p.add_argument("--log-level", help="logging level (overrides output.log_level)")
    return p


def apply_cli_overrides(cfg: MonitorConfig, args: argparse.Namespace) -> MonitorConfig:
    """Copy every CLI flag that was given onto ``cfg`` (in place) and return it."""
    if args.source is not None:
        cfg.video.source = args.source
    if args.weights is not None:
        cfg.detector.weights = args.weights
    if args.display:
        cfg.output.display = True
    if args.save is not None:
        cfg.output.save_video = args.save
    if args.events is not None:
        cfg.alerts.events_jsonl = args.events
    if args.max_frames is not None:
        cfg.video.max_frames = int(args.max_frames)
    if args.infra_weights is not None:
        cfg.infrastructure.weights = args.infra_weights
        cfg.infrastructure.enabled = True
    if args.infra:
        cfg.infrastructure.enabled = True
    if args.no_stabilize:
        cfg.zone.stabilize = False
    if args.print_every is not None:
        cfg.output.print_every_n_frames = int(args.print_every)
    if args.fps is not None:
        cfg.video.fps_override = float(args.fps)
    if args.resize_width is not None:
        cfg.video.resize_width = int(args.resize_width)
    if args.start_seconds is not None:
        cfg.video.start_seconds = float(args.start_seconds)
    if args.loop:
        cfg.video.loop = True
    if args.live:
        cfg.video.live = True
    if args.detect_every is not None:
        cfg.detector.detect_every_n_frames = max(1, int(args.detect_every))
    if args.device is not None:
        cfg.detector.device = args.device
        cfg.infrastructure.device = args.device
    if args.snapshot_dir is not None:
        cfg.output.snapshot_dir = args.snapshot_dir
    if args.webhook is not None:
        cfg.alerts.webhook_url = args.webhook
    if args.no_stdout:
        cfg.alerts.stdout = False
    if args.log_level is not None:
        cfg.output.log_level = args.log_level
    return cfg


def format_summary(summary: Dict[str, Any]) -> str:
    """ASCII one-liner (plus per-stage ms) for the end of a run."""
    stages = summary.get("timings_ms") or {}
    stage_txt = " ".join(f"{k}={v:.1f}ms" for k, v in stages.items())
    errors = int(summary.get("errors", 0) or 0)
    return (f"summary: frames={summary.get('frames', 0)} alarms={summary.get('alarms', 0)} "
            f"warnings={summary.get('warnings', 0)} fps={summary.get('fps', 0.0):.1f} "
            f"elapsed={summary.get('elapsed_s', 0.0):.1f}s final={summary.get('final_status', '-')}"
            + (f" errors={errors}" if errors else "")  # only mentioned when frames were lost
            + (f" | {stage_txt}" if stage_txt else ""))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point (``gradecrossing-monitor`` / ``python -m gradecrossing.monitor``).

    Exit status: 0 success, 1 the run could not start or failed (source cannot be opened,
    weights missing), 2 the config could not be loaded. "Could not be loaded" covers every
    way a config file goes wrong - a missing file, a YAML syntax error (a tab in the
    indentation is the classic one), a top level that is not a mapping, an unknown key or a
    value of the wrong type - because from the operator's point of view they are one
    mistake with one fix, and a traceback is not a useful answer to any of them.
    """
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 - see the exit-status note above
        print(f"error: cannot load config {args.config!r}: {exc}", file=sys.stderr)
        return 2
    apply_cli_overrides(cfg, args)
    level = getattr(logging, str(cfg.output.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    monitor = Monitor(cfg)
    try:
        summary = monitor.run()
    except (RuntimeError, FileNotFoundError, ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_summary(summary), flush=True)
    return 0


__all__ = [
    "ConfigClassifier",
    "Monitor",
    "apply_cli_overrides",
    "build_parser",
    "format_summary",
    "is_live_source",
    "main",
    "open_source",
]


if __name__ == "__main__":
    raise SystemExit(main())
