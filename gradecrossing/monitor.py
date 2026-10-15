"""The frame loop: wires zone, detector, tracker, gates, debris and alerts together.

``Monitor.process_frame`` is the single place where the modules meet, in this order
(see docs/ARCHITECTURE.md, "gradecrossing/monitor.py"):

1. resize to the processing size: the first frame's, at ``video.resize_width`` when that
   is set (the zone polygon is scaled once, on the first frame, from its calibration size -
   ``zone.reference_size``, else the size of ``zone.reference_image``, else the native
   first frame - to the processing size). A later frame of another resolution with the
   same aspect ratio (a live stream that renegotiated it) is resized to that same size, so
   nothing downstream sees a new geometry and nothing is re-initialised (one HEALTH event,
   ``extra['component'] == 'source'``, says so); another aspect ratio raises
   ``FrameGeometryError``, on which ``run`` re-initialises a live source;
2. zone stabilisation, with ``scene_static`` true only when the tracker sees nothing in
   the zone *and* no debris alarm is active (a keyframe refresh must never be anchored on
   a frame with an obstruction in it);
3. the vehicle detector every ``detect_every_n_frames`` frames, otherwise the tracker's
   velocity-propagated boxes;
4. tracker update (per-object dwell / speed / stuck / still counters, measured in the
   reference frame through ``ZoneState.to_reference``; ``detector_ran`` says whether the
   boxes are a real detection, since only those can age a track out);
5. infrastructure (gate) detector, if enabled or injected; one that raises leaves the
   frame without gate information (and the monitor DEGRADED) instead of failing it;
6. debris detector, registered to its background through ``ZoneState.to_reference``
   (timers held while the zone holds), with the boxes of vehicles, persons and trains
   as exclusions (an unmapped class such as 'cow' stays visible to the backstop) and its
   timers held while a train box overlaps the zone;
7. the DEGRADED state (``FrameResult.degraded``): what keeps the monitor from fully
   seeing this frame; a change that lasts ``alerts.health_debounce_seconds`` becomes a
   HEALTH event with ``extra['component'] == 'monitor'`` (zone health is left to the
   zone's own HEALTH events, which are debounced by ``zone.health_report_seconds``, and
   debris warm-ups to the debris component's; the debris trusted/untrusted events of step 6
   are debounced the same way);
8. alert evaluation and event delivery;
9. per-stage timings in milliseconds.

``Monitor.reset`` (a new video, an integrator starting over) closes an open WARNING/ALARM
episode with its END event (``extra['reason']``) before it forgets the state, so a
receiver never keeps showing an alarm that nothing will end. Two callers skip that END:
a looped file (RUN_END already closed the pass) and a live stream re-initialised on a new
aspect ratio (``keep_alerts``: the alert state carries on, because the hazard may still be
there and the re-initialised pipeline cannot know yet).

``Monitor.run`` owns the video I/O (files, cameras, RTSP with reconnect and open/read
timeouts), the display window, the annotated output video, alarm snapshots and the
periodic status line. It frames every run with RUN_START / RUN_END events (the event-log
contract in ``gradecrossing.evaluate``), reports a lost stream, each reconnect attempt and
every abort as HEALTH events, and says in its summary why it stopped; ``main`` is the CLI
behind ``python -m gradecrossing.monitor`` and turns that into the exit status. ``main``
stops the run on SIGTERM / SIGHUP (systemctl stop, docker stop) exactly as on Ctrl-C, so
the RUN_END and the sink flush still happen.

Testability: the detector and the infrastructure detector are built lazily, on the first
frame, right after the zone set-up (``process_frame`` step 3, or ``run``'s
``_require_models`` when that first frame failed), so a ``Monitor`` built with an injected
fake detector never imports ultralytics/torch (``tests/test_monitor.py`` drives the whole
loop with scripted detections and synthetic frames; ``tests/test_monitor_run.py`` drives
``run`` with fake captures).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import re
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import __version__
from .alerting import AlertEngine, ascii_safe, build_sinks
from .background import DebrisDetector
from .config import MonitorConfig, VideoConfig, load_config, validate_config, zone_file_argument, zone_polygon
from .detection import (  # ConfigClassifier is re-exported: gradecrossing.monitor.ConfigClassifier keeps working
    CATEGORY_PERSON,
    CATEGORY_TRAIN,
    CATEGORY_VEHICLE,
    ConfigClassifier,
    InfrastructureDetector,
    VehicleDetector,
)
from .geometry import box_mask_overlap_fraction
from .tracking import StuckVehicleTracker
from .types import Detection, Event, FrameResult, GateState, ObjectState
from .visualize import draw_overlay
from .zone import ASPECT_TOLERANCE, ZoneGeometryError, ZoneStabilizer, build_stabilizer as build_zone_stabilizer

LOGGER = logging.getLogger(__name__)

# URL schemes that mean "live stream": no seeking, wall-clock timestamps, reconnect on
# read failure. Anything else (and any existing path) is treated as a file.
LIVE_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "tcp://", "srt://")
# Live-source reconnect policy: attempts and the first back-off delay (doubles each time:
# 1, 2, 4, 8, 16 s = 31 s of waiting). Each attempt also blocks in cv2.VideoCapture for up
# to video.stream_timeout_seconds (open, then the check read; 10 s by default, the
# backend's 30 s without it), so a dead camera is given up after about 31 s + 5 x 10 s =
# 81 s by default (about 3.5 min with the backend default).
RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_S = 1.0
# A crossing must keep being watched through a transient failure (one bad decode, one
# inference hiccup): the frame is skipped and logged instead of ending the run. Only this
# many *consecutive* failures - which means something is systematically broken, not
# transient - abort the loop.
MAX_CONSECUTIVE_FRAME_ERRORS = 5
# Frame rate assumed when the container/camera reports nothing usable (0, NaN, negative).
DEFAULT_FPS = 30.0
# A reported frame rate outside this range is used but logged: 1 fps CCTV archives are
# real, a camera claiming 1000 fps usually is not.
PLAUSIBLE_FPS = (2.0, 240.0)
# File timestamps come from the container clock (CAP_PROP_POS_MSEC) and are logged once
# when they drift this far from frame_index / fps (a variable-frame-rate file).
CLOCK_DIVERGENCE_WARN_S = 0.1
# Why a run stopped (summary["stop_reason"], RUN_END extra.stop_reason). The aborting ones
# mean the crossing stopped being watched before the input ended: main() exits 1 for
# them, 2 for zone_refused (the calibration does not fit the stream).
STOP_END_OF_INPUT = "end_of_input"  # the file ended
STOP_MAX_FRAMES = "max_frames"  # video.max_frames frames were processed
STOP_INTERRUPTED = "interrupted"  # Ctrl-C
STOP_USER_QUIT = "user_quit"  # q / Esc in the display window
STOP_LOOP = "loop"  # RUN_END of one pass over a looped file; a new RUN_START follows
STOP_FRAME_ERRORS = "frame_errors"  # MAX_CONSECUTIVE_FRAME_ERRORS frames failed in a row (aborted)
STOP_RECONNECT_FAILED = "reconnect_failed"  # a live source could not be re-opened (aborted)
STOP_ZONE_REFUSED = "zone_refused"  # ZoneGeometryError: the zone does not fit the stream (aborted)
STOP_DETECTOR = "detector_unavailable"  # the detector could not be loaded (aborted)
STOP_ERROR = "error"  # an unexpected exception ended the loop (aborted)
# Exit status of main() (see its docstring and the --help epilog).
EXIT_OK = 0  # the input ended normally or the user stopped the run
EXIT_NOT_WATCHING = 1  # the crossing stopped being watched, or watching never started
EXIT_CONFIG = 2  # configuration / usage error
WINDOW_NAME = "gradecrossing"
# A track keeps its debris exclusion this long after its last vehicle/person/train label
# (YOLO flips a waiting car to 'boat' on some frames; that must not expose it as debris).
DEBRIS_MAPPED_MEMORY_S = 5.0
# DEGRADED reason of a debris background warm-up. It is part of FrameResult.degraded_reasons
# but not of the monitor's HEALTH report: the debris component reports the start and the
# end of every warm-up itself (step 6), so the start-up would otherwise be reported twice.
DEGRADED_DEBRIS_WARMUP = "debris warming up"
# Signals on which main() stops the run like Ctrl-C (RUN_END, sinks flushed, exit 0):
# systemctl stop and docker stop send SIGTERM, a closed terminal SIGHUP. A name this
# platform does not have (SIGHUP on Windows) is skipped.
STOP_SIGNALS = ("SIGTERM", "SIGHUP")


class FrameGeometryError(ValueError):
    """A frame whose aspect ratio differs from the processing size the first frame fixed.

    Raised by ``Monitor.process_frame`` (step 1) before any stage sees the frame. Another
    resolution with the same aspect ratio is not an error: it is resized to the processing
    size. ``Monitor.run`` re-initialises a live source on the new geometry (the zone set-up
    then decides whether the calibration fits it; usually it does not and the run stops
    with ``zone_refused``); for a file it is a lost frame like any other error.
    """


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


def _stream_timeout_params(timeout_s: float) -> Optional[List[int]]:
    """``cv2.VideoCapture`` open parameters setting the open and read timeouts to
    ``timeout_s``, or None: no timeout asked for, or a build that cannot apply them (OpenCV
    older than 4.5.2, no FFmpeg stream backend)."""
    open_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if open_prop is None or read_prop is None or not timeout_s > 0.0:
        return None
    try:
        if cv2.CAP_FFMPEG not in cv2.videoio_registry.getStreamBackends():
            return None
    except Exception:  # noqa: BLE001 - a build without the registry API: no timeouts
        return None
    ms = max(1, int(round(float(timeout_s) * 1000.0)))
    return [int(open_prop), ms, int(read_prop), ms]


def _make_capture(source: str, timeout_s: float = 0.0) -> "cv2.VideoCapture":
    """``cv2.VideoCapture`` for a path/URL, or for an integer camera index given as digits.

    A stream URL (``LIVE_SCHEMES``) is opened by the FFmpeg backend with open and read
    timeouts of ``timeout_s`` (``video.stream_timeout_seconds``; 0 = none) where the build
    supports them: without them an unreachable camera blocks every (re)connect attempt for
    the backend default (30 s measured with OpenCV 4.14, plus the other backends' tries) and
    a frozen stream blocks ``read()`` as long. Other sources, and builds without the
    timeouts, get the plain backend choice.
    """
    src = str(source).strip()
    if src.isdigit():
        return cv2.VideoCapture(int(src))
    if src.lower().startswith(LIVE_SCHEMES):
        params = _stream_timeout_params(timeout_s)
        if params is not None:
            try:
                return cv2.VideoCapture(src, cv2.CAP_FFMPEG, params)
            except (TypeError, cv2.error) as exc:  # a binding without the (source, api, params) overload
                LOGGER.warning("cannot open the stream with timeouts (%s); opening it without", exc)
    return cv2.VideoCapture(src)


def _redact_source(source: Any) -> str:
    """``source`` with the ``user:password@`` of a URL replaced by ``***@``.

    Events leave the machine (webhook, shared JSONL logs) and RTSP URLs usually carry the
    camera's credentials.
    """
    return re.sub(r"(?<=://)[^/@\s]+@", "***@", str(source))


def _release(cap: Any) -> None:
    """Release a capture, ignoring every error (it may already be dead)."""
    try:
        cap.release()
    except Exception:  # noqa: BLE001 - nothing useful left to do with a dead capture
        pass


def open_source(video: VideoConfig) -> Tuple["cv2.VideoCapture", float, bool]:
    """Open ``video.source`` and return ``(capture, fps, is_live)``.

    * ``fps`` is ``video.fps_override`` when set, else what the container reports. Any
      finite value > 0 is used (a 1 fps CCTV archive is real), with a warning outside
      ``PLAUSIBLE_FPS``; a missing/NaN/``<= 0`` value (common for RTSP cameras) falls back
      to ``DEFAULT_FPS`` with a warning. ``Monitor.run`` timestamps file frames with the
      container clock (CAP_PROP_POS_MSEC) and uses ``frame_index / fps`` only where that
      clock is unusable or ``fps_override`` is set; live frames get wall-clock time.
    * For files ``video.start_seconds`` seeks before the first read.
    * A stream URL is opened with ``video.stream_timeout_seconds`` as its open and read
      timeout (see ``_make_capture``).
    * Raises ``RuntimeError`` when the source cannot be opened (a typo in a path must be
      an error, not a silent zero-frame run).
    """
    source = str(video.source)
    shown = _redact_source(source)
    is_live = bool(video.live) if video.live is not None else is_live_source(source)
    cap = _make_capture(source, float(video.stream_timeout_seconds or 0.0))
    if not cap.isOpened():
        _release(cap)
        raise RuntimeError(f"cannot open video source {shown!r}")

    reported = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video.fps_override is not None and float(video.fps_override) > 0:
        fps = float(video.fps_override)
    elif not math.isfinite(reported) or reported <= 0.0:
        LOGGER.warning("source %r reports fps=%s; assuming %.1f (set video.fps_override to fix timestamps)",
                       shown, reported, DEFAULT_FPS)
        fps = DEFAULT_FPS
    else:
        fps = reported
        if not PLAUSIBLE_FPS[0] <= fps <= PLAUSIBLE_FPS[1]:
            LOGGER.warning("source %r reports an unusual fps=%.3f; using it (set video.fps_override if it is "
                           "wrong)", shown, fps)

    if not is_live and float(video.start_seconds) > 0:
        if not cap.set(cv2.CAP_PROP_POS_MSEC, float(video.start_seconds) * 1000.0):
            LOGGER.warning("seek to %.2fs not supported by this container; starting at 0", video.start_seconds)
    LOGGER.info("opened %r (%s, fps=%.2f)", shown, "live" if is_live else "file", fps)
    return cap, fps, is_live


def _container_seconds(cap: Any) -> Optional[float]:
    """CAP_PROP_POS_MSEC of the frame just read, in seconds; None when the capture has none."""
    try:
        ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    except Exception:  # noqa: BLE001 - a capture without the property simply has no clock
        return None
    return ms / 1000.0 if math.isfinite(ms) else None


def config_sha256(cfg: MonitorConfig) -> str:
    """SHA-256 of ``cfg`` serialised deterministically (sorted keys, compact JSON).

    RUN_START carries it, so two runs in an event log can be told apart by configuration.
    """
    text = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare_output_paths(snapshot_dir: str = "", save_video: str = "", events_jsonl: str = "") -> None:
    """Create the output locations before the first frame; ``RuntimeError`` if one cannot be.

    ``output.snapshot_dir``, the directory of ``output.save_video`` and (opened for append,
    as the JSONL sink will) ``alerts.events_jsonl``. A path that can never work must stop
    the start-up with one clear message, not the monitor at its first ALARM_START (MISS-03).
    """
    targets = []
    if snapshot_dir:
        targets.append(("output.snapshot_dir", Path(snapshot_dir), False))
    if save_video:
        targets.append(("output.save_video", Path(save_video).parent, False))
    if events_jsonl:
        targets.append(("alerts.events_jsonl", Path(events_jsonl), True))
    for key, path, is_file in targets:
        try:
            (path.parent if is_file else path).mkdir(parents=True, exist_ok=True)
            if is_file:
                with open(path, "a", encoding="utf-8"):
                    pass
        except OSError as exc:
            raise RuntimeError(f"cannot create {key} {str(path)!r}: {exc}") from exc


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
        self._zone_points = zone_polygon(validate_config(cfg).zone)  # bad value / zone file: ValueError, OSError here
        self._detector = detector
        self._infra = infra_detector
        self._config_classifier = ConfigClassifier(cfg.detector)
        # Built on the first frame: the polygon scale depends on the native frame size.
        self.stabilizer: Optional[ZoneStabilizer] = None
        VehicleDetector.check_config(cfg.detector)  # the detector itself is built on the first frame
        self.tracker = StuckVehicleTracker(cfg.stuck, self._classify)
        self.debris: Optional[DebrisDetector] = DebrisDetector(cfg.debris) if cfg.debris.enabled else None
        self.engine = AlertEngine(cfg.stuck, cfg.alerts)
        self._config_sinks = build_sinks(cfg.alerts)
        for sink in list(self._config_sinks) + list(sinks):
            self.engine.add_sink(sink)
        self.fps: Optional[float] = None  # set by run()
        self.last_frame: Optional[np.ndarray] = None  # processing-resolution frame of the last call
        # Result of the last processed frame (integrator API; reset() also takes the END
        # timestamp of an episode it closes from it).
        self.last_result: Optional[FrameResult] = None
        self._frames_processed = 0  # detector cadence counter (reset() zeroes it)
        self._last_debris = None
        self._scale = 1.0
        # Debris helpers (step 6): track ids recently seen with a mapped class, the ignore mask.
        self._debris_mapped_ids: Dict[int, float] = {}
        self._debris_ignore: Optional[Tuple[Any, np.ndarray]] = None
        # DEGRADED bookkeeping (step 7): timestamp of the last detector run and the interval
        # before it, whether the gate detector is failing, the state last reported in a
        # HEALTH event (kept across reset(): it is what the receivers were told) and since
        # when the current, different state has lasted.
        if float(cfg.alerts.health_debounce_seconds) < 0.0:
            raise ValueError("alerts.health_debounce_seconds must be >= 0")
        self._last_detect_t: Optional[float] = None
        self._detect_interval: Optional[float] = None
        self._gate_failing = False
        self._degraded_reported = False
        self._degraded_since: Optional[float] = None
        # Debris trusted/untrusted HEALTH events (step 6) are debounced the same way: the
        # value last reported (None until a warm-up completes) and since when it differs.
        self._debris_trust_reported: Optional[bool] = None
        self._debris_trust_since: Optional[float] = None
        # Step 1: the processing (w, h) the first frame fixed, and the native (w, h) of the
        # last frame (a change is reported once). reset() forgets both.
        self._processing_size: Optional[Tuple[int, int]] = None
        self._native_size: Optional[Tuple[int, int]] = None

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
        """The frame at the processing size (step 1).

        The first frame fixes the processing size: its own, or ``video.resize_width`` wide
        with the aspect ratio kept. Every later frame is brought to that same (w, h). A live
        stream that renegotiates its resolution (a camera or NVR profile change, often after
        a reconnect) keeps its aspect ratio: its frames are resized (``ASPECT_TOLERANCE``,
        as for the zone calibration), so the zone, the tracks and the debris background stay
        valid and nothing is re-initialised - a re-initialisation would re-learn an
        obstruction in view as background. A frame with another aspect ratio raises
        ``FrameGeometryError``: it shows a different picture.
        """
        h, w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"empty frame {w}x{h}")
        size = self._processing_size
        if size is None:
            rw = self.cfg.video.resize_width
            if rw is None or int(rw) <= 0 or w == int(rw):
                size = (w, h)
            else:
                size = (int(rw), max(1, int(round(h * int(rw) / float(w)))))
            self._processing_size = size
        elif (w, h) != size and abs((w / float(h)) / (size[0] / float(size[1])) - 1.0) > ASPECT_TOLERANCE:
            raise FrameGeometryError(
                f"frame size {w}x{h} (aspect {w / float(h):.3f}) does not match the processing size "
                f"{size[0]}x{size[1]} (aspect {size[0] / float(size[1]):.3f}) the first frame fixed")
        if (w, h) == size:
            return frame_bgr
        interp = cv2.INTER_AREA if w > size[0] else cv2.INTER_LINEAR
        return cv2.resize(frame_bgr, size, interpolation=interp)

    def _source_size_event(self, frame_bgr: np.ndarray, timestamp: float, frame_index: int) -> Optional[Event]:
        """HEALTH event (``component`` "source") when the native frame size changed since
        the last frame: the stream renegotiated its resolution and is now resized."""
        native = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        previous, self._native_size = self._native_size, native
        if previous is None or previous == native or self._processing_size is None:
            return None
        pw, ph = self._processing_size
        text = (f"stream resolution changed from {previous[0]}x{previous[1]} to {native[0]}x{native[1]}; "
                f"frames are processed at {pw}x{ph} as before (zone, tracks and debris background kept)")
        LOGGER.warning("%s", text)
        ev = Event(type="HEALTH", timestamp=float(timestamp), frame_index=int(frame_index),
                   status=self.engine.status.value, reasons=[text], track_ids=[],
                   extra={"component": "source", "state": "resolution_changed",
                          "previous_size": list(previous), "native_size": list(native),
                          "processing_size": [pw, ph]})
        self.engine.emit(ev)
        return ev

    # Native (w, h) of the first frame this monitor ever processed. Kept across reset(), so
    # a live re-initialisation after the stream renegotiated its resolution still scales
    # the polygon from the pixels it was calibrated in (zone.reference_size absent).
    _zone_native_size: Optional[Tuple[int, int]] = None
    # Last ZoneGeometryError message, so a refusal repeated on every frame is reported once.
    _zone_refusal: Optional[str] = None

    def _build_stabilizer(self, native_shape: Tuple[int, ...], processed_shape: Tuple[int, ...],
                          timestamp: float = 0.0) -> List[str]:
        """Create the ZoneStabilizer with the polygon scaled from calibration to processing pixels.

        The polygon in the config is in the pixels of the calibration frame:
        ``zone.reference_size`` when set, else the size of ``zone.reference_image``, else
        the native size of the first frame. It is scaled by processed/calibration size on
        both axes; if the monitor downsizes frames or the stream changed resolution, a
        polygon that was not scaled would sit in the wrong place - silently, which is the
        failure mode this package exists to remove. A different aspect ratio raises
        ``ZoneGeometryError`` (see ``gradecrossing.zone.build_stabilizer``). Returns the
        start-up problems that must be reported as HEALTH events.
        """
        native = (int(native_shape[1]), int(native_shape[0]))
        if self._zone_native_size is None:
            self._zone_native_size = native
        processed = (int(processed_shape[1]), int(processed_shape[0]))
        self.stabilizer, (sx, sy), problems = build_zone_stabilizer(  # the points resolved once in __init__
            dataclasses.replace(self.cfg.zone, polygon=self._zone_points), processed, self._zone_native_size, timestamp)
        self._scale = sx
        if abs(sx - 1.0) > 1e-9 or abs(sy - 1.0) > 1e-9:
            LOGGER.info("zone polygon scaled by %.3f x %.3f (calibration -> %dx%d processing pixels)",
                        sx, sy, processed[0], processed[1])
        for problem in problems:
            LOGGER.error("%s", problem)
        return problems

    def _zone_empty_seconds(self, timestamp: float) -> Optional[float]:
        """How long the tracker has seen the zone empty, or None (occupied, nothing known yet,
        or candidate debris accumulating persistence).

        The debris detector reads it in one place only: its illumination guard, when the
        zone covers the whole frame and nothing outside the zone can corroborate a lighting
        change, re-warms the background only while this is not None. Candidate debris
        returns None on purpose: the tracker cannot see debris, so "zone empty" is exactly
        the state in which undetected debris is present, and a re-warm then would forget
        the obstruction. (The background update itself is selective and does not use this;
        see background.py.)
        """
        since = self.tracker.zone_empty_since
        if since is None:
            return None
        if self._last_debris is not None and float(self._last_debris.persist_seconds) > 0.0:
            return None
        return max(0.0, float(timestamp) - float(since))

    def _debris_exclusions(self, detections: List[Detection], zone_mask: np.ndarray,
                           timestamp: float) -> Tuple[List[np.ndarray], bool]:
        """Debris exclusion boxes, and whether a train detection overlaps the zone.

        Only boxes that explain what they cover are excluded: vehicles, persons and trains,
        plus a track that had one of those labels within ``DEBRIS_MAPPED_MEMORY_S`` (a car
        that YOLO calls 'boat' on some frames stays excluded). A box of an unmapped class
        ('cow', 'suitcase', 'bench') is not, so the debris backstop still sees an obstruction
        that the alert engine ignores.
        """
        boxes: List[np.ndarray] = []
        train = False
        mapped_ids = self._debris_mapped_ids
        for d in detections:
            category = self._classify(d.cls_name)
            box = np.asarray(d.xyxy, dtype=np.float32).reshape(4)
            tid = d.track_id
            mapped = category in (CATEGORY_VEHICLE, CATEGORY_PERSON, CATEGORY_TRAIN)
            if mapped and tid is not None:
                mapped_ids[int(tid)] = float(timestamp)
            elif not mapped and tid is not None:
                mapped = float(timestamp) - mapped_ids.get(int(tid), -math.inf) <= DEBRIS_MAPPED_MEMORY_S
            if mapped:
                boxes.append(box)
            if category == CATEGORY_TRAIN and not train and box_mask_overlap_fraction(box, zone_mask) > 0.0:
                train = True
        for tid in [k for k, v in mapped_ids.items() if float(timestamp) - v > DEBRIS_MAPPED_MEMORY_S]:
            del mapped_ids[tid]
        return boxes, train

    @staticmethod
    def _debris_registration(zone: Any) -> Tuple[Optional[np.ndarray], bool]:
        """(current -> background reference matrix or None, registration_ok) for the debris model.

        The stabiliser's ``zone.to_reference`` (current pixels -> calibration-frame pixels,
        composed across keyframe re-anchors) while the zone is healthy; registration is lost
        (``registration_ok`` False: debris timers held, nothing learned) while the zone holds
        or is not anchored yet. A healthy zone without a matrix means the frame already is
        in the reference geometry (the tracker reads it the same way).
        """
        if not bool(zone.healthy):
            return None, False
        to_ref = zone.to_reference
        return (None if to_ref is None else np.asarray(to_ref, dtype=np.float64)), True

    def _debris_ignore_mask(self, shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        """``debris.ignore_polygons`` rasterised at processing resolution (cached)."""
        polygons = self.cfg.debris.ignore_polygons
        if not polygons:
            return None
        key = (tuple(shape_hw), self._scale)
        if self._debris_ignore is None or self._debris_ignore[0] != key:
            mask = np.zeros(shape_hw, dtype=np.uint8)
            for poly in polygons:
                pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2) * self._scale
                cv2.fillPoly(mask, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], 255)
            self._debris_ignore = (key, mask)
        return self._debris_ignore[1]

    @staticmethod
    def _snapshot_objects(objects: List[ObjectState]) -> List[ObjectState]:
        """Copy the tracker's live ObjectStates so a FrameResult does not mutate next frame."""
        return [dataclasses.replace(o, last_box=np.array(o.last_box, copy=True), history=list(o.history))
                for o in objects]

    def _degraded_reasons(self, zone: Any, debris: Any) -> Tuple[List[str], List[str]]:
        """Why this frame is not fully seen: (zone reasons, reasons of every other part).

        Stable texts (no numbers), so a state is the same from frame to frame:

        * "zone uninitialised" (no usable reference frame yet: the tracker uses the
          uncalibrated polygon) or "zone held" (the last good polygon, camera motion unknown);
        * "debris warming up", "debris held (train)" / "debris held (registration)" (timers
          frozen) or "debris untrusted" (registration or too much of the zone unknown); a
          registration hold caused by a zone hold is the zone reason, not a second one;
        * "detector interval > still gap": detector runs further apart than
          ``stuck.still_max_gap_seconds``, so the ID-free still clock cannot run;
        * "gate detector failing": the infrastructure detector raised on this frame.
        """
        zone_reasons: List[str] = []
        if zone.method == "uninitialized":
            zone_reasons.append("zone uninitialised")
        elif not bool(zone.healthy):
            zone_reasons.append("zone held")
        other: List[str] = []
        if debris is not None:
            if not debris.warmed_up:
                other.append(DEGRADED_DEBRIS_WARMUP)
            elif debris.held:
                if not (zone_reasons and debris.state == "held (registration)"):
                    other.append(f"debris {debris.state}" if str(debris.state).startswith("held") else "debris held")
            elif not debris.trusted:
                other.append("debris untrusted")
        stuck = self.cfg.stuck
        if (self._detect_interval is not None and stuck.stationarity != "speed"
                and self._detect_interval > float(stuck.still_max_gap_seconds) + 1e-6):
            other.append("detector interval > still gap")
        if self._gate_failing:
            other.append("gate detector failing")
        return zone_reasons, other

    def _report_degraded(self, reasons: Sequence[str], timestamp: float, frame_index: int) -> Optional[Event]:
        """HEALTH event once a change of the DEGRADED state has lasted ``alerts.health_debounce_seconds``.

        ``reasons`` are the ones this monitor reports itself: zone health has its own
        (equally debounced) HEALTH events, so it is left out unless ``zone.health_events``
        is off, and so do debris warm-ups ("debris warming up" is never reported here: the
        debris component announces the start and the end of every warm-up). Only the
        degraded/not degraded flag is debounced and reported; the event lists the reasons
        of the frame that reported it (``extra['reasons']``).
        """
        degraded = bool(reasons)
        t = float(timestamp)
        if degraded == self._degraded_reported:
            self._degraded_since = None
            return None
        if self._degraded_since is None or t < self._degraded_since:
            self._degraded_since = t
        lasted = t - self._degraded_since
        if lasted + 1e-9 < float(self.cfg.alerts.health_debounce_seconds):
            return None
        self._degraded_reported = degraded
        self._degraded_since = None
        text = "monitor degraded" if degraded else "monitor no longer degraded"
        if lasted > 0:
            text += f" for {lasted:.2f} s"
        if degraded:
            text += ": " + "; ".join(reasons)
        ev = Event(type="HEALTH", timestamp=t, frame_index=int(frame_index), status=self.engine.status.value,
                   reasons=[text], track_ids=[],
                   extra={"component": "monitor", "degraded": degraded, "reasons": list(reasons)})
        self.engine.emit(ev)
        return ev

    def _debris_trust_change(self, debris: Any, timestamp: float) -> Optional[float]:
        """Seconds a change of ``debris.trusted`` has lasted, once that reaches
        ``alerts.health_debounce_seconds`` (the caller then emits the debris HEALTH event);
        None otherwise.

        Debounced like ``_report_degraded``: a registration that fails every other frame
        would otherwise send two events per frame pair to every sink (and push queued
        WARNING / RUN events out of a full webhook queue). The value reported is the one the
        warm-up-complete event announced; nothing is tracked during a warm-up.
        """
        if not bool(debris.warmed_up):
            self._debris_trust_since = None
            return None
        if self._debris_trust_reported is None:  # no warm-up-complete event seen (cannot happen in order)
            self._debris_trust_reported = bool(debris.trusted)
            return None
        t = float(timestamp)
        if bool(debris.trusted) == self._debris_trust_reported:
            self._debris_trust_since = None
            return None
        if self._debris_trust_since is None or t < self._debris_trust_since:
            self._debris_trust_since = t
        lasted = t - self._debris_trust_since
        if lasted + 1e-9 < float(self.cfg.alerts.health_debounce_seconds):
            return None
        self._debris_trust_reported, self._debris_trust_since = bool(debris.trusted), None
        return lasted

    # ------------------------------------------------------------------ per frame
    def process_frame(self, frame_bgr: np.ndarray, timestamp: float, frame_index: int) -> FrameResult:
        """Run every stage on one frame and return the :class:`FrameResult`.

        ``timestamp`` is seconds (``frame_index / fps`` for files, monotonic for live),
        ``frame_index`` is the source frame number. Detector cadence uses an internal
        counter of processed frames so the detector always runs on the first frame after
        construction/reset, whatever the source index is. Frames may change resolution
        (they are resized to the size of the first one) but not aspect ratio
        (``FrameGeometryError``, raised before any state changes).
        """
        timings: Dict[str, float] = {}
        t_all = time.perf_counter()
        timestamp = float(timestamp)
        frame_index = int(frame_index)

        # (1) resize to the processing size the first frame fixed
        t0 = time.perf_counter()
        health_events: List[Event] = []
        frame = self._prepare_frame(frame_bgr)
        self.last_frame = frame
        resized = self._source_size_event(frame_bgr, timestamp, frame_index)
        if resized is not None:
            health_events.append(resized)
        timings["resize"] = (time.perf_counter() - t0) * 1000.0

        # (2) zone: refresh is allowed only when nothing tracked is in the zone and no
        # debris alarm is active (an obstruction must never become the new reference).
        t0 = time.perf_counter()
        if self.stabilizer is None:
            try:
                problems = self._build_stabilizer(frame_bgr.shape, frame.shape, timestamp)
            except ZoneGeometryError as exc:
                # The calibration cannot be mapped onto this stream (another aspect ratio, a
                # malformed reference_size): refuse to watch the wrong pixels, tell the sinks;
                # run() stops and the CLI exits 2.
                LOGGER.error("zone refused: %s", exc)
                self._processing_size = self._native_size = None  # the next frame starts over
                if str(exc) != self._zone_refusal:
                    self._zone_refusal = str(exc)
                    self.engine.emit(Event(type="HEALTH", timestamp=timestamp, frame_index=frame_index,
                                           status=self.engine.status.value, reasons=[f"zone refused: {exc}"],
                                           track_ids=[], extra={"zone_error": str(exc)}))
                raise
            for problem in problems:
                ev = Event(type="HEALTH", timestamp=timestamp, frame_index=frame_index,
                           status=self.engine.status.value, reasons=[problem], track_ids=[],
                           extra={"zone_setup": True})
                self.engine.emit(ev)
                health_events.append(ev)
        assert self.stabilizer is not None
        debris_active = self._last_debris is not None and bool(self._last_debris.triggered)
        scene_static = (not self.tracker.zone_occupied) and (not debris_active)
        zone = self.stabilizer.update(frame, timestamp, scene_static=scene_static)
        # Health changes that lasted zone.health_report_seconds, registration notices.
        for text in (self.stabilizer.health_reports if self.cfg.zone.health_events else []):
            ev = Event(type="HEALTH", timestamp=timestamp, frame_index=frame_index,
                       status=self.engine.status.value, reasons=[text], track_ids=[],
                       extra=dict(self.stabilizer.health()))
            self.engine.emit(ev)
            health_events.append(ev)
        timings["zone"] = (time.perf_counter() - t0) * 1000.0

        # (3) detector every N processed frames, otherwise velocity-propagated boxes
        t0 = time.perf_counter()
        every = max(1, int(self.cfg.detector.detect_every_n_frames))
        detector_ran = self._frames_processed % every == 0
        if detector_ran:
            detections: List[Detection] = list(self._get_detector().detect(frame))
            # Interval between detector runs, for the DEGRADED check against the still gap.
            if self._last_detect_t is not None and timestamp >= self._last_detect_t:
                self._detect_interval = timestamp - self._last_detect_t
            self._last_detect_t = timestamp
        else:
            detections = list(self.tracker.predict_boxes(timestamp))
        timings["detect"] = (time.perf_counter() - t0) * 1000.0

        # (4) tracker: only detector runs can age a track out; the ids the engine had in
        # ALARM are kept as hidden hazards if a train box covers them when they are lost.
        t0 = time.perf_counter()
        objects = self.tracker.update(detections, zone, timestamp, detector_ran=detector_ran,
                                      alarmed_ids=self.engine.alarm_episode_track_ids)
        gap = self.tracker.pop_detection_gap()
        if gap is not None:
            ev = Event(type="HEALTH", timestamp=timestamp, frame_index=frame_index,
                       status=self.engine.status.value,
                       reasons=[f"detector runs {gap:.2f} s apart, more than stuck.still_max_gap_seconds "
                                f"({self.cfg.stuck.still_max_gap_seconds:.2f} s): ID-free stationarity is off"],
                       track_ids=[], extra={"detection_gap_s": float(gap)})
            self.engine.emit(ev)
            health_events.append(ev)
        timings["track"] = (time.perf_counter() - t0) * 1000.0

        # (5) infrastructure / gates. A gate detector that raises must not cost the frame
        # (the stuck and debris rules work without it): no gate information this frame, the
        # normal thresholds apply and the monitor is DEGRADED ("gate detector failing").
        t0 = time.perf_counter()
        infra = self._get_infra()
        gate = GateState(available=False)
        if infra is not None:
            try:
                gate = infra.update(frame, timestamp, frame_index)
            except Exception:  # noqa: BLE001 - see above
                if not self._gate_failing:
                    LOGGER.exception("gate detector failed at t=%.2fs; continuing without gate information", timestamp)
                self._gate_failing = True
            else:
                if self._gate_failing:
                    LOGGER.info("gate detector working again at t=%.2fs", timestamp)
                self._gate_failing = False
        timings["infra"] = (time.perf_counter() - t0) * 1000.0

        # (6) debris: boxes of vehicles, persons and trains are excluded from the foreground
        # (an unmapped class such as 'cow' stays visible to the backstop); a train over the
        # zone holds the debris timers; the frame is registered to the background's geometry.
        t0 = time.perf_counter()
        debris = None
        if self.debris is not None:
            boxes, train_over_zone = self._debris_exclusions(detections, zone.mask, timestamp)
            to_reference, registration_ok = self._debris_registration(zone)
            prev_debris = self._last_debris
            debris = self.debris.update(frame, zone.mask, boxes, self._zone_empty_seconds(timestamp), timestamp,
                                        to_reference=to_reference, registration_ok=registration_ok,
                                        hold=train_over_zone, ignore_mask=self._debris_ignore_mask(frame.shape[:2]))
            self._last_debris = debris
            # The background model had to be re-warmed: report it, because debris detection
            # is blind until the new warm-up finishes.
            illumination = getattr(self.debris, "last_illumination_event", None)
            reasons: List[str] = []
            if illumination:
                extra = dict(illumination)
                if "area_fraction" in illumination:
                    pct = 100.0 * float(illumination.get("area_fraction", 0.0))
                    reasons = [f"background invalid: {pct:.0f}% of the zone changed at once; "
                               f"re-warming the background model"]
                else:
                    reasons = [f"background invalid ({illumination.get('reason')}); re-warming the background model"]
            elif not debris.warmed_up and (prev_debris is None or prev_debris.warmed_up):
                # First warm-up frame: start-up, reset(), or a model rebuilt because the frame
                # geometry or the zone's region changed. Debris is blind until it completes.
                dcfg = self.cfg.debris
                if dcfg.warmup_frames is not None:
                    span = f"{int(dcfg.warmup_frames)} frames"
                    extra = {"warming_up": True, "warmup_frames": int(dcfg.warmup_frames)}
                else:
                    span = f"{float(dcfg.warmup_seconds):.1f} s"
                    extra = {"warming_up": True, "warmup_seconds": float(dcfg.warmup_seconds)}
                reasons = [f"debris detection inactive until the background warm-up completes ({span})"]
            elif prev_debris is not None and not prev_debris.warmed_up and debris.warmed_up:
                extra = {"unknown_fraction": debris.unknown_fraction, "trusted": debris.trusted}
                reasons = [f"debris detection active (warm-up complete; "
                           f"{100.0 * debris.unknown_fraction:.0f}% of the zone not yet seen box-free"
                           f"{'' if debris.trusted else '; untrusted'})"]
                self._debris_trust_reported, self._debris_trust_since = bool(debris.trusted), None
            else:
                trust = self._debris_trust_change(debris, timestamp)
                if trust is not None:
                    lasted = trust
                    extra = {"unknown_fraction": debris.unknown_fraction, "trusted": debris.trusted,
                             "registered": debris.registered, "registration_ok": registration_ok}
                    reasons = [f"debris detection {'trusted again' if debris.trusted else 'untrusted'}"
                               f"{f' for {lasted:.2f} s' if lasted > 0 else ''}: {debris.state}"]
            if reasons:
                extra["component"] = "debris"
                ev = Event(type="HEALTH", timestamp=timestamp, frame_index=frame_index,
                           status=self.engine.status.value, reasons=reasons, track_ids=[], extra=extra)
                self.engine.emit(ev)
                health_events.append(ev)
        timings["debris"] = (time.perf_counter() - t0) * 1000.0

        # (7) DEGRADED: what keeps this frame from being fully seen; lasting changes of the
        # parts the zone does not report itself become a HEALTH event (component 'monitor').
        zone_reasons, other_reasons = self._degraded_reasons(zone, debris)
        degraded_reasons = tuple(zone_reasons + other_reasons)
        # The zone and the debris warm-ups report themselves (steps 2 and 6).
        reported = [r for r in (other_reasons if self.cfg.zone.health_events else zone_reasons + other_reasons)
                    if r != DEGRADED_DEBRIS_WARMUP]
        ev = self._report_degraded(reported, timestamp, frame_index)
        if ev is not None:
            health_events.append(ev)

        # (8) alerts
        t0 = time.perf_counter()
        assessment, events = self.engine.evaluate(objects, debris, gate, timestamp, frame_index)
        timings["alert"] = (time.perf_counter() - t0) * 1000.0

        # (9) bookkeeping
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
            degraded=bool(degraded_reasons),
            degraded_reasons=degraded_reasons,
        )
        self.last_result = result
        return result

    def reset(self, timestamp: Optional[float] = None, keep_alerts: bool = False) -> None:
        """Forget all state (new video, loop restart, live re-initialisation).

        The zone re-initialises on the next frame, whose size becomes the processing size.
        A WARNING or ALARM episode still open is closed first: its END event
        (``extra['reason'] = 'pipeline re-initialised'``, status CLEAR) goes to the sinks at
        ``timestamp`` - by default the last processed frame's - because nothing could end it
        afterwards. If the hazard is still there, the monitor raises it again once its
        clocks have run up. The DEGRADED state last reported is kept (it is what the
        receivers were told).

        ``keep_alerts=True`` re-initialises the perception only (zone, tracks, debris
        background, detectors) and leaves the alert engine as it is: no END event, the
        status, latch and open episode carry on in the same clock, and the episode ends the
        normal way (its latch or hold runs out with no new condition) or with RUN_END. For a
        live stream re-initialised on a new geometry: the hazard is probably still there,
        and a CLEAR sent because the pipeline restarted would be false. A looped file uses
        ``engine.reset()`` first instead (its RUN_END already closed the pass).
        """
        if not keep_alerts:
            last = self.last_result
            t_end = float(timestamp) if timestamp is not None else (last.timestamp if last is not None else 0.0)
            self.engine.close_open_episodes(t_end, last.frame_index if last is not None else -1,
                                            reason="pipeline re-initialised")
        self.stabilizer = None
        self.tracker.reset()
        if self.debris is not None:
            self.debris.reset()
        for component in (self._detector, self._infra):
            if component is not None and hasattr(component, "reset"):
                try:
                    component.reset()
                except Exception:  # a detector reset failure must not abort the run
                    LOGGER.exception("%s.reset() failed", type(component).__name__)
        self._frames_processed = 0
        self._last_debris = None
        self.last_result = None
        self._debris_mapped_ids = {}
        self._last_detect_t = None
        self._detect_interval = None
        self._gate_failing = False
        self._degraded_since = None
        self._debris_trust_reported = None
        self._debris_trust_since = None
        self._processing_size = None
        self._native_size = None

    def close(self) -> None:
        """Close the sinks built from the config: ``sink.close()`` on every one of them.

        That flushes stdout, closes the JSONL file and drains the webhook queue for up to
        one per-event delivery deadline (``WebhookSink.close``), so a final ALARM_END or
        RUN_END is not cut off at shutdown. Sinks passed to the constructor belong to the
        caller and are left open.
        """
        for sink in self._config_sinks:
            try:
                sink.close()
            except Exception:
                LOGGER.exception("closing sink %s failed", type(sink).__name__)

    # ------------------------------------------------------------------ run loop
    # Summary dict of the last run(); also set when run() raises (a refused zone).
    last_summary: Optional[Dict[str, Any]] = None

    def run(self) -> Dict[str, Any]:
        """Open ``cfg.video.source`` and process it to the end (or ``max_frames``).

        Returns a summary dict with at least ``frames`` (processed), ``alarms`` (non-repeat
        ALARM_START events) and ``fps`` (processed frames per wall-clock second), plus
        ``warnings``, ``health_events``, ``elapsed_s``, ``reconnects``, ``errors``,
        ``loops``, ``final_status``, ``interrupted``, mean per-stage ``timings_ms``,
        ``run_id``, ``stop_reason`` (one of the ``STOP_*`` constants) and ``aborted``: True
        when the crossing stopped being watched before the input ended (repeated frame
        errors, a live source that could not be reconnected). Ctrl-C (and, under ``main``,
        SIGTERM / SIGHUP) stops the loop cleanly and still returns the summary.
        ``max_frames`` of ``None`` or ``<= 0`` means "run to the end / forever".

        The output directories are created before the first read (``RuntimeError`` when
        impossible). The detector (and the infrastructure detector when enabled) is built
        on the first frame right after the zone set-up, so a refused zone is reported
        first; a model that cannot be loaded (missing weights, no ultralytics) ends the run
        at once with that one exception (stop reason ``detector_unavailable``) instead of
        ``MAX_CONSECUTIVE_FRAME_ERRORS`` tracebacks and a run that looks finished.

        Events through every sink: RUN_START before the first frame and RUN_END (frames,
        seconds of video covered, last timestamp, final status, stop reason) after the
        last one - also when the run aborts, is interrupted or raises; a looped file gets
        one RUN_START/RUN_END pair per pass. When the input ends during WARNING or ALARM,
        RUN_END is the closing event: no ALARM_END is invented. HEALTH events
        (``extra.component`` "source" or "pipeline") report a lost live stream, the outcome
        of every reconnect attempt, the give-up, an abort after repeated frame errors and
        an output that failed and was switched off.

        Robustness: a frame that makes any stage raise is counted in ``errors``, logged
        with its traceback and skipped; the loop only gives up after
        ``MAX_CONSECUTIVE_FRAME_ERRORS`` failures in a row. A crossing that stops being
        watched because of one bad decoded frame or one transient inference error is a
        worse outcome than a gap in the log. A live stream that changes resolution is
        resized to the first frame's processing size (``process_frame`` step 1: nothing is
        lost or reset); one that changes aspect ratio (``FrameGeometryError``) costs one
        frame and re-initialises the perception with the alert state kept
        (``reset(keep_alerts=True)``, a HEALTH event with ``extra['reset']``). Each output
        (overlay, snapshots, video, display, status line) is guarded on its own. A
        ``ZoneGeometryError`` (the calibration does not fit the stream) ends the run with
        stop reason ``zone_refused`` and is re-raised after RUN_END (``main`` exits 2).

        Timestamps: a file frame gets its container time (CAP_PROP_POS_MSEC) while that is
        positive and increasing - correct on variable-frame-rate files - and otherwise
        continues 1 / fps after the previous frame, which is ``frame_index / fps`` when the
        container has no usable clock; ``video.fps_override`` forces ``frame_index / fps``.
        A live frame gets the wall-clock seconds since the start of the run.

        The config-built sinks are closed when ``run`` returns, so use one ``Monitor``
        per run (construct a new one to run again). ``run`` never writes to ``cfg``.
        """
        cap, fps, is_live = open_source(self.cfg.video)
        self.fps = fps
        try:
            outputs = _RunOutputs(self, fps)
        except BaseException:
            _release(cap)
            self.close()
            raise
        st = self._new_run_state(cap, fps, is_live)
        self._emit_run_start(st)
        try:
            self._run_loop(st, outputs)
        except KeyboardInterrupt as exc:
            st.stop_reason = STOP_INTERRUPTED
            LOGGER.info("interrupted (%s)", str(exc) or "Ctrl-C")
        except ZoneGeometryError:
            st.stop_reason, st.aborted = STOP_ZONE_REFUSED, True
            raise  # the zone must be recalibrated for this stream: main() exits 2
        except BaseException:
            if not st.aborted:  # detector_unavailable is set where it happens
                st.stop_reason, st.aborted = STOP_ERROR, True
            raise
        finally:
            _release(st.cap)
            outputs.close()
            self._emit_run_end(st)
            self.close()
            self.last_summary = self._run_summary(st)
        return self.last_summary

    def _new_run_state(self, cap: Any, fps: float, is_live: bool) -> "_RunState":
        video = self.cfg.video
        index0 = int(round(float(video.start_seconds) * fps)) if (not is_live and video.start_seconds > 0) else 0
        max_frames = int(video.max_frames) if (video.max_frames is not None and int(video.max_frames) > 0) else None
        forced_fps = video.fps_override is not None and float(video.fps_override) > 0
        run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        now = time.monotonic()
        return _RunState(cap=cap, fps=float(fps), is_live=bool(is_live), index0=index0, frame_index=index0,
                         last_index=index0, max_frames=max_frames, run_id=run_id, pass_id=run_id,
                         config_sha256=config_sha256(self.cfg), container_clock=not is_live and not forced_fps,
                         wall_start=now, t_live0=now)

    def _run_loop(self, st: "_RunState", outputs: "_RunOutputs") -> None:
        """Read, process and output frames until a stop condition sets ``st.stop_reason``."""
        while True:
            if st.max_frames is not None and st.processed >= st.max_frames:
                st.stop_reason = STOP_MAX_FRAMES
                return
            loop_t0 = time.perf_counter()
            ok, frame = st.cap.read()
            if not ok or frame is None:
                if self._on_read_failure(st):
                    continue
                return
            st.pass_read += 1
            timestamp = self._frame_timestamp(st)
            result = self._process_guarded(frame, timestamp, st)
            if result is None:
                if st.aborted:
                    return
                st.frame_index += 1
                continue
            alarm_started = self._tally(st, result)
            if not outputs.write(result, alarm_started, st):
                st.stop_reason = STOP_USER_QUIT
                return
            dt = time.perf_counter() - loop_t0
            inst = 1.0 / dt if dt > 0 else 0.0
            st.fps_est = inst if st.fps_est is None else 0.9 * st.fps_est + 0.1 * inst
            outputs.status_line(result, st)
            st.frame_index += 1

    def _on_read_failure(self, st: "_RunState") -> bool:
        """End of a file, a looped file's restart or a lost live stream; True = keep reading."""
        if st.is_live:
            cap = self._reconnect(st.cap, st)
            if cap is None:
                st.stop_reason, st.aborted = STOP_RECONNECT_FAILED, True
                return False
            st.cap = cap
            st.reconnects += 1
            return True
        if self.cfg.video.loop and st.pass_read > 0:
            self._restart_pass(st)
            return True
        st.stop_reason = STOP_END_OF_INPUT
        return False

    def _restart_pass(self, st: "_RunState") -> None:
        """Re-open a looped file: RUN_END for the pass, all state forgotten, a new RUN_START."""
        st.loops += 1
        LOGGER.info("end of file; looping (pass %d)", st.loops + 1)
        _release(st.cap)
        cap, fps, _ = open_source(self.cfg.video)  # raising here ends the run (RUN_END "error")
        st.stop_reason = STOP_LOOP
        self._emit_run_end(st)
        st.cap, st.fps = cap, float(fps)
        # RUN_END closed the pass, open WARNING/ALARM included (the event-log contract): no
        # END may follow it. Then timestamps restart with the file; all state is forgotten.
        self.engine.reset()
        self.reset()
        st.start_pass(f"{st.run_id}-p{st.loops + 1}")
        self._emit_run_start(st)

    def _frame_timestamp(self, st: "_RunState") -> float:
        """Timestamp of the frame just read (see "Timestamps" in ``run``)."""
        if st.is_live:
            ts = time.monotonic() - st.t_live0
        else:
            nominal = st.frame_index / st.fps
            ts = nominal
            if st.container_clock:
                pos = _container_seconds(st.cap)
                if pos is not None and pos > 0.0 and (st.last_ts is None or pos > st.last_ts):
                    ts = pos
                elif st.last_ts is not None:
                    ts = st.last_ts + 1.0 / st.fps  # no usable container time: keep the nominal pace
                if not st.clock_warned and abs(ts - nominal) > CLOCK_DIVERGENCE_WARN_S:
                    st.clock_warned = True
                    LOGGER.warning("frame %d: the container clock (%.3f s) and frame_index / fps (%.3f s) differ by "
                                   "more than %.1f s (variable frame rate?); timestamps follow the container clock",
                                   st.frame_index, ts, nominal, CLOCK_DIVERGENCE_WARN_S)
        if st.first_ts is None:
            st.first_ts = ts
        st.last_ts = ts
        st.last_index = st.frame_index
        return ts

    def _process_guarded(self, frame: np.ndarray, timestamp: float, st: "_RunState") -> Optional[FrameResult]:
        """``process_frame`` under the per-frame error policy; None when the frame was lost."""
        try:
            result = self.process_frame(frame, timestamp, st.frame_index)
        except (KeyboardInterrupt, ZoneGeometryError):
            raise  # a zone that cannot be mapped onto this stream: stop (main() exits 2)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the monitor
            self._require_models(timestamp, st)
            self._on_frame_error(exc, timestamp, st)
            return None
        st.consecutive_errors = 0
        return result

    def _require_models(self, timestamp: float, st: "_RunState") -> None:
        """After a failed frame: build the detectors that do not exist yet; fatal if impossible.

        ``process_frame`` builds them on the first frame, after the zone set-up. When that
        frame failed and one is still missing, building it here tells a model that cannot
        be loaded (raised: the run ends with that one error) from a transient frame error.
        """
        if self._detector is not None and (self._infra is not None or not self.cfg.infrastructure.enabled):
            return
        try:
            self._get_detector()
            self._get_infra()
        except Exception as exc:
            LOGGER.error("cannot load the detector: %s", exc)
            st.stop_reason, st.aborted = STOP_DETECTOR, True
            self._run_health(st, timestamp, f"monitoring stopped: the detector cannot be loaded ({exc})",
                             "pipeline", error=str(exc))
            raise

    def _on_frame_error(self, exc: Exception, timestamp: float, st: "_RunState") -> None:
        """Count a lost frame; re-initialise after a live aspect-ratio change; abort when systematic."""
        st.errors += 1
        st.consecutive_errors += 1
        LOGGER.exception("frame %d failed (%d consecutive error(s)); continuing",
                         st.frame_index, st.consecutive_errors)
        if st.is_live and isinstance(exc, FrameGeometryError):
            # The stream now has another aspect ratio (a new resolution with the same one is
            # resized in process_frame and never gets here; any other error is a lost frame
            # like the rest). Re-initialise the perception on the new geometry - the zone
            # set-up on the next frame decides whether the calibration fits it, else the run
            # stops with zone_refused - but keep the alert state: the hazard is probably
            # still there, so no CLEAR END is sent, and the debris background is re-learned
            # with whatever is in view now.
            prior = self.engine.status.value
            debris_alarm = self._last_debris is not None and bool(self._last_debris.triggered)
            LOGGER.warning("re-initialising the pipeline on the new frame geometry (status %s kept; tracks and "
                           "background model are discarded)", prior)
            self.reset(keep_alerts=True)
            text = f"frame rejected ({exc}); pipeline re-initialised on the new geometry, status {prior} kept"
            if self.debris is not None:
                text += ("; the debris background is re-learned with the current view: debris detection is "
                         "untrusted for an obstruction that is there now" if debris_alarm else
                         "; the debris background is re-learned")
            ev = Event(type="HEALTH", timestamp=timestamp, frame_index=st.frame_index,
                       status=self.engine.status.value, reasons=[text], track_ids=[],
                       extra={"error": str(exc), "reset": True, "previous_status": prior,
                              "debris_untrusted": debris_alarm})
            self.engine.emit(ev)
            st.health += 1
        if st.consecutive_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
            LOGGER.error("aborting after %d consecutive frame errors", st.consecutive_errors)
            st.stop_reason, st.aborted = STOP_FRAME_ERRORS, True
            self._run_health(st, timestamp, f"monitoring stopped: {st.consecutive_errors} consecutive frame "
                                            f"errors (last: {exc})", "pipeline",
                             consecutive_errors=st.consecutive_errors, error=str(exc))

    @staticmethod
    def _tally(st: "_RunState", result: FrameResult) -> bool:
        """Count one processed frame and its events; True when it started a (non-repeat) ALARM."""
        st.processed += 1
        st.pass_frames += 1
        st.final_status = result.assessment.status.value
        for k, v in result.timings_ms.items():
            st.sums[k] = st.sums.get(k, 0.0) + v
        alarm_started = False
        for ev in result.events:
            if ev.type == "ALARM_START" and not ev.extra.get("repeat"):
                st.alarms += 1
                alarm_started = True
            elif ev.type == "WARNING_START":
                st.warnings += 1
            elif ev.type == "HEALTH":
                st.health += 1
        return alarm_started

    def _reconnect(self, cap: Any, st: "_RunState") -> Optional[Any]:
        """Re-open a live source with exponential back-off; None after RECONNECT_ATTEMPTS.

        Nobody is watching the crossing meanwhile, so the sinks hear about every step
        (HEALTH, ``component`` "source"): the loss, each attempt's outcome and the give-up.
        """
        source = str(self.cfg.video.source)
        shown = _redact_source(source)
        self._run_health(st, self._source_time(st), f"stream lost: reading {shown} failed; reconnecting",
                         "source", state="lost")
        for attempt in range(1, RECONNECT_ATTEMPTS + 1):
            delay = RECONNECT_BACKOFF_S * (2 ** (attempt - 1))
            LOGGER.warning("live source %r read failed; reconnect attempt %d/%d in %.0fs",
                           shown, attempt, RECONNECT_ATTEMPTS, delay)
            _release(cap)
            time.sleep(delay)
            try:
                cap = _make_capture(source, float(self.cfg.video.stream_timeout_seconds or 0.0))
                ok = bool(cap.isOpened()) and bool(cap.read()[0])
            except Exception:  # noqa: BLE001 - a capture that throws is a failed attempt
                ok = False
            self._run_health(st, self._source_time(st),
                             f"reconnect attempt {attempt}/{RECONNECT_ATTEMPTS} to {shown} "
                             f"{'succeeded' if ok else 'failed'}", "source",
                             state="reconnected" if ok else "attempt_failed", attempt=attempt)
            if ok:
                LOGGER.info("reconnected to %r", shown)
                return cap
        LOGGER.error("giving up on %r after %d reconnect attempts", shown, RECONNECT_ATTEMPTS)
        _release(cap)
        self._run_health(st, self._source_time(st),
                         f"monitoring stopped: {shown} could not be reconnected after {RECONNECT_ATTEMPTS} "
                         f"attempts", "source", state="gave_up", attempts=RECONNECT_ATTEMPTS)
        return None

    @staticmethod
    def _source_time(st: "_RunState") -> float:
        """Timestamp for an event that is not tied to a decoded frame (reconnects)."""
        if st.is_live:
            return time.monotonic() - st.t_live0
        return st.last_ts if st.last_ts is not None else st.index0 / st.fps

    def _run_health(self, st: "_RunState", timestamp: float, text: str, component: str, **extra: Any) -> None:
        """Emit a HEALTH event of the run loop (``extra.component`` = ``component``) to every sink."""
        extra["component"] = component
        self.engine.emit(Event(type="HEALTH", timestamp=float(timestamp), frame_index=int(st.frame_index),
                               status=self.engine.status.value, reasons=[text], track_ids=[], extra=extra))
        st.health += 1

    def _emit_run_start(self, st: "_RunState") -> None:
        """RUN_START (the event-log contract in ``gradecrossing.evaluate``) for the current pass."""
        source = _redact_source(self.cfg.video.source)
        t0 = 0.0 if st.is_live else st.index0 / st.fps
        self.engine.emit(Event(type="RUN_START", timestamp=t0, frame_index=st.frame_index,
                               status=self.engine.status.value, reasons=[f"monitoring started: {source}"],
                               track_ids=[], extra={"run_id": st.pass_id, "source": source,
                                                    "config_sha256": st.config_sha256,
                                                    "package_version": __version__}))

    def _emit_run_end(self, st: "_RunState") -> None:
        """RUN_END of the current pass: the closing event of the run, whatever the status is."""
        status = self.engine.status.value
        t = st.last_ts if st.last_ts is not None else (0.0 if st.is_live else st.index0 / st.fps)
        self.engine.emit(Event(type="RUN_END", timestamp=float(t), frame_index=int(st.last_index), status=status,
                               reasons=[f"monitoring stopped: {st.stop_reason}"], track_ids=[],
                               extra={"run_id": st.pass_id, "frames": st.pass_frames,
                                      "duration_s": round(self._covered_seconds(st), 6),
                                      "last_timestamp": st.last_ts, "final_status": status,
                                      "stop_reason": st.stop_reason}))

    @staticmethod
    def _covered_seconds(st: "_RunState") -> float:
        """Seconds of video the current pass covered: for a file the span of its frames plus
        the last frame's 1 / fps (= frames / fps at a constant rate); live: wall clock."""
        if st.is_live:
            return max(0.0, time.monotonic() - st.t_live0)
        if st.first_ts is None or st.last_ts is None:
            return 0.0
        return max(0.0, st.last_ts - st.first_ts) + 1.0 / st.fps

    def _run_summary(self, st: "_RunState") -> Dict[str, Any]:
        elapsed = time.monotonic() - st.wall_start
        processed = st.processed
        summary: Dict[str, Any] = {
            "frames": processed,
            "alarms": st.alarms,
            "fps": (processed / elapsed) if elapsed > 0 else 0.0,
            "warnings": st.warnings,
            "health_events": st.health,
            "elapsed_s": elapsed,
            "reconnects": st.reconnects,
            "errors": st.errors,
            "loops": st.loops,
            "final_status": st.final_status,
            "interrupted": st.stop_reason == STOP_INTERRUPTED,
            "aborted": st.aborted,
            "stop_reason": st.stop_reason,
            "run_id": st.run_id,
            "timings_ms": {k: v / processed for k, v in st.sums.items()} if processed else {},
        }
        LOGGER.log(logging.ERROR if st.aborted else logging.INFO,
                   "run finished (%s%s): %d frames, %d alarms, %.1f fps", st.stop_reason,
                   ", aborted" if st.aborted else "", processed, st.alarms, summary["fps"])
        return summary

    @staticmethod
    def _status_line(result: FrameResult, fps_est: Optional[float]) -> str:
        """One ASCII status line for the periodic console print (``DEGRADED(...)`` after the
        status while the monitor cannot fully see; a zone not anchored yet reads NOT ANCHORED)."""
        in_zone = sum(1 for o in result.objects if o.in_zone)
        debris = result.debris
        if debris is None:
            debris_txt = "off"
        elif not debris.warmed_up or debris.held or not debris.trusted:
            debris_txt = debris.state or ("warmup" if not debris.warmed_up else "-")
        else:
            debris_txt = f"{100.0 * debris.area_fraction:.1f}%"
        zone = result.zone
        if zone.healthy:
            zone_flag = ""
        else:
            zone_flag = " NOT ANCHORED" if zone.method == "uninitialized" else " HELD"
        degraded_txt = f"DEGRADED({'; '.join(result.degraded_reasons)}) " if result.degraded else ""
        total = result.timings_ms.get("total", 0.0)
        fps_txt = f" {fps_est:.1f}fps" if fps_est else ""
        return ascii_safe(f"[f={result.frame_index} t={result.timestamp:.1f}s] {result.assessment.status.value:<7} "
                          f"{degraded_txt}objs={len(result.objects)} inzone={in_zone} debris={debris_txt} "
                          f"zone={zone.method}/{zone.inliers}{zone_flag} "
                          f"{total:.1f}ms{fps_txt}")


# --------------------------------------------------------------------------------------
# Run state and outputs (Monitor.run)
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class _RunState:
    """Book-keeping of one ``Monitor.run`` call (the config itself is never written to)."""

    cap: Any
    fps: float
    is_live: bool
    index0: int  # source frame index of the first frame (start_seconds * fps for a seek)
    frame_index: int
    last_index: int  # source frame index of the last frame read
    max_frames: Optional[int]
    run_id: str
    pass_id: str  # run_id of the current RUN_START: run_id, then "<run_id>-p<N>" for loop pass N
    config_sha256: str
    container_clock: bool  # file timestamps from CAP_PROP_POS_MSEC (False: frame_index / fps)
    wall_start: float
    t_live0: float
    processed: int = 0
    alarms: int = 0
    warnings: int = 0
    health: int = 0
    reconnects: int = 0
    loops: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    sums: Dict[str, float] = dataclasses.field(default_factory=dict)
    fps_est: Optional[float] = None
    final_status: str = "CLEAR"
    stop_reason: str = STOP_END_OF_INPUT
    aborted: bool = False
    clock_warned: bool = False
    # The current pass (between one RUN_START and its RUN_END).
    pass_read: int = 0  # frames read
    pass_frames: int = 0  # frames processed
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None

    def start_pass(self, pass_id: str) -> None:
        """Start the next pass over a looped file."""
        self.pass_id = pass_id
        self.frame_index = self.last_index = self.index0
        self.pass_read = self.pass_frames = 0
        self.first_ts = self.last_ts = None
        self.stop_reason = STOP_END_OF_INPUT


class _RunOutputs:
    """The optional per-frame outputs of ``Monitor.run``: overlay, alarm snapshots,
    annotated video, display window and the periodic status line.

    Each one is guarded on its own: when it fails it is logged once, reported with one
    HEALTH event (``component`` "pipeline", ``output`` naming it) and switched off for the
    rest of the run while monitoring continues - a full disk or a missing GUI must never
    stop the crossing from being watched. Without the overlay the other outputs get the
    plain processed frame. The constructor creates the output directories
    (``RuntimeError`` when impossible); the config is only read.
    """

    def __init__(self, monitor: Monitor, fps: float) -> None:
        out = monitor.cfg.output
        self._mon = monitor
        self._fps = float(fps)
        self.display = bool(out.display)
        self.save_video = str(out.save_video or "")
        self.snapshot_dir = str(out.snapshot_dir or "")
        self.print_every = int(out.print_every_n_frames or 0)
        self.overlay = True
        self._writer: Optional[Any] = None
        self._window = False  # a window was shown (destroyed at the end)
        _prepare_output_paths(snapshot_dir=self.snapshot_dir, save_video=self.save_video)

    def write(self, result: FrameResult, alarm_started: bool, st: _RunState) -> bool:
        """Render one processed frame to the enabled outputs; False when the user quit (q/Esc)."""
        snapshot = alarm_started and bool(self.snapshot_dir)
        if not (self.display or self.save_video or snapshot):
            return True
        image = self._draw(result, st)
        if snapshot:
            self._snapshot(image, result, st)
        if self.save_video:
            self._video(image, st)
        if self.display:
            return self._show(image, st)
        return True

    def status_line(self, result: FrameResult, st: _RunState) -> None:
        if self.print_every > 0 and st.processed % self.print_every == 0:
            try:
                print(self._mon._status_line(result, st.fps_est), flush=True)
            except Exception as exc:  # noqa: BLE001 - a closed stdout must not stop the monitor
                self.print_every = 0
                self._disable("status_line", exc, st)

    def close(self) -> None:
        if self._writer is not None:
            _release(self._writer)
            self._writer = None
        if self._window:
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass

    def _draw(self, result: FrameResult, st: _RunState) -> np.ndarray:
        frame = self._mon.last_frame
        if not self.overlay:
            return frame
        try:
            stuck = self._mon.cfg.stuck
            footprint = stuck.footprint_fraction if stuck.zone_test == "footprint" else None
            return draw_overlay(frame, result, st.fps_est, footprint_fraction=footprint)
        except Exception as exc:  # noqa: BLE001
            self.overlay = False
            self._disable("overlay", exc, st)
            return frame

    def _snapshot(self, image: np.ndarray, result: FrameResult, st: _RunState) -> None:
        path = Path(self.snapshot_dir) / f"alarm_f{result.frame_index:07d}_t{result.timestamp:08.2f}s.jpg"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"cv2.imwrite could not write {path}")
            LOGGER.info("snapshot %s", path)
        except Exception as exc:  # noqa: BLE001
            self.snapshot_dir = ""
            self._disable("snapshot", exc, st)

    def _video(self, image: np.ndarray, st: _RunState) -> None:
        try:
            if self._writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(self.save_video, cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
                if not writer.isOpened():
                    _release(writer)
                    raise RuntimeError(f"cannot open video writer {self.save_video}")
                self._writer = writer
            self._writer.write(image)
        except Exception as exc:  # noqa: BLE001
            self.save_video = ""
            if self._writer is not None:
                _release(self._writer)
                self._writer = None
            self._disable("save_video", exc, st)

    def _show(self, image: np.ndarray, st: _RunState) -> bool:
        try:
            cv2.imshow(WINDOW_NAME, image)
            self._window = True
            key = cv2.waitKey(1) & 0xFF
        except Exception as exc:  # noqa: BLE001 - cv2.error on opencv-python-headless (CI, Jetson without X)
            self.display = False
            self._disable("display", exc, st)
            return True
        if key in (27, ord("q")):
            LOGGER.info("display window closed by user")
            return False
        return True

    def _disable(self, name: str, exc: Exception, st: _RunState) -> None:
        LOGGER.error("output %s failed (%s); disabled for the rest of the run, monitoring continues", name, exc)
        self._mon._run_health(st, st.last_ts if st.last_ts is not None else 0.0,
                              f"output {name} disabled: {exc}", "pipeline", output=name, error=str(exc))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
class CliOverride(NamedTuple):
    """One command-line flag that overrides config values (a row of ``CLI_OVERRIDES``).

    ``type`` converts the flag's argument (``str``, ``int``, ``float`` or a function);
    ``None`` makes the flag an on/off switch that writes ``const``. The value goes to every
    dotted path in ``paths``, and the ``(path, value)`` pairs in ``also`` are written
    whenever the flag is given.
    """

    flag: str
    paths: Tuple[str, ...]
    type: Optional[Callable[[str], Any]]
    help: str
    metavar: Optional[str] = None
    const: Any = True
    also: Tuple[Tuple[str, Any], ...] = ()

    @property
    def dest(self) -> str:
        """The argparse attribute name (``--max-frames`` -> ``max_frames``)."""
        return self.flag.lstrip("-").replace("-", "_")


def _frame_count(text: str) -> int:
    """argparse type for frame cadences: an integer; values below 1 mean 1."""
    try:
        return max(1, int(text))
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None


# Every flag of ``python -m gradecrossing.monitor`` that overrides the YAML, in --help order.
# build_parser and apply_cli_overrides are generated from it; tools/benchmark.py reuses rows.
CLI_OVERRIDES: Tuple[CliOverride, ...] = (
    CliOverride("--source", ("video.source",), str,
                "video file, camera index or stream URL (overrides video.source)"),
    CliOverride("--weights", ("detector.weights",), str,
                "vehicle detector weights .pt/.engine (overrides detector.weights)"),
    CliOverride("--display", ("output.display",), None, "show an annotated window (needs an OpenCV GUI build)"),
    CliOverride("--save", ("output.save_video",), str,
                "write the annotated video (overrides output.save_video)", "OUT.mp4"),
    CliOverride("--events", ("alerts.events_jsonl",), str,
                "append events as JSON lines (overrides alerts.events_jsonl)", "events.jsonl"),
    CliOverride("--max-frames", ("video.max_frames",), int, "stop after N frames (overrides video.max_frames)", "N"),
    CliOverride("--infra-weights", ("infrastructure.weights",), str, "enable the gate detector with these weights",
                "W", also=(("infrastructure.enabled", True),)),
    CliOverride("--infra", ("infrastructure.enabled",), None, "enable the gate detector with the configured weights"),
    CliOverride("--zone", ("zone.polygon_file",), zone_file_argument,
                'zone polygon file, team format {"vertices": [[x, y], ...]} (overrides zone.polygon_file)',
                "ZONE.json"),
    CliOverride("--no-stabilize", ("zone.stabilize",), None, "disable zone stabilisation (zone.stabilize=false)",
                const=False),
    CliOverride("--print-every", ("output.print_every_n_frames",), int, "print a status line every N frames", "N"),
    CliOverride("--fps", ("video.fps_override",), float,
                "force the source frame rate; file timestamps become frame_index / FPS "
                "(overrides video.fps_override)"),
    CliOverride("--resize-width", ("video.resize_width",), int, "process at this width (zone polygon is scaled)", "W"),
    CliOverride("--start-seconds", ("video.start_seconds",), float, "seek before processing (files only)"),
    CliOverride("--loop", ("video.loop",), None, "loop a file (soak test)"),
    CliOverride("--live", ("video.live",), None, "treat the source as live (wall-clock timestamps, reconnect)"),
    CliOverride("--detect-every", ("detector.detect_every_n_frames",), _frame_count,
                "run the detector every N frames", "N"),
    CliOverride("--device", ("detector.device", "infrastructure.device"), str,
                "inference device for the detectors ('' auto, '0' CUDA, 'cpu')"),
    CliOverride("--snapshot-dir", ("output.snapshot_dir",), str, "save an annotated JPEG on every ALARM_START", "DIR"),
    CliOverride("--webhook", ("alerts.webhook_url",), str, "POST events to this URL (overrides alerts.webhook_url)",
                "URL"),
    CliOverride("--no-stdout", ("alerts.stdout",), None, "do not print events to stdout", const=False),
    CliOverride("--log-level", ("output.log_level",), str, "logging level (overrides output.log_level)"),
)

EXIT_STATUS_HELP = """\
exit status:
  0  the input ended (end of file, --max-frames) or the run was stopped
     (Ctrl-C, q/Esc in the display window, SIGTERM/SIGHUP: systemctl or docker stop)
  1  the crossing stopped being watched, or watching never started: the source
     cannot be opened, the detector cannot be loaded (missing weights, no
     ultralytics), an output path cannot be created, too many consecutive frame
     errors, a live source that could not be reconnected (summary: ABORTED)
  2  configuration or usage error: the config cannot be loaded or an override
     applied, the monitor rejects a setting, the zone calibration does not fit
     the stream (zone refused)"""


def add_cli_arguments(parser: argparse.ArgumentParser,
                      overrides: Sequence[CliOverride] = CLI_OVERRIDES) -> argparse.ArgumentParser:
    """Add one argparse option per row of ``overrides`` (default: every monitor flag)."""
    for row in overrides:
        kwargs: Dict[str, Any] = {"dest": row.dest, "help": row.help}
        if row.type is None:
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = row.type
            if row.metavar:
                kwargs["metavar"] = row.metavar
        parser.add_argument(row.flag, **kwargs)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for ``python -m gradecrossing.monitor``; every flag overrides the YAML."""
    p = argparse.ArgumentParser(
        prog="python -m gradecrossing.monitor",
        description="AI rail grade-crossing monitor: stuck-vehicle, pedestrian and debris alerts "
                    "on a video file, camera or RTSP stream.",
        epilog="Example: python -m gradecrossing.monitor --config configs/stresstest.yaml --events events.jsonl"
               "\n\n" + EXIT_STATUS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", metavar="YAML", help="config file (default: built-in defaults; see configs/default.yaml)")
    add_cli_arguments(p)
    return p


def _set_config_path(cfg: MonitorConfig, path: str, value: Any) -> None:
    section, _, key = path.partition(".")
    target = getattr(cfg, section)
    if not hasattr(target, key):
        raise AttributeError(f"unknown config key {path!r}")
    setattr(target, key, value)


def apply_cli_overrides(cfg: MonitorConfig, args: argparse.Namespace,
                        overrides: Sequence[CliOverride] = CLI_OVERRIDES) -> MonitorConfig:
    """Copy every CLI flag that was given onto ``cfg`` (in place) and return it.

    ``overrides`` is the table the parser was built from; a row whose flag ``args`` does
    not have is skipped, so a tool can reuse part of ``CLI_OVERRIDES``.
    """
    for row in overrides:
        value = getattr(args, row.dest, None)
        if value is None or value is False:
            continue  # not given
        if row.type is None:
            value = row.const
        for path in row.paths:
            _set_config_path(cfg, path, value)
        for path, const in row.also:
            _set_config_path(cfg, path, const)
    return cfg


def format_summary(summary: Dict[str, Any]) -> str:
    """ASCII one-liner (plus per-stage ms) for the end of a run; says ABORTED and why."""
    stages = summary.get("timings_ms") or {}
    stage_txt = " ".join(f"{k}={v:.1f}ms" for k, v in stages.items())
    errors = int(summary.get("errors", 0) or 0)
    aborted = bool(summary.get("aborted"))
    return (f"summary: frames={summary.get('frames', 0)} alarms={summary.get('alarms', 0)} "
            f"warnings={summary.get('warnings', 0)} fps={summary.get('fps', 0.0):.1f} "
            f"elapsed={summary.get('elapsed_s', 0.0):.1f}s final={summary.get('final_status', '-')}"
            + (f" ABORTED stop_reason={summary.get('stop_reason', '-')}" if aborted else "")
            + (f" errors={errors}" if errors else "")  # only mentioned when frames were lost
            + (f" | {stage_txt}" if stage_txt else ""))


def one_line(exc: BaseException) -> str:
    """``str(exc)`` on one line (a YAML syntax error spans several) for an ``error:`` message."""
    return " ".join(str(exc).split())


def _stop_on_signal(signum: int, _frame: Any) -> None:
    """Handler of ``STOP_SIGNALS``: stop the run exactly like Ctrl-C."""
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = f"signal {signum}"
    raise KeyboardInterrupt(name)


def _install_stop_handlers() -> List[Tuple[int, Any]]:
    """Make the ``STOP_SIGNALS`` that exist here stop the run like Ctrl-C.

    Their default action kills the process at once: ``run``'s ``finally`` never runs, so
    the receivers get no RUN_END, an open WARNING/ALARM is left open and the queued webhook
    events are lost. As a KeyboardInterrupt, ``run`` ends with stop reason
    ``interrupted``, emits RUN_END, closes the sinks (the webhook drain takes at most one
    per-event deadline, well inside systemd's 90 s stop timeout) and ``main`` exits 0.
    Returns the ``(signal, previous handler)`` pairs for ``_restore_handlers``; a signal
    that cannot be handled here (not the main thread) is left alone.
    """
    installed: List[Tuple[int, Any]] = []
    for name in STOP_SIGNALS:
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            installed.append((signum, signal.signal(signum, _stop_on_signal)))
        except (ValueError, OSError, RuntimeError) as exc:
            LOGGER.warning("cannot handle %s (%s); it will kill the monitor without a RUN_END", name, exc)
    return installed


def _restore_handlers(installed: Sequence[Tuple[int, Any]]) -> None:
    """Put back the handlers ``_install_stop_handlers`` replaced."""
    for signum, previous in installed:
        try:
            signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
        except (ValueError, OSError, RuntimeError):
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point (``gradecrossing-monitor`` / ``python -m gradecrossing.monitor``).

    Exit status (also in ``--help``), so a supervisor (systemd ``Restart=on-failure``) or a
    CI smoke step can tell whether the crossing was watched:

    * 0 (``EXIT_OK``): the input ended normally (end of file, ``--max-frames``) or the run
      was stopped (Ctrl-C, q/Esc in the display window, SIGTERM or SIGHUP - systemctl stop,
      docker stop - which stop it like Ctrl-C: RUN_END, sinks flushed).
    * 1 (``EXIT_NOT_WATCHING``): the crossing stopped being watched, or watching never
      started - the source cannot be opened, the detector cannot be loaded (missing
      weights, no ultralytics), an output path cannot be created (snapshot directory,
      video directory, events log), ``MAX_CONSECUTIVE_FRAME_ERRORS`` frames failed in a
      row, or a live source could not be reconnected. The summary line says ``ABORTED``
      with the stop reason; RUN_END and HEALTH events tell the sinks.
    * 2 (``EXIT_CONFIG``): configuration or usage error - the config file cannot be loaded
      (missing file, YAML syntax error such as a tab in the indentation, a top level that
      is not a mapping, an unknown key), a command-line override cannot be applied,
      ``Monitor(cfg)`` rejects a setting (``ValueError``/``TypeError``/``KeyError``/
      ``OSError`` at construction: a bad zone file, detector scores outside (0, 1]), or the
      zone calibration cannot be mapped onto the stream (``ZoneGeometryError``: another
      aspect ratio than ``zone.reference_size``, also after a live stream renegotiated
      it). From the operator's point of view each is one mistake with one fix, so it is
      one line on stderr, never a traceback.
    """
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 - see the exit-status note above
        print(f"error: cannot load config {args.config!r}: {one_line(exc)}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        apply_cli_overrides(cfg, args)
    except Exception as exc:  # noqa: BLE001
        print(f"error: invalid command-line override: {one_line(exc)}", file=sys.stderr)
        return EXIT_CONFIG
    level = getattr(logging, str(cfg.output.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        # The JSONL sink opens its file when the Monitor is built: check the path first so
        # an unusable events log is an output error (1), not a config error.
        _prepare_output_paths(events_jsonl=cfg.alerts.events_jsonl)
    except RuntimeError as exc:
        print(f"error: {one_line(exc)}", file=sys.stderr)
        return EXIT_NOT_WATCHING
    except (TypeError, ValueError) as exc:  # not a path at all (a number, a list)
        print(f"error: invalid configuration: alerts.events_jsonl: {one_line(exc)}", file=sys.stderr)
        return EXIT_CONFIG
    try:
        monitor = Monitor(cfg)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"error: invalid configuration: {one_line(exc)}", file=sys.stderr)
        return EXIT_CONFIG
    stop_handlers = _install_stop_handlers()  # SIGTERM / SIGHUP stop the run like Ctrl-C
    try:
        summary = monitor.run()
    except ZoneGeometryError as exc:  # the config's zone does not fit this stream: recalibrate
        if monitor.last_summary is not None:
            print(format_summary(monitor.last_summary), flush=True)
        print(f"error: zone refused: {one_line(exc)}", file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:  # Ctrl-C / SIGTERM before the loop (e.g. while the source opens)
        print("interrupted", file=sys.stderr)
        return EXIT_OK
    except (RuntimeError, OSError, ImportError) as exc:  # source, model, output paths
        print(f"error: {one_line(exc)}", file=sys.stderr)
        return EXIT_NOT_WATCHING
    except Exception as exc:  # noqa: BLE001 - unexpected: keep the traceback in the log
        LOGGER.exception("the monitor stopped on an unexpected error")
        print(f"error: {type(exc).__name__}: {one_line(exc)}", file=sys.stderr)
        return EXIT_NOT_WATCHING
    finally:
        _restore_handlers(stop_handlers)
    print(format_summary(summary), flush=True)
    return EXIT_NOT_WATCHING if summary.get("aborted") else EXIT_OK


__all__ = [
    "CLI_OVERRIDES",
    "CliOverride",
    "ConfigClassifier",
    "FrameGeometryError",
    "Monitor",
    "add_cli_arguments",
    "apply_cli_overrides",
    "build_parser",
    "config_sha256",
    "format_summary",
    "is_live_source",
    "main",
    "one_line",
    "open_source",
]


if __name__ == "__main__":
    raise SystemExit(main())
