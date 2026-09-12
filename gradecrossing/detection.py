"""YOLO detectors for the grade-crossing monitor.

Two thin wrappers around ultralytics models live here:

* :class:`VehicleDetector` finds road users (cars, trucks, pedestrians, trains...) and, when a
  tracker config is set, gives every box a persistent ``track_id`` (ByteTrack by default). The
  stuck-vehicle logic in ``tracking.py`` is *per object*, so stable ids are what make the whole
  safety chain work: without them a queue of cars would look like one car that never leaves.
* :class:`InfrastructureDetector` finds the crossing hardware (gate arms, signal lights) and turns
  the detections into a slowly-varying :class:`GateState`. Gates down means a train is imminent,
  so the alert engine shortens its thresholds when this reports ``gates_down=True``.

Design rules honoured here (see ``docs/ARCHITECTURE.md``):

* **This is the only module that imports ultralytics/torch, and only inside functions.** The rest
  of the package, the tests and the calibration tool must import on a laptop or Jetson without
  torch. Both detector classes also accept an already-constructed ``model`` object so the
  conversion and classification logic can be unit-tested with a stub.
* **Weights format is transparent.** ``.pt`` (PyTorch), ``.engine`` (TensorRT, see
  ``tools/export_tensorrt.py``) and ``.onnx`` are all loaded by ultralytics; class names come from
  the checkpoint / engine metadata so nothing here depends on the file type.
* **No silent downloads in the field.** A missing local weights path raises immediately instead of
  letting ultralytics attempt a GitHub download from a camera box without internet.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import DetectorConfig, InfrastructureConfig
from .types import Detection, GateState

LOGGER = logging.getLogger(__name__)

# Categories returned by VehicleDetector.classify. Kept as constants so the tracker and the
# alert engine compare against the same strings.
CATEGORY_VEHICLE = "vehicle"
CATEGORY_PERSON = "person"
CATEGORY_TRAIN = "train"
CATEGORY_OTHER = "other"

# Bare file names that ultralytics is allowed to download from its release assets when they are
# not present locally (e.g. the default ``yolo11n.pt``). Anything else that is missing is an error.
_OFFICIAL_ASSET_RE = re.compile(r"^(yolo|rtdetr|sam|fastsam|mobile_sam)[a-z0-9_\-.]*\.pt$", re.IGNORECASE)


def _normalise_name(name: Any) -> str:
    """Lower-case, whitespace-stripped class name used for all case-insensitive comparisons."""
    return str(name).strip().lower()


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    """Convert a torch tensor (any device/dtype), numpy array or list to a numpy array.

    Written without importing torch: tensors are detected by duck typing so the same code path
    serves real ultralytics results and the plain-numpy stubs used in the unit tests.
    """
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def load_yolo(weights: str, task: str = "detect") -> Any:
    """Import ultralytics lazily and load ``weights`` (``.pt``, ``.engine``, ``.onnx``...).

    Why the existence check: ``ultralytics.YOLO("foo.pt")`` tries to *download* ``foo.pt`` from
    GitHub when the file is missing. On a field unit without internet that turns a typo in the
    config into a multi-minute hang with a confusing network error. We only let official
    ultralytics asset names (``yolo11n.pt`` ...) through to the downloader; any other missing path
    fails fast with the path in the message.
    """
    weights_str = str(weights)
    path = Path(weights_str)
    if not path.exists():
        bare_name = os.path.dirname(weights_str) == "" and _OFFICIAL_ASSET_RE.match(weights_str) is not None
        if not bare_name:
            raise FileNotFoundError(
                f"weights file not found: '{weights_str}' (cwd={os.getcwd()}). "
                "Only official ultralytics asset names such as 'yolo11n.pt' are auto-downloaded."
            )
        LOGGER.warning("weights '%s' not found locally; ultralytics will try to download them", weights_str)
    try:
        from ultralytics import YOLO  # local import: keep torch out of the package import graph
    except ImportError as exc:  # pragma: no cover - exercised only on machines without ultralytics
        raise ImportError(
            "ultralytics is required for the YOLO detectors (pip install ultralytics plus a torch "
            "build for your platform). Every other gradecrossing module works without it."
        ) from exc
    LOGGER.info("loading YOLO weights %s (task=%s)", weights_str, task)
    return YOLO(weights_str, task=task)


def results_to_detections(results: Any, names: Dict[int, str]) -> List[Detection]:
    """Convert the first ``ultralytics`` result of a predict/track call into :class:`Detection` objects.

    ``xyxy`` becomes a float32 (4,) numpy array in pixel coordinates of the *input* frame (ultralytics
    already rescales from the letterboxed inference size), ``track_id`` is taken from ``boxes.id``
    when a tracker produced ids and is ``None`` otherwise.
    """
    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return []
    xyxy = _to_numpy(getattr(boxes, "xyxy", None))
    if xyxy is None or xyxy.size == 0:
        return []
    xyxy = xyxy.astype(np.float32).reshape(-1, 4)
    conf = _to_numpy(boxes.conf).astype(np.float64).reshape(-1)
    cls = _to_numpy(boxes.cls).astype(np.int64).reshape(-1)
    ids_raw = getattr(boxes, "id", None)
    ids = _to_numpy(ids_raw).reshape(-1) if ids_raw is not None else None
    detections: List[Detection] = []
    for i in range(xyxy.shape[0]):
        cls_id = int(cls[i])
        track_id = int(ids[i]) if ids is not None and i < ids.shape[0] else None
        detections.append(
            Detection(
                xyxy=xyxy[i].copy(),
                conf=float(conf[i]),
                cls_id=cls_id,
                cls_name=names.get(cls_id, str(cls_id)),
                track_id=track_id,
            )
        )
    return detections


class _YoloBase:
    """Plumbing shared by both detectors: lazy model load, class names, one inference call.

    ``model`` may be injected (anything with ``.names`` and ``.predict``/``.track``) so the
    higher-level logic is testable without loading weights.
    """

    def __init__(self, weights: str, device: str, half: bool, model: Optional[Any] = None) -> None:
        self._model = model if model is not None else load_yolo(weights)
        # ultralytics semantics: "" is not a valid device string, None means "auto".
        self._device: Optional[str] = device or None
        self._half = bool(half)
        self._names = self._read_names(self._model)

    @staticmethod
    def _read_names(model: Any) -> Dict[int, str]:
        raw = getattr(model, "names", None)
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {int(k): str(v) for k, v in raw.items()}
        return {int(i): str(v) for i, v in enumerate(raw)}

    @property
    def names(self) -> Dict[int, str]:
        """Class id -> class name as stored in the checkpoint / engine metadata (copy)."""
        return dict(self._names)

    def _infer(
        self,
        frame_bgr: np.ndarray,
        tracker: str,
        conf: float,
        imgsz: int,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
    ) -> Any:
        """Run one predict/track call and return the raw ultralytics results list.

        The keyword set must be *identical on every call*: ultralytics re-derives the predictor
        arguments per call and rebuilds the predictor (and with it the trackers, losing all track
        state) when the effective precision changes. ``half`` is therefore either passed on every
        call (``cfg.half`` True) or never (False, the ultralytics default). Passing ``half=False``
        explicitly would be equivalent but makes ultralytics >= 8.4 print a deprecation warning on
        every frame (it maps ``half`` to ``quantize=16`` internally; 8.3 releases still need the
        ``half`` spelling, which is why we do not switch to ``quantize``).
        """
        kwargs: Dict[str, Any] = {
            "conf": float(conf),
            "imgsz": int(imgsz),
            "device": self._device,
            "verbose": False,
        }
        if iou is not None:
            kwargs["iou"] = float(iou)
        if max_det is not None:
            kwargs["max_det"] = int(max_det)
        if self._half:
            kwargs["half"] = True
        if tracker:
            # persist=True keeps the tracker alive between calls; without it every frame would be
            # treated as a new video and ids would restart at 1 each time.
            return self._model.track(frame_bgr, persist=True, tracker=tracker, **kwargs)
        return self._model.predict(frame_bgr, **kwargs)


class VehicleDetector(_YoloBase):
    """Road-user detector with optional multi-object tracking.

    ``detect`` returns one :class:`Detection` per box with a ``track_id`` when ``cfg.tracker`` is a
    tracker YAML (``bytetrack.yaml``, ``botsort.yaml``); with ``cfg.tracker == ""`` it runs plain
    detection and ``tracking.py`` falls back to its own IoU matcher. ``classify`` maps a class name
    to ``"vehicle" | "person" | "train" | "other"`` using the case-insensitive lists in the config,
    which is how COCO names (``car``) and the fine-tuned CCTV names (``Car``, ``Jeep``) are treated
    alike.
    """

    def __init__(self, cfg: DetectorConfig, model: Optional[Any] = None) -> None:
        self.cfg = cfg
        super().__init__(cfg.weights, cfg.device, cfg.half, model=model)
        self._vehicle = {_normalise_name(n) for n in cfg.vehicle_classes}
        self._person = {_normalise_name(n) for n in cfg.person_classes}
        self._train = {_normalise_name(n) for n in cfg.train_classes}
        if cfg.detect_every_n_frames > 1 and not cfg.tracker:
            LOGGER.warning(
                "detect_every_n_frames=%d without a tracker: skipped frames rely on the IoU fallback",
                cfg.detect_every_n_frames,
            )
        unknown = [n for n in self._names.values() if self.classify(n) == CATEGORY_OTHER]
        if unknown:
            # Deliberately a warning, once, at start-up: an unmapped class is a blind spot,
            # not a detail. Such a box never feeds the stuck/pedestrian logic *and* is still
            # subtracted from the debris mask, so a stationary obstruction the model labels
            # with one of these names raises nothing at all. It does block the empty-zone
            # background refresh (tracking.StuckVehicleTracker.zone_occupied), which is the
            # only protection left. Map anything that can obstruct a crossing into
            # detector.vehicle_classes / person_classes.
            LOGGER.warning(
                "model classes not mapped to vehicle/person/train: %s - objects of these "
                "classes raise no WARNING/ALARM and are excluded from the debris mask; add "
                "the ones that can obstruct the crossing to detector.vehicle_classes or "
                "detector.person_classes", unknown)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detect (and track) objects in one BGR frame.

        Frame skipping is the Monitor's job (``detect_every_n_frames``); this method always runs
        the model. Track ids are only meaningful when consecutive frames of the same video are
        fed in order; call :meth:`reset` when switching videos.
        """
        results = self._infer(
            frame_bgr,
            tracker=self.cfg.tracker,
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
        )
        return results_to_detections(results, self._names)

    def reset(self) -> None:
        """Forget all tracks (new video / after a reconnect).

        ultralytics keeps one tracker per stream on its predictor; ``BYTETracker.reset`` clears the
        track lists and the global id counter so ids restart at 1. Before the first ``detect`` call
        there is no predictor yet and this is a no-op.
        """
        predictor = getattr(self._model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) or []
        for tracker in trackers:
            if hasattr(tracker, "reset"):
                tracker.reset()
        if trackers:
            LOGGER.info("vehicle tracker state reset (%d tracker(s))", len(trackers))

    def classify(self, cls_name: str) -> str:
        """Map a detector class name to ``"vehicle" | "person" | "train" | "other"`` (case-insensitive).

        Trains are deliberately their own category: a train in the zone is normal and must never
        feed the stuck-vehicle timers, yet its box still has to be excluded from the debris mask.
        """
        key = _normalise_name(cls_name)
        if key in self._vehicle:
            return CATEGORY_VEHICLE
        if key in self._person:
            return CATEGORY_PERSON
        if key in self._train:
            return CATEGORY_TRAIN
        return CATEGORY_OTHER


class InfrastructureDetector(_YoloBase):
    """Gate-arm / signal-light detector producing a debounced :class:`GateState`.

    The model runs only every ``cfg.every_n_frames`` frames (gate state changes over seconds, not
    frames) and the last state is returned in between. ``gates_down`` is held ``True`` for
    ``cfg.hold_seconds`` after the last gate-down detection so one missed frame (occlusion by a
    passing truck, motion blur) does not flip the alert thresholds back and forth. This is a
    safety-conservative choice: while the hold is active the alert engine keeps using the shorter
    gates-down thresholds.
    """

    def __init__(self, cfg: InfrastructureConfig, model: Optional[Any] = None) -> None:
        self.cfg = cfg
        super().__init__(cfg.weights, cfg.device, cfg.half, model=model)
        self._gate_down = {_normalise_name(n) for n in cfg.gate_down_classes}
        self._light_on = {_normalise_name(n) for n in cfg.light_on_classes}
        self._every = max(1, int(cfg.every_n_frames))
        self._last_seen_down: Optional[float] = None
        self._state = GateState(available=True)
        self.last_detections: List[Detection] = []  # boxes from the last model run (any class)
        missing = [n for n in (self._gate_down | self._light_on) if n not in {_normalise_name(v) for v in self._names.values()}]
        if missing and self._names:
            LOGGER.warning("infrastructure model has no class named %s (model classes: %s)", missing, list(self._names.values()))

    @property
    def state(self) -> GateState:
        """The most recent :class:`GateState` (what ``update`` returns on cached frames)."""
        return self._state

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run the infrastructure model once and return every box (gates, signs, lights).

        Exposed separately from :meth:`update` so ``autozone.propose_zone`` and the calibration
        tool can use gate/sign boxes to bound the crossing.
        """
        results = self._infer(frame_bgr, tracker="", conf=self.cfg.conf, imgsz=self.cfg.imgsz)
        return results_to_detections(results, self._names)

    def update(self, frame_bgr: np.ndarray, timestamp: float, frame_index: int) -> GateState:
        """Refresh the gate state on frame 0 and every ``every_n_frames``; otherwise return the cache.

        The model also runs on the very first call regardless of ``frame_index`` (e.g. after a
        ``start_seconds`` seek) so the monitor never operates on the empty initial state.
        """
        due = (frame_index % self._every == 0) or self._state.last_update is None
        if not due:
            return self._state
        detections = self.detect(frame_bgr)
        self.last_detections = detections
        down_conf = 0.0
        lights_on = False
        for det in detections:
            key = _normalise_name(det.cls_name)
            if key in self._gate_down:
                down_conf = max(down_conf, float(det.conf))
            if key in self._light_on:
                lights_on = True
        if down_conf > 0.0:
            self._last_seen_down = float(timestamp)
        gates_down = self._last_seen_down is not None and (timestamp - self._last_seen_down) <= self.cfg.hold_seconds
        self._state = GateState(
            gates_down=bool(gates_down),
            lights_on=lights_on,
            confidence=down_conf,
            last_update=float(timestamp),
            last_seen_down=self._last_seen_down,
            available=True,
        )
        return self._state

    def reset(self) -> None:
        """Forget the hold timer and cached state (new video / timestamps restarting at 0)."""
        self._last_seen_down = None
        self._state = GateState(available=True)
        self.last_detections = []


__all__ = [
    "CATEGORY_OTHER",
    "CATEGORY_PERSON",
    "CATEGORY_TRAIN",
    "CATEGORY_VEHICLE",
    "InfrastructureDetector",
    "VehicleDetector",
    "load_yolo",
    "results_to_detections",
]
