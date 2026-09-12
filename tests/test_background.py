"""Tests for gradecrossing.background.DebrisDetector on synthetic 320x240 frames.

The scene is a static noisy texture (fixed per seed) plus mild per-frame sensor noise.
"Debris" is a bright rectangle painted inside the zone. Every test seeds numpy, so the
suite is deterministic, and no test needs more than a few hundred MOG2 frames.

The two tests to read first are ``test_frozen_background_never_absorbs_static_object``
(the safety property the module exists for) and
``test_auto_learning_rate_absorbs_static_object`` (the prototype's failure mode).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest

from gradecrossing.background import AUTO_LEARNING_RATE, DebrisDetector
from gradecrossing.config import DebrisConfig
from gradecrossing.geometry import polygon_mask

H, W = 240, 320
FPS = 30.0
# Quadrilateral zone; shoelace area = 23000 px.
ZONE = np.array([[40, 120], [280, 110], [300, 200], [30, 210]], dtype=np.float32)
ZONE_MASK = polygon_mask((H, W), ZONE)
ZONE_AREA = int(np.count_nonzero(ZONE_MASK))
# 60x40 = 2400 px rectangle fully inside the zone (10.4 % of the zone area).
RECT = (150, 140, 210, 180)
RECT_AREA = (RECT[2] - RECT[0]) * (RECT[3] - RECT[1])
# Same size rectangle entirely outside the zone (above its top edge).
RECT_OUTSIDE = (20, 20, 80, 60)
WHITE = (255.0, 255.0, 255.0)


def make_cfg(**overrides) -> DebrisConfig:
    """Defaults for tests: short warm-up, refresh disabled unless a test enables it."""
    base = dict(warmup_frames=60, empty_seconds_before_refresh=0.0)
    base.update(overrides)
    return DebrisConfig(**base)


class Scene:
    """Static noisy background + optional painted objects + per-frame sensor noise."""

    NOISE_BANK = 8  # per-frame sensor noise cycles through a small seeded bank (speed)

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.base = np.clip(self.rng.normal(120.0, 12.0, size=(H, W, 3)), 0, 255).astype(np.float32)
        self.noise = self.rng.normal(0.0, 3.0, size=(self.NOISE_BANK, H, W, 3)).astype(np.float32)
        self.calls = 0

    def frame(self, rect: Optional[Tuple[int, int, int, int]] = None,
              color: Sequence[float] = WHITE, brightness: float = 0.0,
              shade: Optional[float] = None) -> np.ndarray:
        img = self.base.copy()
        if rect is not None:
            x1, y1, x2, y2 = rect
            if shade is not None:
                # Darker copy of the background with the same chromaticity (a shadow,
                # or a dark neutral object): what MOG2's shadow detector looks for.
                img[y1:y2, x1:x2] *= shade
            else:
                img[y1:y2, x1:x2] = np.asarray(color, dtype=np.float32)
        img += self.noise[self.calls % self.NOISE_BANK]
        if brightness:
            img += brightness
        self.calls += 1
        return np.clip(img, 0, 255).astype(np.uint8)


class Session:
    """Drives a DebrisDetector through frames with a running timestamp."""

    def __init__(self, cfg: DebrisConfig, seed: int = 0, fps: float = FPS) -> None:
        self.det = DebrisDetector(cfg)
        self.scene = Scene(seed)
        self.fps = fps
        self.i = 0

    def step(self, n: int, rect=None, boxes: Sequence[np.ndarray] = (), empty: Optional[float] = None,
             **frame_kw) -> List:
        out = []
        for _ in range(n):
            t = self.i / self.fps
            frame = self.scene.frame(rect=rect, **frame_kw)
            out.append(self.det.update(frame, ZONE_MASK, list(boxes), empty, t))
            self.i += 1
        return out

    def warm_up(self) -> None:
        self.step(self.det.cfg.warmup_frames)
        assert self.det.warmed_up


def first_index(results, predicate) -> Optional[int]:
    for i, r in enumerate(results):
        if predicate(r):
            return i
    return None


# --------------------------------------------------------------------------- warm-up


def test_warmup_reports_nothing_then_flips():
    np.random.seed(0)
    s = Session(make_cfg(warmup_frames=20))
    # Debris present from frame 0 is learned as background: warm-up must be on a clear crossing.
    res = s.step(20, rect=RECT)
    assert all(not r.warmed_up for r in res)
    assert all(not r.triggered and r.area_px == 0 and r.area_fraction == 0.0 for r in res)
    assert all(r.learning_rate_used == AUTO_LEARNING_RATE for r in res)
    assert all(r.mask.shape == (H, W) and r.mask.dtype == np.uint8 and not r.mask.any() for r in res)
    assert s.det.warmed_up and s.det.frames_seen == 20
    nxt = s.step(1, rect=RECT)[0]
    assert nxt.warmed_up
    assert nxt.area_px == 0  # the rectangle was in every warm-up frame, so it *is* the background


# ------------------------------------------------------------------- safety property


def test_frozen_background_never_absorbs_static_object():
    """Spec test 1: with monitoring_learning_rate=0 a static object stays for 20 s and triggers."""
    np.random.seed(0)
    cfg = make_cfg(monitoring_learning_rate=0.0, persist_seconds=3.0)
    s = Session(cfg)
    s.warm_up()
    bg_before = s.det.background_image().copy()

    n = int(20 * FPS)  # 20 s of frames
    res = s.step(n, rect=RECT)
    areas = [r.area_px for r in res]
    assert areas[0] >= 0.9 * RECT_AREA, "object must be visible on its first frame"
    assert min(areas) >= 0.9 * areas[0], f"object faded: min {min(areas)} vs initial {areas[0]}"
    assert all(r.learning_rate_used == 0.0 for r in res)
    assert all(r.warmed_up for r in res)

    # Timer: first over-threshold frame is t0; triggered once t - t0 >= persist_seconds.
    k = int(round(cfg.persist_seconds * FPS))  # 90 frames
    assert not res[k - 1].triggered and res[k - 1].persist_seconds == pytest.approx((k - 1) / FPS)
    assert res[k].triggered and res[k].persist_seconds == pytest.approx(k / FPS)
    assert all(r.triggered for r in res[k:])
    assert res[-1].persist_seconds == pytest.approx((n - 1) / FPS)
    assert s.det.triggered

    # Strongest proof of "frozen": the model itself is bit-identical after 600 frames.
    assert np.array_equal(bg_before, s.det.background_image())

    # Blob bookkeeping: one component with the rectangle's box.
    x, y, w, h, area = res[-1].blobs[0]
    assert len(res[-1].blobs) == 1
    assert abs(x - RECT[0]) <= 2 and abs(y - RECT[1]) <= 2
    assert abs(w - (RECT[2] - RECT[0])) <= 3 and abs(h - (RECT[3] - RECT[1])) <= 3
    assert area == res[-1].area_px
    assert res[-1].area_fraction == pytest.approx(res[-1].area_px / ZONE_AREA)
    assert int(np.count_nonzero(res[-1].mask)) == res[-1].area_px


def test_auto_learning_rate_absorbs_static_object():
    """Spec test 2 (documented failure mode of the prototype): OpenCV's automatic rate
    with history=30 absorbs the same static object within a few hundred frames.
    ``scripts/pipeline_helpers.apply_mog2_mask`` did exactly this on every frame."""
    np.random.seed(0)
    cfg = make_cfg(monitoring_learning_rate=-1.0, history=30, persist_seconds=3.0)
    s = Session(cfg)
    s.warm_up()
    bg_before = s.det.background_image().copy()

    res = s.step(300, rect=RECT)
    initial = res[0].area_px
    assert initial >= 0.9 * RECT_AREA
    absorbed_at = first_index(res, lambda r: r.area_px < 0.5 * initial)
    assert absorbed_at is not None, "automatic rate should absorb a static object"
    assert absorbed_at < 300
    # The analytic estimate (~3 frames at rate 1/30) is a tight upper bound on this.
    predicted = DebrisDetector.absorption_frames(1.0 / cfg.history)
    assert absorbed_at <= math.ceil(predicted) + 5
    # ... and the object vanished long before the persistence timer could trigger.
    assert not any(r.triggered for r in res)
    assert not np.array_equal(bg_before, s.det.background_image())


def test_absorption_frames_formula():
    assert DebrisDetector.absorption_frames(0.0) == math.inf
    assert DebrisDetector.absorption_frames(-1.0) == math.inf
    assert DebrisDetector.absorption_frames(1.0) == 1.0
    assert DebrisDetector.absorption_frames(1.0 / 30) == pytest.approx(3.11, abs=0.05)
    assert DebrisDetector.absorption_frames(0.005) == pytest.approx(21.0, abs=0.1)
    assert DebrisDetector.absorption_frames(0.0005) == pytest.approx(210.7, abs=0.5)


# ----------------------------------------------------------------- masking rules


def test_exclusion_box_removes_object():
    """Spec test 3: a detection box covering the rectangle yields area_px == 0."""
    np.random.seed(0)
    s = Session(make_cfg())
    s.warm_up()
    box = np.array(RECT, dtype=np.float32)
    res = s.step(5, rect=RECT, boxes=[box])
    assert all(r.area_px == 0 and r.area_fraction == 0.0 and r.blobs == [] for r in res)
    assert all(not r.mask.any() for r in res)
    assert all(r.persist_seconds == 0.0 and not r.triggered for r in res)
    # The exclusion is local: a box elsewhere does not hide the object.
    far_box = np.array([40, 120, 100, 160], dtype=np.float32)
    res = s.step(3, rect=RECT, boxes=[far_box])
    assert all(r.area_px >= 0.9 * RECT_AREA for r in res)


def test_exclusion_box_is_dilated():
    """A box that under-covers the object by less than exclusion_dilate_px still hides it
    (dilation absorbs tracker jitter and MOG2's halo around moving objects)."""
    np.random.seed(0)
    s = Session(make_cfg(exclusion_dilate_px=12))
    s.warm_up()
    shrunk = np.array([RECT[0] + 8, RECT[1] + 8, RECT[2] - 8, RECT[3] - 8], dtype=np.float32)
    res = s.step(3, rect=RECT, boxes=[shrunk])
    assert all(r.area_px == 0 for r in res)
    # Without dilation the uncovered rim shows up (if large enough to be a blob).
    s2 = Session(make_cfg(exclusion_dilate_px=0, min_blob_area_fraction=0.0))
    s2.warm_up()
    res2 = s2.step(3, rect=RECT, boxes=[shrunk])
    assert all(r.area_px > 0 for r in res2)


def test_object_outside_zone_is_ignored():
    """Spec test 4: a rectangle outside the zone produces area_px == 0."""
    np.random.seed(0)
    s = Session(make_cfg())
    s.warm_up()
    res = s.step(5, rect=RECT_OUTSIDE)
    assert all(r.area_px == 0 and r.blobs == [] and not r.mask.any() for r in res)
    assert all(not r.triggered for r in res)


def test_small_blobs_dropped_by_zone_fraction():
    """Components below min_blob_area_fraction * zone_area are speckle, not debris."""
    np.random.seed(0)
    tiny = (170, 155, 176, 161)  # 36 px, below 0.002 * 23000 = 46 px
    s = Session(make_cfg(min_blob_area_fraction=0.002))
    s.warm_up()
    res = s.step(3, rect=tiny)
    assert all(r.area_px == 0 and r.blobs == [] for r in res)
    s2 = Session(make_cfg(min_blob_area_fraction=0.0))
    s2.warm_up()
    res2 = s2.step(3, rect=tiny)
    assert all(r.area_px > 0 and len(r.blobs) == 1 for r in res2)


def test_shadow_pixels_are_dropped():
    """Documented limitation: a darker same-chromaticity patch is a 'shadow' to MOG2
    and is dropped when detect_shadows=True; detect_shadows=False reports it."""
    np.random.seed(0)
    s = Session(make_cfg(detect_shadows=True))
    s.warm_up()
    res = s.step(3, rect=RECT, shade=0.6)
    assert all(r.area_px == 0 for r in res)
    s2 = Session(make_cfg(detect_shadows=False))
    s2.warm_up()
    res2 = s2.step(3, rect=RECT, shade=0.6)
    assert all(r.area_px >= 0.9 * RECT_AREA for r in res2)


# ------------------------------------------------------------------- refresh path


def test_refresh_absorbs_sub_threshold_change():
    """Spec test 5 (refresh path): with the zone reported empty for long enough, a change
    that stays *below* min_area_fraction is learned at empty_learning_rate and disappears
    within absorption_frames frames. This is all the refresh may still do since the
    persistence guard was added: it nibbles drift away while it is small, so gradual
    lighting change never accumulates into a candidate."""
    np.random.seed(0)
    cfg = make_cfg(empty_seconds_before_refresh=5.0, empty_learning_rate=0.005)
    small = (150, 140, 180, 160)  # 30x20 = 600 px = 2.6 % of the zone, under min_area_fraction
    assert (small[2] - small[0]) * (small[3] - small[1]) / ZONE_AREA < cfg.min_area_fraction
    s = Session(cfg)
    s.warm_up()
    res = s.step(150, rect=small, empty=100.0)
    assert all(r.learning_rate_used == pytest.approx(0.005) for r in res)
    assert res[0].area_px > 0, "the change must initially look like foreground"
    assert all(not r.triggered and r.persist_seconds == 0.0 for r in res)
    gone_at = first_index(res, lambda r: r.area_px == 0)
    assert gone_at is not None
    assert gone_at <= math.ceil(DebrisDetector.absorption_frames(0.005)) + 15
    assert all(r.area_px == 0 for r in res[gone_at + 20:])

    # Contrast: the same change with the zone occupancy unknown is never absorbed.
    s2 = Session(cfg)
    s2.warm_up()
    res2 = s2.step(150, rect=small, empty=None)
    assert all(r.learning_rate_used == 0.0 for r in res2)
    assert all(r.area_px >= 0.9 * res2[0].area_px for r in res2)


def test_global_illumination_step_rewarms_instead_of_alarming():
    """A sudden global brightness step on an empty zone covers more than max_area_fraction,
    so it is reported as an illumination event and the model is re-warmed. Without this the
    step would start the persistence timer, the timer would block the refresh that could
    have absorbed it, and the ALARM would latch forever."""
    np.random.seed(0)
    cfg = make_cfg(empty_seconds_before_refresh=5.0, empty_learning_rate=0.005,
                   persist_seconds=1.0, warmup_frames=40)
    s = Session(cfg)
    s.warm_up()
    first = s.step(1, brightness=60.0, empty=100.0)[0]
    assert not first.warmed_up and first.area_px == 0 and not first.triggered
    ev = s.det.last_illumination_event
    assert ev is not None and ev["area_fraction"] >= cfg.max_area_fraction
    assert s.det.frames_seen == 1  # the model restarted its warm-up on this frame

    # Re-warm on the new lighting, then the crossing reads clear again: no alarm, ever.
    rest = s.step(80, brightness=60.0, empty=100.0)
    assert s.det.last_illumination_event is None  # reported exactly once
    assert all(not r.triggered for r in rest)
    assert all(r.area_px == 0 for r in rest[45:])

    # Contrast (and the documented escape hatch): with the guard disabled the same step is
    # treated as debris and alarms after persist_seconds, and is never absorbed.
    s2 = Session(make_cfg(empty_seconds_before_refresh=5.0, empty_learning_rate=0.005,
                          persist_seconds=1.0, warmup_frames=40, max_area_fraction=0.0))
    s2.warm_up()
    res2 = s2.step(90, brightness=60.0, empty=100.0)
    assert s2.det.last_illumination_event is None
    assert res2[-1].triggered and res2[-1].area_fraction > 0.5


def test_illumination_guard_needs_the_rest_of_the_frame_to_change_too():
    """A huge obstruction on a tightly drawn zone must not be mistaken for a lighting step:
    a lighting change lights up the whole image, an object does not. Here a block covers the
    whole zone while the rest of the frame is untouched, so the guard stays out of the way
    and the object alarms as debris."""
    np.random.seed(0)
    cfg = make_cfg(persist_seconds=1.0)
    covering = (30, 105, 305, 215)  # bounding box of the whole zone
    s = Session(cfg)
    s.warm_up()
    res = s.step(int(2 * FPS), rect=covering, empty=100.0)
    assert res[0].area_fraction >= cfg.max_area_fraction, "the object must fill the zone"
    assert s.det.last_illumination_event is None
    assert all(r.warmed_up for r in res)
    assert res[-1].triggered


def test_illumination_guard_needs_the_tracker_to_agree():
    """The guard fires only when the tracker reports the zone empty; while the zone is
    occupied (or unknown) a large foreground is left alone, so a huge obstruction that the
    tracker *can* see is never re-warmed away."""
    np.random.seed(0)
    s = Session(make_cfg(persist_seconds=1.0))
    s.warm_up()
    res = s.step(40, brightness=60.0, empty=None)
    assert s.det.last_illumination_event is None
    assert all(r.warmed_up for r in res)
    assert res[-1].triggered


def test_refresh_requires_threshold_and_enable():
    np.random.seed(0)
    cfg = make_cfg(empty_seconds_before_refresh=5.0, empty_learning_rate=0.005)
    s = Session(cfg)
    s.warm_up()
    assert s.step(1, empty=4.9)[0].learning_rate_used == 0.0
    assert s.step(1, empty=5.0)[0].learning_rate_used == pytest.approx(0.005)
    assert s.step(1, empty=None)[0].learning_rate_used == 0.0
    # empty_seconds_before_refresh == 0 disables refresh entirely.
    s2 = Session(make_cfg(empty_seconds_before_refresh=0.0, empty_learning_rate=0.005))
    s2.warm_up()
    assert s2.step(1, empty=1e9)[0].learning_rate_used == 0.0


def test_refresh_is_suppressed_while_triggered():
    """Safety guard: once debris has triggered, the 'zone empty' refresh must not be
    allowed to erase it (the tracker cannot see debris, so it *will* say empty)."""
    np.random.seed(0)
    cfg = make_cfg(empty_seconds_before_refresh=5.0, empty_learning_rate=0.005, persist_seconds=3.0)
    s = Session(cfg)
    s.warm_up()
    res = s.step(int(4 * FPS), rect=RECT, empty=None)
    assert res[-1].triggered
    initial = res[0].area_px

    held = s.step(200, rect=RECT, empty=100.0)
    assert all(r.learning_rate_used == 0.0 for r in held), "refresh ran while triggered"
    assert all(r.triggered for r in held)
    assert min(r.area_px for r in held) >= 0.9 * initial

    # Once the debris is gone the timer resets and refresh is allowed again.
    cleared = s.step(3, empty=100.0)
    assert cleared[0].area_px == 0 and not cleared[0].triggered and cleared[0].persist_seconds == 0.0
    assert cleared[0].learning_rate_used == 0.0  # rate chosen before this frame was classified
    assert cleared[1].learning_rate_used == pytest.approx(0.005)


def test_refresh_does_not_absorb_untriggered_debris():
    """The hole this closes: debris that appears while the tracker already reports the zone
    empty used to be absorbed at empty_learning_rate in ~21 frames, i.e. before the default
    persist_seconds could elapse, so it never alarmed. The refresh is now suppressed while
    an over-threshold streak is in progress, not only while the alarm is triggered, and the
    tracker cannot see debris - so 'zone empty' is exactly when this matters."""
    np.random.seed(0)
    cfg = make_cfg(empty_seconds_before_refresh=5.0, empty_learning_rate=0.005, persist_seconds=3.0)
    s = Session(cfg)
    s.warm_up()
    res = s.step(int(3 * FPS) + 1, rect=RECT, empty=100.0)
    # The rate for a frame is chosen before that frame is classified, so exactly one frame
    # (the one the object appears on) is still learned at the refresh rate; from the next
    # frame on the streak freezes the model. One frame at 0.005 is ~0.5 % of the weight
    # MOG2 needs, so the object does not fade.
    assert res[0].learning_rate_used == pytest.approx(0.005)
    assert all(r.learning_rate_used == 0.0 for r in res[1:])
    assert min(r.area_px for r in res) >= 0.9 * res[0].area_px
    assert not res[-2].triggered and res[-1].triggered
    assert res[-1].persist_seconds == pytest.approx(3.0)


def test_moving_object_never_triggers_debris():
    """Persistence alone is not evidence: an object the detector missed that takes longer
    than persist_seconds to cross the zone (a train under weights without a 'train' class,
    a crowd) must not raise a debris alarm. The immobility test restarts the streak on
    every frame whose candidate pixels are still changing."""
    np.random.seed(0)
    cfg = make_cfg(persist_seconds=1.0, min_area_fraction=0.02)
    s = Session(cfg)
    s.warm_up()
    # A 60x40 block sliding across the zone at 4 px/frame: 4 s of frames, four times
    # persist_seconds, and it is over the area threshold the whole way.
    res = []
    for k in range(120):
        x = 40 + 2 * k
        res.extend(s.step(1, rect=(x, 140, x + 60, 180)))
    assert all(r.area_fraction >= cfg.min_area_fraction for r in res[2:])
    assert max(r.persist_seconds for r in res) < cfg.persist_seconds
    assert not any(r.triggered for r in res)
    # ... and the moment it stops, the same object does trigger.
    parked = s.step(int(1.5 * FPS), rect=(280, 140, 340, 180))
    assert parked[-1].triggered


# --------------------------------------------------------------- persistence timer


def test_persistence_timer_is_timestamp_based():
    """The same number of frames at 10 fps and 30 fps trigger at the same *seconds*."""
    np.random.seed(0)
    cfg = make_cfg(persist_seconds=3.0)
    fast = Session(cfg, fps=30.0)
    slow = Session(cfg, fps=10.0)
    fast.warm_up()
    slow.warm_up()
    rf = fast.step(91, rect=RECT)
    rs = slow.step(31, rect=RECT)
    assert first_index(rf, lambda r: r.triggered) == 90
    assert first_index(rs, lambda r: r.triggered) == 30
    assert rf[-1].persist_seconds == pytest.approx(3.0)
    assert rs[-1].persist_seconds == pytest.approx(3.0)


def test_persistence_timer_resets_when_debris_leaves():
    np.random.seed(0)
    s = Session(make_cfg(persist_seconds=1.0))
    s.warm_up()
    res = s.step(int(2 * FPS), rect=RECT)
    assert res[-1].triggered and res[-1].persist_seconds == pytest.approx((2 * FPS - 1) / FPS)
    gone = s.step(5)
    assert all(r.area_px == 0 and r.persist_seconds == 0.0 and not r.triggered for r in gone)
    assert not s.det.triggered
    back = s.step(int(FPS) + 1, rect=RECT)
    assert back[0].persist_seconds == 0.0 and not back[0].triggered
    assert not back[-2].triggered and back[-1].triggered


def test_timer_uses_min_area_fraction():
    """Debris smaller than min_area_fraction of the zone is reported but never times."""
    np.random.seed(0)
    s = Session(make_cfg(min_area_fraction=0.2, persist_seconds=0.5))  # rect is 10.4 %
    s.warm_up()
    res = s.step(int(FPS), rect=RECT)
    assert all(r.area_px >= 0.9 * RECT_AREA for r in res)
    assert all(r.persist_seconds == 0.0 and not r.triggered for r in res)


def test_timer_restarts_if_time_goes_backwards():
    np.random.seed(0)
    s = Session(make_cfg(persist_seconds=1.0))
    s.warm_up()
    s.step(int(2 * FPS), rect=RECT)
    assert s.det.triggered
    # A looped file without reset(): timestamps jump back; the streak must restart.
    r = s.det.update(s.scene.frame(rect=RECT), ZONE_MASK, [], None, 0.0)
    assert r.persist_seconds == 0.0 and not r.triggered
    r = s.det.update(s.scene.frame(rect=RECT), ZONE_MASK, [], None, 0.5)
    assert r.persist_seconds == pytest.approx(0.5)


# ---------------------------------------------------------------------- plumbing


def test_reset_clears_model_and_timers():
    np.random.seed(0)
    s = Session(make_cfg(persist_seconds=0.5))
    s.warm_up()
    s.step(int(FPS), rect=RECT)
    assert s.det.triggered and s.det.frames_seen > 0
    s.det.reset()
    assert not s.det.warmed_up and s.det.frames_seen == 0 and not s.det.triggered
    assert s.det.background_image() is None
    r = s.step(1, rect=RECT)[0]
    assert not r.warmed_up and r.area_px == 0


def test_shape_mismatch_raises():
    det = DebrisDetector(make_cfg(warmup_frames=0))
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        det.update(frame, np.zeros((H, W + 1), dtype=np.uint8), [], None, 0.0)
    with pytest.raises(ValueError):
        det.update(frame, np.zeros((H, W, 3), dtype=np.uint8), [], None, 0.0)
    with pytest.raises(ValueError):
        det.update(np.zeros((H,), dtype=np.uint8), ZONE_MASK, [], None, 0.0)


def test_accepts_bool_zone_mask_and_grey_frames():
    np.random.seed(0)
    s = Session(make_cfg())
    s.warm_up()
    frame = s.scene.frame(rect=RECT)
    r = s.det.update(frame, ZONE_MASK > 0, [], None, 100.0)
    assert r.area_px >= 0.9 * RECT_AREA
    # Grey frames are accepted too (a separate model, since MOG2 fixes channels on first apply).
    g = Session(make_cfg())
    for _ in range(g.det.cfg.warmup_frames):
        g.det.update(g.scene.frame()[:, :, 1], ZONE_MASK, [], None, 0.0)
    rg = g.det.update(g.scene.frame(rect=RECT)[:, :, 1], ZONE_MASK, [], None, 1.0)
    assert rg.area_px >= 0.9 * RECT_AREA


def test_no_false_debris_on_static_scene():
    """Sensor noise alone never produces debris on a frozen model (the varThreshold
    default is well above the noise level); an idle camera stays CLEAR."""
    np.random.seed(0)
    s = Session(make_cfg())
    s.warm_up()
    res = s.step(int(5 * FPS))
    assert all(r.area_px == 0 and not r.triggered for r in res)
