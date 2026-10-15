"""Tests for gradecrossing.background.DebrisDetector on synthetic 320x240 frames.

The scene is a static noisy texture (fixed per seed) plus mild per-frame sensor noise.
"Debris" is a bright rectangle painted inside the zone. Every test seeds numpy, so the
suite is deterministic, and no test needs more than a few thousand small MOG2 frames.

The tests to read first are ``test_frozen_background_never_absorbs_static_object`` (the
safety property the module exists for), ``test_auto_learning_rate_absorbs_static_object``
(the prototype's failure mode) and the selective-update / warm-up tests, which pin how
the model is allowed to learn. Lighting, jitter, registration, hold and ghost scenes live
in ``tests/test_debris_robustness.py``; the monitor wiring in ``tests/test_debris_monitor.py``.
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
    """Defaults for tests: a 60-frame warm-up and the selective update disabled, so the
    model is bit-for-bit frozen unless a test enables the update."""
    base = dict(warmup_frames=60, update_time_constant_seconds=0.0)
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
              shade: Optional[float] = None, gain: float = 1.0) -> np.ndarray:
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
        if gain != 1.0:
            img *= gain
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

    @property
    def t(self) -> float:
        return self.i / self.fps

    def step(self, n: int, rect=None, boxes: Sequence[np.ndarray] = (), empty: Optional[float] = None,
             **frame_kw) -> List:
        out = []
        upd = {k: frame_kw.pop(k) for k in ("hold", "registration_ok", "to_reference", "ignore_mask")
               if k in frame_kw}
        for _ in range(n):
            t = self.i / self.fps
            frame = self.scene.frame(rect=rect, **frame_kw)
            out.append(self.det.update(frame, ZONE_MASK, list(boxes), empty, t, **upd))
            self.i += 1
        return out

    def warm_up(self) -> None:
        n = 0
        while not self.det.warmed_up:
            self.step(1)
            n += 1
            assert n < 10000
        assert self.det.warmed_up


def first_index(results, predicate) -> Optional[int]:
    for i, r in enumerate(results):
        if predicate(r):
            return i
    return None


# --------------------------------------------------------------------------- warm-up


def test_warmup_reports_nothing_then_flips():
    s = Session(make_cfg(warmup_frames=20))
    # Debris present from frame 0 with no detection box is learned as background: nothing
    # can tell an undetected object present at start-up from the scene.
    res = s.step(20, rect=RECT)
    assert all(not r.warmed_up for r in res)
    assert all(not r.triggered and r.area_px == 0 and r.area_fraction == 0.0 for r in res)
    assert all(r.learning_rate_used == AUTO_LEARNING_RATE for r in res)
    assert all(r.mask.shape == (H, W) and r.mask.dtype == np.uint8 and not r.mask.any() for r in res)
    assert all(not r.trusted and r.unknown_fraction == 1.0 for r in res)
    assert res[3].state == "warmup 4/20f"
    assert s.det.warmed_up and s.det.frames_seen == 20
    nxt = s.step(1, rect=RECT)[0]
    assert nxt.warmed_up and nxt.trusted and nxt.state == "ok"
    assert nxt.area_px == 0  # the rectangle was in every warm-up frame, so it *is* the background


def test_warmup_is_measured_in_seconds_not_frames():
    """warmup_seconds is wall-clock time of the stream: 2 s is 10 frames at 5 fps and 60 at
    30 fps (a frame count would blind a 5 fps Jetson stream six times longer)."""
    for fps, frames in ((5.0, 10), (30.0, 60)):
        s = Session(DebrisConfig(warmup_seconds=2.0, update_time_constant_seconds=0.0), fps=fps)
        res = s.step(frames)  # t = 0 .. 2.0 s - one frame
        assert not any(r.warmed_up for r in res) and not s.det.warmed_up
        assert res[frames // 2].state == f"warmup {(frames // 2) / fps:.1f}/2.0s"
        last = s.step(1)[0]  # t = 2.0 s: the last warm-up frame
        assert not last.warmed_up and s.det.warmed_up, fps
        assert s.step(1)[0].warmed_up
    assert DebrisConfig().warmup_frames is None and DebrisConfig().warmup_seconds == 2.0


def test_warmup_never_learns_under_boxes_and_leaves_no_ghost():
    """A detected car waiting on the crossing during the whole warm-up is never learned: its
    pixels are UNKNOWN (excluded, reported) until they have been seen box-free; then they are
    learned, so the road it uncovers is no ghost (clip05's warm-up pickup latched a debris
    ALARM twice) and a later obstruction on the same spot still alarms."""
    cfg = make_cfg(update_time_constant_seconds=10.0, learn_seconds=1.0, persist_seconds=1.0)
    car = (150, 140, 210, 180)
    box = np.array(car, dtype=np.float32)
    s = Session(cfg)
    res = s.step(60, rect=car, color=(40, 40, 40), boxes=[box])
    assert not res[-1].warmed_up
    # still parked: its (dilated) box area is unknown, not debris
    parked = s.step(30, rect=car, color=(40, 40, 40), boxes=[box])
    assert all(r.area_px == 0 and not r.triggered for r in parked)
    unknown0 = parked[-1].unknown_fraction
    assert 0.10 < unknown0 < 0.30  # the 84x64 dilated box over the 23000 px zone
    # it drives off: the uncovered road must never become debris, and becomes known
    gone = s.step(int(15 * FPS))
    assert not any(r.triggered for r in gone)
    assert max(r.persist_seconds for r in gone) == 0.0
    assert gone[-1].unknown_fraction == 0.0
    # a real obstruction on the same spot now alarms
    later = s.step(int(1.5 * FPS), rect=car)
    assert later[-1].triggered


def test_unknown_pixels_are_learned_only_after_learn_seconds():
    cfg = make_cfg(update_time_constant_seconds=2.0, learn_seconds=2.0)
    box = np.array(RECT, dtype=np.float32)
    s = Session(cfg)
    s.step(60, rect=RECT, color=(40, 40, 40), boxes=[box])
    res = s.step(int(8 * FPS))  # box-free from now on
    unk = [r.unknown_fraction for r in res]
    assert unk[0] > 0.1
    assert unk[int(1.0 * FPS)] == unk[0]  # not before learn_seconds of looking like background
    assert unk[-1] == 0.0
    assert all(not r.triggered and r.area_px == 0 for r in res)


# ------------------------------------------------------------------- safety property


def test_frozen_background_never_absorbs_static_object():
    """Spec test 1: with monitoring_learning_rate=0 a static object stays for 20 s and triggers."""
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


# ------------------------------------------------------------- selective update


def test_selective_update_never_absorbs_in_zone_foreground():
    """The update learns only pixels classified background outside boxes and candidates, so
    even a fast time constant (2 s, i.e. MOG2 would absorb an unprotected change in ~1 s)
    never learns a static obstruction: it stays at full area and triggered for 60 s."""
    cfg = make_cfg(update_time_constant_seconds=2.0, persist_seconds=1.0)
    s = Session(cfg)
    s.warm_up()
    res = s.step(int(60 * FPS), rect=RECT)
    initial = res[0].area_px
    assert initial >= 0.9 * RECT_AREA
    assert min(r.area_px for r in res) >= 0.95 * initial
    k = int(cfg.persist_seconds * FPS)
    assert all(r.triggered for r in res[k:])
    assert any(r.learning_rate_used > 0.0 for r in res)  # the update did run all along


def test_selective_update_never_learns_under_boxes():
    """A car parked under its box for a minute is never learned, however fast the update:
    when it leaves there is no ghost."""
    cfg = make_cfg(update_time_constant_seconds=2.0, persist_seconds=1.0)
    box = np.array(RECT, dtype=np.float32)
    s = Session(cfg)
    s.warm_up()
    s.step(int(60 * FPS), rect=RECT, color=(40, 40, 40), boxes=[box])
    gone = s.step(int(5 * FPS))
    assert all(r.area_px == 0 and not r.triggered for r in gone)


def test_selective_update_follows_drift_on_a_zone_that_is_never_empty():
    """Lighting drift confined to the zone (so the global gain, measured outside it, cannot
    explain it) on flat asphalt (no texture for the relit test to recognise) is followed by
    the update although a box always covers part of the zone. The old empty-zone refresh
    never ran on such a crossing (clip03/06), and the drift became debris; with the update
    disabled it still does."""
    box = np.array([40, 110, 110, 215], dtype=np.float32)  # a car always in the zone's left part
    in_zone = ZONE_MASK > 0

    def run(tau: float) -> List:
        s = Session(make_cfg(update_time_constant_seconds=tau, persist_seconds=3.0))
        s.scene.base[in_zone] = 110.0  # flat asphalt across the crossing
        s.warm_up()
        out = []
        n = int(60 * FPS)
        for k in range(n):
            frame = s.scene.frame().astype(np.float32)
            frame[in_zone] += 35.0 * k / n  # +35 grey levels over 60 s, only in the zone
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            out.append(s.det.update(frame, ZONE_MASK, [box], None, s.i / FPS))
            s.i += 1
        return out

    followed = run(10.0)
    assert not any(r.triggered for r in followed)
    assert max(r.area_fraction for r in followed) < 0.02
    frozen = run(0.0)
    assert frozen[-1].triggered and frozen[-1].area_fraction > 0.5


def test_learning_apply_runs_every_update_interval_at_interval_over_tau():
    cfg = make_cfg(update_time_constant_seconds=20.0, update_interval_seconds=0.5)
    s = Session(cfg)
    s.warm_up()
    res = s.step(int(10 * FPS))
    rates = [r.learning_rate_used for r in res if r.learning_rate_used > 0.0]
    assert 19 <= len(rates) <= 21
    assert all(r == pytest.approx(0.5 / 20.0, rel=0.1) for r in rates)
    assert sum(r.learning_rate_used == 0.0 for r in res) == len(res) - len(rates)  # classification only


# ----------------------------------------------------------------- masking rules


def test_exclusion_box_removes_object():
    """Spec test 3: a detection box covering the rectangle yields area_px == 0."""
    s = Session(make_cfg())
    s.warm_up()
    box = np.array(RECT, dtype=np.float32)
    res = s.step(5, rect=RECT, boxes=[box])
    assert all(r.area_px == 0 and r.area_fraction == 0.0 and r.blobs == [] for r in res)
    assert all(not r.mask.any() for r in res)
    assert all(r.persist_seconds == 0.0 and not r.triggered for r in res)
    assert all(r.covered_fraction > 0.10 for r in res)
    # The exclusion is local: a box elsewhere does not hide the object.
    far_box = np.array([40, 120, 100, 160], dtype=np.float32)
    res = s.step(3, rect=RECT, boxes=[far_box])
    assert all(r.area_px >= 0.9 * RECT_AREA for r in res)


def test_exclusion_box_is_dilated():
    """A box that under-covers the object by less than exclusion_dilate_px still hides it
    (dilation absorbs tracker jitter and MOG2's halo around moving objects)."""
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
    s = Session(make_cfg())
    s.warm_up()
    res = s.step(5, rect=RECT_OUTSIDE)
    assert all(r.area_px == 0 and r.blobs == [] and not r.mask.any() for r in res)
    assert all(not r.triggered for r in res)


def test_ignore_mask_excludes_pixels_from_every_decision():
    """debris.ignore_polygons (rasterised by the monitor): an object resting entirely inside
    an ignored area - where a lowered gate arm lies in the zone - never becomes a candidate."""
    ignore = np.zeros((H, W), np.uint8)
    ignore[130:190, 140:220] = 255  # covers RECT with a margin
    s = Session(make_cfg(persist_seconds=0.5))
    s.warm_up()
    res = s.step(int(2 * FPS), rect=RECT, ignore_mask=ignore)
    assert all(r.area_px == 0 and not r.triggered for r in res)
    other = (60, 150, 100, 190)  # outside the ignored area
    res = s.step(int(1 * FPS), rect=other, ignore_mask=ignore)
    assert res[-1].triggered


def test_small_blobs_dropped_by_zone_fraction():
    """Components below min_blob_area_fraction * zone_area are speckle, not debris."""
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
    """Documented limitation: a dark, neutral, uniform object on grey ground (a tyre) is a
    darker same-chromaticity copy of the background to MOG2's shadow test and is dropped when
    detect_shadows=True; detect_shadows=False reports it. A real shadow - the background's
    own texture, darker - is dropped either way (MOG2's shadow test, or the relit test)."""
    tyre = (70.0, 70.0, 70.0)
    s = Session(make_cfg(detect_shadows=True))
    s.warm_up()
    res = s.step(3, rect=RECT, color=tyre)
    assert all(r.area_px < 0.2 * RECT_AREA for r in res)
    s2 = Session(make_cfg(detect_shadows=False))
    s2.warm_up()
    res2 = s2.step(3, rect=RECT, color=tyre)
    assert all(r.area_px >= 0.9 * RECT_AREA for r in res2)
    for shadows in (True, False):
        s3 = Session(make_cfg(detect_shadows=shadows, persist_seconds=0.5))
        s3.warm_up()
        res3 = s3.step(int(2 * FPS), rect=RECT, shade=0.6)
        assert all(r.area_px == 0 and not r.triggered for r in res3)


# --------------------------------------------------------------- size threshold


def test_absolute_minimum_size_catches_small_object_on_a_long_zone():
    """PORT-01: a 30x24 px object is 2.4 % of a 300x100 zone, under min_area_fraction (3 %),
    so the zone share alone never timed it. The frame-relative bar (0.2 % of the frame) does;
    sensor noise alone still never triggers (test_no_false_debris_on_static_scene)."""
    zone = np.zeros((H, W), np.uint8)
    zone[110:210, 10:310] = 255  # 30000 px
    small = (150, 140, 180, 164)  # 30x24 = 720 px

    def run(cfg: DebrisConfig) -> List:
        det = DebrisDetector(cfg)
        sc = Scene(0)
        out = []
        for i in range(60 + int(4 * FPS)):
            rect = small if i >= 60 else None
            out.append(det.update(sc.frame(rect=rect), zone, [], None, i / FPS))
        return out

    res = run(make_cfg(persist_seconds=3.0))
    assert res[-1].area_fraction == pytest.approx(720 / 30000.0, rel=0.1)
    assert res[-1].triggered
    zone_only = run(make_cfg(persist_seconds=3.0, min_object_frame_fraction=1.0))
    assert not any(r.triggered for r in zone_only)


def test_size_bar_is_a_share_of_the_whole_frame_not_of_the_modelled_region():
    """Only a band around the zone is modelled, but the absolute bar is a share of the whole
    processing frame, as documented. On a thin band zone in a 640x480 frame the modelled region
    is a quarter of the frame; a bar taken on the region (154 px instead of 615 px) let blobs of
    ~130 px (registration residue after a camera bump) time as debris. Here a 20x20 px blob
    stays below the bar and a 40x24 px object triggers."""
    hh, ww = 480, 640
    rng = np.random.default_rng(0)
    base = np.clip(rng.normal(120.0, 12.0, size=(hh, ww, 3)), 0, 255).astype(np.float32)
    noise = rng.normal(0.0, 3.0, size=(8, hh, ww, 3)).astype(np.float32)
    zone = np.zeros((hh, ww), np.uint8)
    zone[220:260, 20:620] = 255  # 24000 px: 3 % = 720 px > 0.2 % of the frame = 615 px

    def run(rect: Tuple[int, int, int, int]) -> List:
        det = DebrisDetector(make_cfg(persist_seconds=2.0))
        out = []
        for i in range(60 + int(3 * FPS)):
            img = base.copy()
            if i >= 60:
                x1, y1, x2, y2 = rect
                img[y1:y2, x1:x2] = WHITE
            img += noise[i % 8]
            out.append(det.update(np.clip(img, 0, 255).astype(np.uint8), zone, [], None, i / FPS))
        return out

    small = run((300, 230, 320, 250))  # 400 px
    assert small[-1].area_px == pytest.approx(400, rel=0.15)
    assert not any(r.triggered for r in small)
    large = run((300, 228, 340, 252))  # 960 px
    assert large[-1].triggered


# ------------------------------------------------------------------ illumination guard


def test_illumination_guard_needs_the_rest_of_the_frame_to_change_too():
    """A huge obstruction on a tightly drawn zone must not be mistaken for a lighting change:
    a lighting change lights up the whole image, an object does not. Here a block covers the
    whole zone while the rest of the frame is untouched, so the guard stays out of the way
    and the object alarms as debris."""
    cfg = make_cfg(persist_seconds=1.0)
    covering = (30, 105, 305, 215)  # bounding box of the whole zone
    s = Session(cfg)
    s.warm_up()
    res = s.step(int(2 * FPS), rect=covering, empty=100.0)
    assert res[0].area_fraction >= cfg.max_area_fraction, "the object must fill the zone"
    assert s.det.last_illumination_event is None
    assert all(r.warmed_up for r in res)
    assert res[-1].triggered


def test_change_that_the_gain_cannot_explain_rewarms_instead_of_alarming():
    """An IR-cut filter switch or floodlights change the whole image in a way no gain
    explains: foreground over the zone AND the rest of the frame re-warms the model (reported
    in last_illumination_event) instead of latching an ALARM; max_area_fraction=0 disables it."""
    # Night mode: a high-contrast pattern unrelated to the learned scene, everywhere.
    yy, xx = np.mgrid[0:H, 0:W]
    night = np.where(((yy // 8) + (xx // 8)) % 2 == 0, 30.0, 220.0).astype(np.float32)
    noise = np.random.default_rng(3).normal(0.0, 3.0, size=(8, H, W)).astype(np.float32)

    def run(cfg: DebrisConfig) -> Tuple[List, List]:
        s = Session(cfg)
        s.warm_up()
        res, events = [], []
        for k in range(int(4 * FPS)):
            frame = np.clip(night + noise[k % 8], 0, 255).astype(np.uint8)
            res.append(s.det.update(np.dstack([frame] * 3), ZONE_MASK, [], None, s.i / FPS))
            events.append(s.det.last_illumination_event)
            s.i += 1
        return res, events

    res, events = run(make_cfg(persist_seconds=1.0, warmup_frames=40))
    assert events[0] is not None and events[0]["reason"] == "illumination step"
    assert sum(e is not None for e in events) == 1  # reported exactly once
    assert not any(r.triggered for r in res)
    assert all(r.area_px == 0 for r in res[45:])
    res2, events2 = run(make_cfg(persist_seconds=1.0, warmup_frames=40, max_area_fraction=0.0,
                                 gain_compensation=False))
    assert not any(events2)
    assert res2[-1].triggered and res2[-1].area_fraction > 0.5


# --------------------------------------------------------------- persistence timer


def test_moving_object_never_triggers_debris():
    """Persistence alone is not evidence: an object the detector missed that takes longer
    than persist_seconds to cross the zone (a train under weights without a 'train' class,
    a crowd) must not raise a debris alarm. Each pixel is covered for less than
    persist_seconds and the footprint keeps advancing over background."""
    cfg = make_cfg(persist_seconds=1.0, min_area_fraction=0.02)
    s = Session(cfg)
    s.warm_up()
    # A 60x40 block sliding across the zone at 2 px/frame: 4 s of frames, four times
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


def test_persistence_timer_is_timestamp_based():
    """The same number of frames at 10 fps and 30 fps trigger at the same *seconds*."""
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
    """Debris smaller than the size bar is reported but never times. The bar is
    min(min_area_fraction of the zone, min_object_frame_fraction of the frame)."""
    s = Session(make_cfg(min_area_fraction=0.2, min_object_frame_fraction=1.0,
                         persist_seconds=0.5))  # rect is 10.4 % of the zone
    s.warm_up()
    res = s.step(int(FPS), rect=RECT)
    assert all(r.area_px >= 0.9 * RECT_AREA for r in res)
    assert all(r.persist_seconds == 0.0 and not r.triggered for r in res)
    s2 = Session(make_cfg(min_area_fraction=0.2, min_object_frame_fraction=0.002, persist_seconds=0.5))
    s2.warm_up()
    assert s2.step(int(FPS), rect=RECT)[-1].triggered  # 2400 px > 0.2 % of the 76800 px frame


def test_timer_restarts_if_time_goes_backwards():
    s = Session(make_cfg(persist_seconds=1.0))
    s.warm_up()
    s.step(int(2 * FPS), rect=RECT)
    assert s.det.triggered
    # A looped file without reset(): timestamps jump back; the streak must restart.
    r = s.det.update(s.scene.frame(rect=RECT), ZONE_MASK, [], None, 0.0)
    assert r.persist_seconds == 0.0 and not r.triggered
    r = s.det.update(s.scene.frame(rect=RECT), ZONE_MASK, [], None, 0.5)
    assert r.persist_seconds == pytest.approx(0.5)


def test_time_going_backwards_reanchors_hold_history_and_update_cadence():
    """Deadlines kept in the old time base must not stall the detector after a jump back:
    a hold set at t = 20 s would otherwise last until the new clock reaches 22 s, and the
    motion history and the learning cadence would never advance again."""
    s = Session(make_cfg(update_time_constant_seconds=10.0, hold_linger_seconds=2.0))
    s.warm_up()
    s.i = int(20 * FPS)
    s.step(5, hold=True)
    s.i = 0  # the file loops without reset()
    res = s.step(int(2 * FPS))
    assert not any(r.held for r in res[1:])
    assert any(r.learning_rate_used > 0.0 for r in res)
    # the immobility test still works on the new clock: a textured block creeping at 10 px/s
    # covers each pixel for 6 s, so only the content test (which needs the history) stops it
    tex = np.clip(np.random.default_rng(4).normal(128.0, 60.0, size=(40, 60, 3)), 0, 255).astype(np.float32)
    slide = []
    for k in range(int(8 * FPS)):
        x = 40 + int(10 * k / FPS)
        img = s.scene.base.copy()
        img[140:180, x:x + 60] = tex
        img += s.scene.noise[s.scene.calls % Scene.NOISE_BANK]
        s.scene.calls += 1
        slide.append(s.det.update(np.clip(img, 0, 255).astype(np.uint8), ZONE_MASK, [], None, s.i / FPS))
        s.i += 1
    assert not any(r.triggered for r in slide)


# ---------------------------------------------------------------------- plumbing


def test_reset_clears_model_and_timers():
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
    with pytest.raises(ValueError):
        det.update(frame, ZONE_MASK, [], None, 0.0, to_reference=np.eye(2))


def test_accepts_bool_zone_mask_and_grey_frames():
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
    """Sensor noise alone never produces debris on a frozen or selectively updated model (the
    varThreshold default is well above the noise level); an idle camera stays CLEAR."""
    for tau in (0.0, 5.0):
        s = Session(make_cfg(update_time_constant_seconds=tau))
        s.warm_up()
        res = s.step(int(20 * FPS))
        assert all(r.area_px == 0 and not r.triggered for r in res)
        assert all(r.trusted and r.state == "ok" and r.gain == pytest.approx(1.0, abs=0.02) for r in res)


def test_process_width_downscales_internally_and_maps_results_back():
    """HD frames are processed at process_width (cost); the mask, blobs and area come back in
    the caller's pixels."""
    big_h, big_w = 2 * H, 2 * W
    zone = polygon_mask((big_h, big_w), ZONE * 2.0)
    sc = Scene(0)
    det = DebrisDetector(make_cfg(process_width=W, persist_seconds=0.5))
    rect = (2 * RECT[0], 2 * RECT[1], 2 * RECT[2], 2 * RECT[3])
    res = []
    for i in range(60 + int(FPS)):
        small = sc.frame(rect=RECT if i >= 60 else None)
        frame = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        res.append(det.update(frame, zone, [], None, i / FPS))
    assert det.background_image().shape[:2] == (H, W)
    r = res[-1]
    assert r.triggered and r.mask.shape == (big_h, big_w)
    assert r.area_px == pytest.approx(4 * RECT_AREA, rel=0.1)
    assert int(np.count_nonzero(r.mask)) == pytest.approx(r.area_px, rel=0.1)
    x, y, w, h, area = r.blobs[0]
    assert abs(x - rect[0]) <= 4 and abs(y - rect[1]) <= 4 and abs(w - 120) <= 6 and abs(h - 80) <= 6
