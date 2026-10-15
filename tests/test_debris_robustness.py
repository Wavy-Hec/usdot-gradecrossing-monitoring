"""Debris backstop on the conditions that broke it on real footage (synthetic 320x240 scenes).

Each test is a small scene that reproduces one reviewed failure and pins the fixed
behaviour, usually together with the old failure as a contrast:

* lighting ramps and steps (ours-correctness M1), a relight followed by an object (PORT-01);
* sub-pixel camera jitter on a static object (PORT-02), camera shifts with and without
  registration (F05), unregistrable frames;
* slow textured trains, hazard lights (PORT-03 / MISS-04), the train hold;
* candidates under boxes: traffic over debris, sporadic detections (M2, PORT-03);
* ghosts of objects the warm-up learned (OCC-01: no ghost deadlock), and real objects across
  a rail that look like ghosts (never healed), trust reporting;
* sensitivity next to busy, box-free motion outside a thin zone (no global variance floor).

The background is smooth (blurred) noise, so a sub-pixel shift changes it the way it changes
a real, band-limited camera image.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytest

from gradecrossing.background import DebrisDetector
from gradecrossing.config import DebrisConfig
from gradecrossing.geometry import polygon_mask

H, W = 240, 320
FPS = 30.0
ZONE = np.array([[30, 120], [290, 110], [300, 200], [20, 210]], dtype=np.float32)
ZONE_MASK = polygon_mask((H, W), ZONE)
OBJ = (140, 140, 200, 180)  # 60x40, inside the zone


def make_cfg(**overrides) -> DebrisConfig:
    base = dict(warmup_frames=60)
    base.update(overrides)
    return DebrisConfig(**base)


def textured_patch(seed: int, h: int, w: int, color: Sequence[float] = (60.0, 70.0, 170.0),
                   std: float = 60.0) -> np.ndarray:
    """An object: a distinct colour with smooth texture (std ~15 after the blur)."""
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.normal(0.0, std, size=(h, w)).astype(np.float32), (0, 0), 1.0)
    return np.clip(np.asarray(color, np.float32)[None, None, :] + tex[..., None], 0, 255)


class Scene:
    """Smooth textured background with hard-edged blocks (features), sprites, global gain,
    camera shift and sensor noise."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        base = cv2.GaussianBlur(rng.normal(0.0, 70.0, size=(H, W)).astype(np.float32), (0, 0), 1.0)
        base = np.dstack([110.0 + base, 115.0 + base, 120.0 + base])
        for _ in range(40):  # blocks give the scene strong edges, like kerbs, rails and signs
            x, y = int(rng.integers(0, W - 30)), int(rng.integers(0, H - 20))
            w, h = int(rng.integers(8, 30)), int(rng.integers(6, 20))
            base[y:y + h, x:x + w] = rng.uniform(40.0, 200.0, size=3)
        self.base = np.clip(cv2.GaussianBlur(base.astype(np.float32), (0, 0), 0.7), 20, 220)
        self.noise = rng.normal(0.0, 2.0, size=(8, H, W, 3)).astype(np.float32)
        self.calls = 0

    def frame(self, sprites: Sequence[Tuple[Tuple[int, int], np.ndarray]] = (), gain: float = 1.0,
              shift: Tuple[float, float] = (0.0, 0.0), zone_gain: Optional[float] = None) -> np.ndarray:
        img = self.base.copy()
        for (x, y), spr in sprites:
            h, w = spr.shape[:2]
            x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            if x1 > x0 and y1 > y0:
                img[y0:y1, x0:x1] = spr[y0 - y:y1 - y, x0 - x:x1 - x]
        if zone_gain is not None:
            img[ZONE_MASK > 0] *= zone_gain
        img *= gain
        if shift != (0.0, 0.0):
            m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            img = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = img + self.noise[self.calls % 8]
        self.calls += 1
        return np.clip(img, 0, 255).astype(np.uint8)


class Run:
    """A detector fed one scene; records every result."""

    def __init__(self, cfg: DebrisConfig, seed: int = 0) -> None:
        self.det = DebrisDetector(cfg)
        self.scene = Scene(seed)
        self.i = 0
        self.results: List = []

    @property
    def t(self) -> float:
        return self.i / FPS

    def step(self, n: int = 1, sprites=(), boxes=(), gain: float = 1.0, shift=(0.0, 0.0), zone=None,
             zone_gain=None, **upd) -> List:
        out = []
        for _ in range(n):
            frame = self.scene.frame(sprites=sprites, gain=gain, shift=shift, zone_gain=zone_gain)
            r = self.det.update(frame, ZONE_MASK if zone is None else zone, list(boxes), None, self.t, **upd)
            out.append(r)
            self.i += 1
        self.results.extend(out)
        return out

    def warm_up(self, **kw) -> None:
        while not self.det.warmed_up:
            self.step(1, **kw)


def first_trigger(results) -> Optional[int]:
    return next((k for k, r in enumerate(results) if r.triggered), None)


# --------------------------------------------------------------------------- lighting


@pytest.mark.parametrize("gain,ramp_frames", [(1.6, 1), (1.6, 10), (1.6, 30), (1.6, 90),
                                              (1.25, 1), (1.25, 30), (0.7, 30), (0.6, 1)])
def test_lighting_ramps_and_steps_never_latch_debris(gain, ramp_frames):
    """M1: a gain ramp over a few frames, a step that covers less than 75 % of the zone, or a
    darkening used to start a streak that blocked the refresh and latched a 50-100 % debris
    ALARM. The global gain is now divided out before classification."""
    run = Run(make_cfg())
    run.warm_up()
    run.step(int(FPS))
    for k in range(int(20 * FPS)):
        g = 1.0 + (gain - 1.0) * min(1.0, (k + 1) / float(ramp_frames))
        run.step(1, gain=g)
    after = run.results[-int(20 * FPS):]
    assert not any(r.triggered for r in after)
    assert max(r.area_fraction for r in after) < 0.03
    assert after[-1].gain == pytest.approx(gain, rel=0.03)
    assert run.det.last_illumination_event is None


def test_lighting_step_while_a_vehicle_covers_the_zone_does_not_latch():
    """M1: x1.6 in one frame while a train / vehicle box covers most of the zone."""
    box = np.array([60, 100, 260, 215], dtype=np.float32)
    car = textured_patch(3, 115, 200)
    run = Run(make_cfg())
    run.warm_up()
    run.step(int(FPS), sprites=[((60, 100), car)], boxes=[box])
    run.step(int(3 * FPS), sprites=[((60, 100), car)], boxes=[box], gain=1.6)
    after = run.step(int(10 * FPS), gain=1.6)  # the vehicle leaves; the light stays
    assert not any(r.triggered for r in run.results)
    assert max(r.area_fraction for r in after) < 0.03


def test_object_after_a_lighting_step_still_alarms():
    """PORT-01 relight probe: x1.2 at once, an object 2 s later. The old guard re-warmed and
    learned the object during the blind warm-up; now the step is compensated and the object
    alarms persist_seconds after it appears."""
    obj = textured_patch(5, 40, 60)
    run = Run(make_cfg())
    run.warm_up()
    run.step(int(FPS))
    run.step(int(2 * FPS), gain=1.2)
    res = run.step(int(4 * FPS), sprites=[((OBJ[0], OBJ[1]), obj)], gain=1.2)
    k = first_trigger(res)
    assert k is not None and abs(k / FPS - 3.0) <= 0.2
    assert run.det.last_illumination_event is None


def test_gain_far_outside_the_trusted_range_rewarms():
    """Night falls (x0.35): the compensation still keeps debris quiet, but a gain beyond
    rewarm_gain for a second means the model is stale, so it is re-warmed (reported once)."""
    run = Run(make_cfg(rewarm_gain=2.0))
    run.warm_up()
    events = []
    for _ in range(int(2 * FPS)):
        run.step(1, gain=0.35)
        events.append(run.det.last_illumination_event)
    fired = [e for e in events if e is not None]
    assert len(fired) == 1 and fired[0]["reason"] == "gain"
    assert fired[0]["gain"][0] == pytest.approx(0.35, rel=0.05)
    assert fired[0]["timestamp"] == pytest.approx(2.0 + 1.0, abs=0.05)  # 1 s after the gain left the range
    assert not any(r.triggered for r in run.results)


# ------------------------------------------------------------- jitter & registration


def jitter_table(std: float, n: int, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, std, size=(n, 2))


def test_sub_pixel_camera_jitter_does_not_blind_the_immobility_test():
    """PORT-02: 0.3 px std frame-to-frame jitter (E7's median shake) made every textured
    static object 'moving' for the old frame-to-frame test, so a stopped car never alarmed.
    The motion test now compares against a frame 0.5 s old with a +-1 px tolerance."""
    jit = jitter_table(0.3, 2000)
    car = textured_patch(3, 40, 60)
    run = Run(make_cfg())
    while not run.det.warmed_up:
        run.step(1, shift=tuple(jit[run.i]))
    res = []
    for _ in range(int(6 * FPS)):
        res.extend(run.step(1, sprites=[((OBJ[0], OBJ[1]), car)], shift=tuple(jit[run.i])))
    k = first_trigger(res)
    assert k is not None and k / FPS <= 3.0 + 0.5
    assert all(r.triggered for r in res[k:])


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def shifted_zone(dx: float, dy: float) -> np.ndarray:
    return polygon_mask((H, W), ZONE + np.float32([dx, dy]))


def test_registered_camera_shift_raises_no_debris_and_reports_off_frame_pixels():
    """F05 / PORT-02: a permanent (8, 4) px camera bump, tracked by the stabiliser, latched a
    debris ALARM because the model stayed in raw-frame coordinates. With to_reference the
    frame is warped back; the pixels that left the frame are unknown, not debris."""
    shift = (8.0, 4.0)
    to_ref = translation(-shift[0], -shift[1])
    run = Run(make_cfg())
    run.warm_up(to_reference=np.eye(3))
    res = run.step(int(8 * FPS), shift=shift, zone=shifted_zone(*shift), to_reference=to_ref)
    assert not any(r.triggered for r in res)
    assert max(r.area_fraction for r in res) < 0.02
    assert all(r.registered for r in res)
    # a real object placed after the bump still alarms, at its true place
    obj = textured_patch(5, 40, 60)
    res = run.step(int(4 * FPS), sprites=[((OBJ[0], OBJ[1]), obj)], shift=shift, zone=shifted_zone(*shift),
                   to_reference=to_ref)
    assert res[-1].triggered
    x, y, w, h, _a = max(res[-1].blobs, key=lambda b: b[4])
    assert abs(x - (OBJ[0] + shift[0])) <= 3 and abs(y - (OBJ[1] + shift[1])) <= 3  # caller coordinates


def test_pixels_pushed_out_of_the_frame_are_unknown_not_debris():
    left = (-24.0, 0.0)  # the zone's left end (x = 20..30) leaves the frame
    run = Run(make_cfg())
    run.warm_up(to_reference=np.eye(3))
    res = run.step(int(5 * FPS), shift=left, zone=shifted_zone(*left), to_reference=translation(24.0, 0.0))
    assert 0.0 < res[-1].unknown_fraction < 0.2
    assert res[-1].trusted
    assert not any(r.triggered for r in res) and max(r.area_fraction for r in res) < 0.02


def test_unregistered_camera_shift_is_what_used_to_alarm():
    """Contrast for the test above: the same bump without registration becomes debris."""
    shift = (8.0, 4.0)
    run = Run(make_cfg())
    run.warm_up()
    res = run.step(int(8 * FPS), shift=shift, zone=shifted_zone(*shift))
    assert res[-1].triggered


def test_registered_exclusion_boxes_are_transformed_with_the_frame():
    shift = (10.0, 5.0)
    to_ref = translation(-shift[0], -shift[1])
    car = textured_patch(3, 40, 60)
    box_cur = np.array([OBJ[0] + shift[0], OBJ[1] + shift[1], OBJ[2] + shift[0], OBJ[3] + shift[1]], np.float32)
    run = Run(make_cfg(exclusion_dilate_px=4, persist_seconds=0.5))
    run.warm_up(to_reference=np.eye(3))
    res = run.step(int(2 * FPS), sprites=[((OBJ[0], OBJ[1]), car)], boxes=[box_cur], shift=shift,
                   zone=shifted_zone(*shift), to_reference=to_ref)
    assert all(r.area_px == 0 and not r.triggered for r in res)
    assert res[-1].covered_fraction > 0.05


def test_registration_failure_holds_timers():
    """While the frame cannot be registered nothing is classified: no new candidate starts,
    an existing candidate neither advances nor resets, nothing is learned, trusted is False."""
    obj = textured_patch(5, 40, 60)
    spr = [((OBJ[0], OBJ[1]), obj)]
    run = Run(make_cfg(persist_seconds=3.0, update_time_constant_seconds=2.0))
    run.warm_up()
    a = run.step(int(1.5 * FPS), sprites=spr)
    held = run.step(int(5 * FPS), sprites=spr, registration_ok=False)
    assert all(r.held and not r.trusted and r.state == "held (registration)" for r in held)
    assert all(r.persist_seconds == pytest.approx(a[-1].persist_seconds) for r in held)
    assert all(r.learning_rate_used == 0.0 for r in held)
    back = run.step(int(2 * FPS), sprites=spr)
    assert back[0].persist_seconds == pytest.approx(a[-1].persist_seconds + 1.0 / FPS, abs=0.05)
    k = first_trigger(back)
    assert k is not None and abs((k + 1) / FPS - (3.0 - 1.5)) <= 0.15
    # an object that appears during the failure is not a candidate before it is seen
    run2 = Run(make_cfg(persist_seconds=1.0))
    run2.warm_up()
    held2 = run2.step(int(3 * FPS), sprites=spr, registration_ok=False)
    assert all(r.persist_seconds == 0.0 and not r.triggered for r in held2)


# ----------------------------------------------------------------- immobility


def test_slow_textured_train_never_triggers():
    """A train without a detection box (weights with no train class) wider than the zone,
    crossing at 15 px/s: every pixel stays foreground for the whole passage, but its texture
    keeps changing against the 0.5 s old frame, so no candidate survives."""
    train = textured_patch(11, 100, 900, color=(150.0, 60.0, 50.0), std=80.0)
    run = Run(make_cfg(persist_seconds=3.0))
    run.warm_up()
    res = []
    for k in range(int(30 * FPS)):
        x = int(round(-880 + 15.0 * k / FPS))
        res.extend(run.step(1, sprites=[((x, 110), train)]))
    assert max(r.area_fraction for r in res) > 0.5  # it really covered the zone
    assert not any(r.triggered for r in res)
    assert max(r.persist_seconds for r in res) < 1.5


def test_hazard_lights_on_a_stalled_car_do_not_delay_the_alarm():
    """MISS-04: blinking lamps made the advisor's motion gate hold a stalled car's cells; here
    only the lamp pixels restart, the body keeps its persistence."""
    car = textured_patch(3, 40, 60)
    lamp_on = np.array([40, 160, 255], np.float32)
    lamp_off = np.array([20, 20, 90], np.float32)
    lamps = [(2, 8, 2, 12), (2, 8, 48, 58), (32, 38, 2, 12), (32, 38, 48, 58)]
    run = Run(make_cfg(persist_seconds=3.0))
    run.warm_up()
    res = []
    for k in range(int(6 * FPS)):
        spr = car.copy()
        on = (k / FPS * 1.5) % 1.0 < 0.5
        for y0, y1, x0, x1 in lamps:
            spr[y0:y1, x0:x1] = lamp_on if on else lamp_off
        res.extend(run.step(1, sprites=[((OBJ[0], OBJ[1]), spr)]))
    k = first_trigger(res)
    assert k is not None and k / FPS <= 3.0 + 0.2


def test_train_hold_freezes_timers_blocks_new_candidates_and_learning():
    obj = textured_patch(5, 40, 60)
    spr = [((OBJ[0], OBJ[1]), obj)]
    cfg = make_cfg(persist_seconds=3.0, hold_linger_seconds=1.0, update_time_constant_seconds=2.0)
    run = Run(cfg)
    run.warm_up()
    a = run.step(int(1.0 * FPS), sprites=spr)
    bg_before = run.det.background_image().copy()
    held = run.step(int(4 * FPS), sprites=spr, hold=True)
    assert all(r.held and r.state == "held (train)" for r in held)
    assert all(r.persist_seconds == pytest.approx(a[-1].persist_seconds) for r in held)
    assert all(r.learning_rate_used == 0.0 for r in held)
    assert np.array_equal(bg_before, run.det.background_image())
    linger = run.step(int(1.0 * FPS) - 1, sprites=spr)  # hold=False, but within hold_linger_seconds
    assert all(r.held for r in linger)
    after = run.step(int(3 * FPS), sprites=spr)
    assert not after[-1].held
    k = first_trigger(after)
    assert k is not None and abs((k + 1) / FPS - 2.0) <= 0.2  # 1 s seen before + 2 s after the hold
    # nothing that appears during a hold is a candidate
    run2 = Run(cfg)
    run2.warm_up()
    held2 = run2.step(int(4 * FPS), sprites=spr, hold=True)
    assert all(r.persist_seconds == 0.0 and not r.triggered for r in held2)


def test_a_new_hold_restarts_the_candidates_seen_since_the_last_one():
    """FO1 (clip03): the detector does not label a passing train on every frame. In a gap
    between two holds the train is an ordinary candidate (a uniform car passes the immobility
    test), and frozen rather than reset by each hold, its age added up over the gaps of one
    passage (2.0 of 3.0 s on clip03; an ALARM with a smaller size bar). A hold that starts
    again within persist_seconds restarts the candidates of the gap. Something that was there
    before the first hold keeps its age."""
    car = [((OBJ[0], OBJ[1]), textured_patch(5, 40, 60))]
    cfg = make_cfg(persist_seconds=3.0, hold_linger_seconds=0.0, update_time_constant_seconds=2.0)
    run = Run(cfg)
    run.warm_up()
    run.step(int(0.5 * FPS), hold=True)  # the train arrives
    passage = []
    for _ in range(3):  # three 1.5 s gaps between its labels: 4.5 s of car in all
        passage += run.step(int(1.5 * FPS), sprites=car)
        passage += run.step(int(0.5 * FPS), sprites=car, hold=True)
    passage += run.step(int(1.5 * FPS), sprites=car)
    assert not any(r.triggered for r in passage)
    assert max(r.persist_seconds for r in passage) <= 1.5 + 0.1
    held = [r for r in passage if r.held]
    assert held and all(r.persist_seconds == 0.0 for r in held)  # reported without the gap's pixels
    # an obstruction seen before the first hold keeps its age through the passage
    run2 = Run(cfg)
    run2.warm_up()
    run2.step(int(1.0 * FPS), sprites=car)
    run2.step(int(0.5 * FPS), sprites=car, hold=True)
    gap = run2.step(int(1.0 * FPS), sprites=car)
    assert gap[-1].persist_seconds >= 2.0 - 0.1
    run2.step(int(0.5 * FPS), sprites=car, hold=True)
    after = run2.step(int(2 * FPS), sprites=car)
    k = first_trigger(after)
    assert k is not None and abs((k + 1) / FPS - 1.0) <= 0.2  # 1 s before + 1 s in the gap + 1 s


def test_a_hold_long_after_the_last_one_restarts_nothing():
    """A gap of persist_seconds or more is not one train seen between its detections: a
    candidate of that gap keeps its age through the next hold (fail-safe for an obstruction
    that arrived shortly before a train)."""
    obj = [((OBJ[0], OBJ[1]), textured_patch(5, 40, 60))]
    cfg = make_cfg(persist_seconds=3.0, hold_linger_seconds=0.0, update_time_constant_seconds=2.0)
    run = Run(cfg)
    run.warm_up()
    run.step(int(0.5 * FPS), hold=True)  # a train passed
    run.step(int(2.0 * FPS))  # nothing for 2 s
    run.step(int(1.5 * FPS), sprites=obj)  # an obstruction, 3.5 s after that hold ended
    run.step(int(1.0 * FPS), sprites=obj, hold=True)  # the next train
    after = run.step(int(3 * FPS), sprites=obj)
    k = first_trigger(after)
    assert k is not None and abs((k + 1) / FPS - 1.5) <= 0.2  # 1.5 s before the hold + 1.5 s after


# ------------------------------------------------------- covered and sporadic


def test_traffic_over_debris_keeps_the_persistence():
    """PORT-03: cars crossing over an obstruction every 2.5 s (covering it ~40 % of the time)
    used to restart the 3 s clock on every pass (first alarm only after the traffic stopped).
    A covered candidate is unknown, not gone: it keeps its seen-static time."""
    obj = textured_patch(5, 40, 60)
    car = textured_patch(9, 70, 50, color=(200.0, 200.0, 60.0))
    run = Run(make_cfg(persist_seconds=3.0))
    run.warm_up()
    res = []
    for k in range(int(12 * FPS)):
        t = k / FPS
        phase = t % 2.5
        sprites = [((OBJ[0], OBJ[1]), obj)]
        boxes = []
        if phase < 1.0:  # a car drives down across the object for 1 s of every 2.5 s
            y = int(round(60 + 120 * phase))
            sprites.append(((OBJ[0] + 5, y), car))
            boxes.append(np.array([OBJ[0] + 5, y, OBJ[0] + 55, y + 70], np.float32))
        res.extend(run.step(1, sprites=sprites, boxes=boxes))
    k = first_trigger(res)
    assert k is not None and k / FPS <= 7.0


def test_sporadic_detection_box_does_not_restart_the_clock():
    """M2: an obstruction boxed by the detector once every 45 frames (1.5 s) never reached 3 s."""
    obj = textured_patch(5, 40, 60)
    box = np.array(OBJ, dtype=np.float32)
    run = Run(make_cfg(persist_seconds=3.0))
    run.warm_up()
    res = []
    for k in range(int(5 * FPS)):
        res.extend(run.step(1, sprites=[((OBJ[0], OBJ[1]), obj)], boxes=[box] if k % 45 == 44 else []))
    k = first_trigger(res)
    assert k is not None and k / FPS <= 3.3


def test_occlusion_bridge_is_bounded():
    obj = textured_patch(5, 40, 60)
    spr = [((OBJ[0], OBJ[1]), obj)]
    box = np.array(OBJ, dtype=np.float32)
    run = Run(make_cfg(persist_seconds=3.0, occlusion_bridge_seconds=2.0))
    run.warm_up()
    run.step(int(2 * FPS), sprites=spr)
    run.step(int(3 * FPS), sprites=spr, boxes=[box])  # covered longer than the bridge
    back = run.step(int(1 * FPS), sprites=spr)
    assert back[0].persist_seconds == 0.0
    assert not any(r.triggered for r in back)


# --------------------------------------------------------------------- ghosts


def test_warmup_ghost_of_an_undetected_object_heals_without_alarming():
    """OCC-01 (do not port the ghost deadlock) and clip03's burned-in caption: something the
    warm-up learned without a detection box leaves; the road it uncovers differs from the
    model. Its outline exists only in the model, so it is re-learned instead of latching an
    ALARM - and a real obstruction on the same spot afterwards still alarms."""
    sign = textured_patch(21, 30, 70, color=(40.0, 200.0, 40.0))
    run = Run(make_cfg(persist_seconds=3.0, ghost_heal_seconds=1.0, update_time_constant_seconds=10.0))
    run.warm_up(sprites=[((OBJ[0], OBJ[1]), sign)])
    run.step(int(1 * FPS), sprites=[((OBJ[0], OBJ[1]), sign)])
    gone = run.step(int(12 * FPS))
    assert not any(r.triggered for r in gone)
    assert gone[-1].area_px == 0 and gone[-1].unknown_fraction == 0.0
    obj = textured_patch(5, 40, 60)
    res = run.step(int(4 * FPS), sprites=[((OBJ[0], OBJ[1]), obj)])
    assert res[-1].triggered


def test_object_left_behind_by_a_departing_vehicle_is_not_healed():
    """F04: the ghost test must not absorb a load shed by a truck that then drives off. The
    load has its own outline in the frame, so it alarms."""
    truck = textured_patch(9, 70, 100, color=(190.0, 180.0, 170.0))
    load = textured_patch(5, 40, 60)
    box = np.array([OBJ[0] - 20, OBJ[1] - 15, OBJ[0] + 80, OBJ[1] + 55], np.float32)
    run = Run(make_cfg(persist_seconds=3.0, ghost_heal_seconds=1.0, update_time_constant_seconds=10.0))
    run.warm_up()
    run.step(int(3 * FPS), sprites=[((OBJ[0] - 20, OBJ[1] - 15), truck)], boxes=[box])
    res = run.step(int(5 * FPS), sprites=[((OBJ[0], OBJ[1]), load)])
    assert res[-1].triggered
    assert all(r.unknown_fraction == 0.0 for r in res)


class RailScene:
    """Dark textured ballast crossed by bright rails, as on the team's thin band zones.

    A dark object on dark ballast is 'shadow' or background there and differs from the model
    only where it covers a rail, so what MOG2 sees of it has the rail's outline: a colour step
    in the model and none in the frame - exactly what a rail-shaped ghost looks like."""

    ZONE = polygon_mask((H, W), np.array([[10, 130], [310, 130], [310, 205], [10, 205]], np.float32))

    def __init__(self, rails: Sequence[Tuple[int, int]] = ((150, 158),)) -> None:
        rng = np.random.default_rng(0)
        tex = cv2.GaussianBlur(rng.normal(0.0, 30.0, (H, W)).astype(np.float32), (0, 0), 1.0)
        ballast = np.dstack([62.0 + tex, 66.0 + tex, 70.0 + tex])
        base = ballast.copy()
        for y0, y1 in rails:
            base[y0:y1] = (165.0, 170.0, 175.0)
        self.base = np.clip(cv2.GaussianBlur(base, (0, 0), 0.7), 5, 250)
        self.ballast = np.clip(cv2.GaussianBlur(ballast, (0, 0), 0.7), 5, 250)  # no rails
        self.dark = np.clip(np.dstack([58.0, 62.0, 66.0]) + cv2.GaussianBlur(
            rng.normal(0.0, 25.0, (46, 80)).astype(np.float32), (0, 0), 1.0)[..., None], 0, 255)
        self.noise = rng.normal(0.0, 2.0, (8, H, W, 3)).astype(np.float32)
        self.calls = 0

    def frame(self, obj: Optional[str] = None) -> np.ndarray:
        img = self.base.copy()
        if obj == "ballast":  # looks exactly like the ballast: a stretch of rail disappears
            img[140:170, 120:200] = self.ballast[140:170, 120:200]
        elif obj == "dark":  # dark textured object across two rails
            img[145:191, 120:200] = self.dark
        img = img + self.noise[self.calls % 8]
        self.calls += 1
        return np.clip(img, 0, 255).astype(np.uint8)


def run_rail(cfg: DebrisConfig, obj: str, rails=((150, 158),), onset_after_warmup: float = 10.0,
             seconds_after_onset: float = 8.0) -> List:
    """Warm up on the empty scene, watch it for ``onset_after_warmup`` s, then the object."""
    det, scene = DebrisDetector(cfg), RailScene(rails)
    i = 0
    while not det.warmed_up:
        det.update(scene.frame(), RailScene.ZONE, [], None, i / FPS)
        i += 1
    for _ in range(int(onset_after_warmup * FPS)):
        det.update(scene.frame(), RailScene.ZONE, [], None, i / FPS)
        i += 1
    out = []
    for _ in range(int(seconds_after_onset * FPS)):
        out.append(det.update(scene.frame(obj), RailScene.ZONE, [], None, i / FPS))
        i += 1
    return out


@pytest.mark.parametrize("obj,rails", [("ballast", ((150, 158),)), ("dark", ((150, 156), (180, 186)))])
def test_object_across_a_rail_on_long_watched_road_is_never_healed(obj, rails):
    """Review of the ghost healer: on clip04-06 it re-learned undetected real objects lying
    across a rail (a dark patch at clip04 x=460 never alarmed; base alarmed at 21.0 s). The
    image evidence is ambiguous, so a heal also needs young model content: this road had been
    watched for 10 s before the object arrived, so the object alarms and stays alarmed."""
    res = run_rail(make_cfg(warmup_frames=None, warmup_seconds=2.0), obj)
    k = first_trigger(res)
    assert k is not None and k / FPS == pytest.approx(3.0, abs=0.2)
    assert all(r.triggered for r in res[k:])
    assert res[-1].unknown_fraction == 0.0


def test_ghost_heal_never_runs_while_debris_is_triggered():
    """Without the age rule the same object is indistinguishable from a ghost (it is healed
    before it can alarm). A heal that only fires after the ALARM would end it silently while the
    object is still there, so no heal runs while debris is triggered."""
    no_age = dict(warmup_frames=None, warmup_seconds=2.0, ghost_heal_max_seen_seconds=0.0)
    early = run_rail(make_cfg(ghost_heal_seconds=2.0, **no_age), "ballast")
    assert not any(r.triggered for r in early)  # image evidence alone: healed as a ghost
    late = run_rail(make_cfg(ghost_heal_seconds=5.0, **no_age), "ballast")
    k = first_trigger(late)
    assert k is not None and all(r.triggered for r in late[k:])


# -------------------------------------------------------------------- variance


def test_busy_motion_outside_the_zone_does_not_blind_the_zone():
    """Review: before every learning apply the selective update raised MOG2's global varMin to
    half the noise measured over the whole modelled region, motion outside the zone included
    (capped at 37.5 against a default of 4), which made every zone pixel less sensitive - a
    dark patch base caught at clip04 x=556 was never foreground. Covered pixels now keep their
    own residual variance (dithered paste), and foliage-like flicker above and below a thin
    zone leaves a low-contrast object in it detectable."""
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.normal(0.0, 12.0, (H, W)).astype(np.float32), (0, 0), 1.0)
    base = np.clip(np.dstack([100.0 + tex, 105.0 + tex, 110.0 + tex]), 10, 240)
    zone = polygon_mask((H, W), np.array([[20, 110], [300, 110], [300, 150], [20, 150]], np.float32))
    busy = np.zeros((H, W), bool)
    busy[60:106] = True
    busy[154:200] = True
    noise = rng.normal(0.0, 2.0, (8, H, W, 3)).astype(np.float32)
    flicker = rng.normal(0.0, 18.0, (16, H, W, 1)).astype(np.float32)
    obj = (np.median(base[115:145, 130:190].reshape(-1, 3), axis=0) + 20.0
           + rng.normal(0.0, 4.0, (30, 60, 1)).astype(np.float32))
    det = DebrisDetector(make_cfg(warmup_frames=None, warmup_seconds=2.0))
    res = []
    for i in range(int(11 * FPS)):
        img = base + noise[i % 8]
        img[busy] += flicker[i % 16][busy]
        if i >= int(6 * FPS):
            img[115:145, 130:190] = obj + noise[i % 8][115:145, 130:190]
        res.append(det.update(np.clip(img, 0, 255).astype(np.uint8), zone, [], None, i / FPS))
    assert not any(r.triggered for r in res[:int(6 * FPS)])
    k = first_trigger(res)
    assert k is not None and (k - int(6 * FPS)) / FPS == pytest.approx(3.0, abs=0.3)
    assert all(r.triggered for r in res[k:])


# ---------------------------------------------------------------------- health


def test_trust_follows_the_unknown_share_of_the_zone():
    """A long vehicle across most of the zone during the warm-up leaves most of it unknown:
    trusted is False (the monitor reports it), and recovers once the zone has been seen."""
    box = np.array([10, 100, 300, 220], dtype=np.float32)
    bus = textured_patch(9, 120, 290, color=(60.0, 160.0, 200.0))
    run = Run(make_cfg(update_time_constant_seconds=4.0, learn_seconds=1.0, max_unknown_fraction=0.5))
    run.warm_up(sprites=[((10, 100), bus)], boxes=[box])
    res = run.step(int(1 * FPS), sprites=[((10, 100), bus)], boxes=[box])
    assert res[-1].unknown_fraction > 0.9 and not res[-1].trusted and res[-1].state.startswith("untrusted")
    res = run.step(int(15 * FPS))
    assert res[-1].trusted and res[-1].unknown_fraction == 0.0 and res[-1].state == "ok"
    assert not any(r.triggered for r in res)
