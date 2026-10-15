"""Regression tests for the zone stabilizer's motion model, dead-zone, temporal-consistency
gate, ``to_reference`` transform (keyframe re-anchors included) and featureless start-up.

Synthetic, seeded imagery only (real-footage numbers belong in the review reports). The
scenes reproduce, in miniature, the failure modes measured on the team's clips:

* clip05/clip03: a passing train occludes most of the background while the camera shakes;
  the old 8-DOF homography fitted the surviving minority of points and slid the polygon by
  up to the full height of the band zone (GAEL-20);
* clip02: single-frame zone jumps of 45-64 px reported healthy (GAEL-M2 / F11 / F10);
* static cameras: the polygon jittered by 2-6 px with nothing moving (GAEL-20);
* keyframe re-anchors chained estimation errors into a random walk (F19), and the transform
  the debris path needs had to be composed across them (F05 CORRECTED);
* a black first frame anchored the stabilizer on nothing (D13 / OCC-03).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from gradecrossing.config import ZoneConfig
from gradecrossing.geometry import transform_points
from gradecrossing.zone import ZoneStabilizer

W, H = 640, 480
POLY = [[150, 300], [450, 280], [520, 400], [120, 420]]


def textured_image(seed: int = 0) -> np.ndarray:
    """640x480 BGR image with rich, corner-heavy texture (same recipe as test_zone.py)."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    gray = cv2.GaussianBlur(noise, (3, 3), 0)
    for _ in range(12):
        x, y = int(rng.integers(20, W - 80)), int(rng.integers(20, H - 80))
        w, h = int(rng.integers(20, 70)), int(rng.integers(20, 70))
        cv2.rectangle(gray, (x, y), (x + w, y + h), int(rng.choice([0, 255])), -1)
    for _ in range(8):
        x, y = int(rng.integers(30, W - 30)), int(rng.integers(30, H - 30))
        cv2.circle(gray, (x, y), int(rng.integers(8, 25)), int(rng.choice([0, 255])), -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def shifted(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def similarity(angle_deg: float, tx: float, ty: float, center=(W / 2.0, H / 2.0)) -> np.ndarray:
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return np.vstack([M, [0.0, 0.0, 1.0]])


def make(polygon=POLY, **overrides) -> ZoneStabilizer:
    return ZoneStabilizer(ZoneConfig(polygon=[list(map(float, p)) for p in polygon], **overrides))


def raw_polygon(state, polygon=POLY) -> np.ndarray:
    """Polygon implied by the (never dead-zoned) to_reference transform."""
    return transform_points(np.float32(polygon), np.linalg.inv(state.to_reference)).astype(np.float64)


def max_err(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), axis=1).max())


@pytest.fixture(autouse=True)
def _seed_opencv():
    cv2.setRNGSeed(0)


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    return textured_image(0)


# ----------------------------------------------------------------------------- motion model
def test_default_motion_model_is_similarity_and_unknown_models_are_refused():
    assert ZoneConfig().motion_model == "similarity"
    assert make().motion_model == "similarity"
    for model in ("similarity", "affine", "homography", " Homography "):
        assert make(motion_model=model).motion_model == model.strip().lower()
    with pytest.raises(ValueError, match="motion_model"):
        make(motion_model="projective")


def test_each_motion_model_recovers_its_own_transform_class(image):
    """affine recovers an anisotropic scale + shear that a similarity cannot express, the
    homography a mild perspective; the similarity is exact for rotation + shift."""
    cases = {
        "similarity": similarity(1.5, 6.0, -3.0),
        "affine": np.array([[1.03, 0.02, -8.0], [0.0, 0.975, 5.0], [0.0, 0.0, 1.0]]),
        "homography": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [4e-5, 6e-5, 1.0]]),
    }
    for model, T in cases.items():
        warped = cv2.warpPerspective(image, T, (W, H))
        expected = transform_points(np.float32(POLY), T)
        stab = make(motion_model=model)
        stab.initialize(image, 0.0)
        state = stab.update(warped, 1.0 / 30.0)
        assert state.method == "lk" and state.healthy, model
        assert max_err(raw_polygon(state), expected) <= 1.0, model
    # the anisotropic case is out of reach of the 4-DOF model (proves the model is switched)
    T = cases["affine"]
    stab = make(motion_model="similarity", confirm_frames=1)
    stab.initialize(image, 0.0)
    state = stab.update(cv2.warpPerspective(image, T, (W, H)), 1.0 / 30.0)
    assert max_err(raw_polygon(state), transform_points(np.float32(POLY), T)) > 2.0


def _train_scene():
    """Background texture only in a top and a bottom strip; a textured 'train' slides
    through the middle (where the band zone is) and JPEG artefacts mimic a CCTV stream."""
    tex = textured_image(3)
    train = textured_image(9)
    yy, xx = np.mgrid[0:H, 0:W]
    smooth = (90 + 60 * np.sin(xx / 97.0) * np.cos(yy / 71.0)).astype(np.uint8)
    base = cv2.merge([smooth] * 3)
    base[:150] = tex[:150]
    base[440:] = tex[440:]
    zone = [[0, 330], [640, 322], [640, 350], [0, 358]]  # thin band across the frame

    def frame(i: int, T: np.ndarray) -> np.ndarray:
        f = base.copy()
        f[150:300] = np.roll(train, (17 * i) % W, axis=1)[150:300]
        f = cv2.warpAffine(f, T[:2], (W, H), borderMode=cv2.BORDER_REPLICATE)
        _ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 25])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return zone, frame


def test_similarity_keeps_a_thin_zone_registered_through_a_train_passage_with_shake():
    """GAEL-20 in miniature: most of the background hidden by a moving train, the camera
    swaying, a 28 px band zone. The default model must keep every reported vertex within
    2 px of the truth (the old homography slid the clip05 band by up to 21 px)."""
    zone, frame = _train_scene()
    stab = make(polygon=zone)
    stab.initialize(frame(0, np.eye(3)), 0.0)
    worst = 0.0
    for i in range(1, 90):
        t = i / 30.0
        T = similarity(0.25 * math.sin(2 * math.pi * t / 1.7), 5.0 * math.sin(2 * math.pi * t / 2.5),
                       3.0 * math.sin(2 * math.pi * t / 3.1 + 1.0))
        state = stab.update(frame(i, T), t)
        assert state.healthy, (i, stab.health()["reason"])
        worst = max(worst, max_err(state.polygon, transform_points(np.float32(zone), T)))
    assert worst <= 2.0


# ----------------------------------------------------------------------------- dead-zone
def test_dead_zone_keeps_the_reported_polygon_still_but_not_to_reference(image):
    """A static camera wobbling by +-0.6 px (which moves every integer vertex across a
    rounding boundary) with sensor noise: the reported polygon is bit-identical every frame
    (GAEL-20: the homography jittered 2-6 px on static clips), while to_reference keeps the
    raw estimate - the debris registration must never be dead-zoned. Without the dead-zone
    the same frames move the rounded polygon every time."""
    rng = np.random.default_rng(1)
    frames = []
    for i in range(1, 25):
        dx = 0.6 if i % 2 else -0.6
        noisy = shifted(image, dx, 0.0).astype(np.float32) + rng.normal(0, 2.0, image.shape)
        frames.append((dx, np.clip(noisy, 0, 255).astype(np.uint8)))
    stab = make()  # default deadzone_px = 1
    stab.initialize(image, 0.0)
    for i, (dx, frame) in enumerate(frames, start=1):
        state = stab.update(frame, i / 30.0)
        assert state.healthy and state.method == "lk"
        assert np.array_equal(state.polygon, np.int32(POLY))
        raw = raw_polygon(state)
        assert float(np.mean(raw[:, 0] - np.float32(POLY)[:, 0])) == pytest.approx(dx, abs=0.2)
        assert max_err(raw, POLY) > 0.4
    assert stab.health()["offset_px"] == 0.0
    plain = make(deadzone_px=0.0)
    plain.initialize(image, 0.0)
    previous, changed = np.int32(POLY), 0
    for i, (_dx, frame) in enumerate(frames, start=1):
        state = plain.update(frame, i / 30.0)
        changed += int(not np.array_equal(state.polygon, previous))
        previous = state.polygon
    assert changed == len(frames)


def test_sub_dead_zone_shift_reaches_to_reference_only(image):
    stab = make(deadzone_px=1.0)
    stab.initialize(image, 0.0)
    state = stab.update(shifted(image, 0.6, 0.0), 1.0 / 30.0)
    assert state.healthy
    assert np.array_equal(state.polygon, np.int32(POLY))  # 0.6 px < dead-zone: polygon unchanged
    # to_reference maps current pixels back to the reference: x_ref = x_cur - 0.6
    assert state.to_reference[0, 2] == pytest.approx(-0.6, abs=0.15)
    assert state.to_reference[1, 2] == pytest.approx(0.0, abs=0.15)
    # a 2.5 px move is beyond the dead-zone and moves the reported polygon
    state = stab.update(shifted(image, 2.5, 0.0), 2.0 / 30.0)
    assert max_err(state.polygon, np.float32(POLY) + [2.5, 0.0]) <= 0.8
    # without a dead-zone the 0.6 px shift shows up in the (rounded) polygon at once
    plain = make(deadzone_px=0.0)
    plain.initialize(image, 0.0)
    state = plain.update(shifted(image, 0.6, 0.0), 1.0 / 30.0)
    assert not np.array_equal(state.polygon, np.int32(POLY))


# ----------------------------------------------------------------------------- temporal gate
def test_max_step_scales_with_zone_thickness():
    band = [[0, 262], [640, 265], [640, 284], [0, 284]]  # clip05: a 20 px band across 640 px
    assert make(polygon=band).max_step_px() == pytest.approx(3.0)  # max(3 px, 10% of ~20 px)
    big = [[0, 0], [400, 0], [400, 400], [0, 400]]
    assert make(polygon=big).max_step_px() == pytest.approx(0.1 * 200.0)  # 2A/P = 200 px
    assert make(polygon=band, max_step_px=0.0, max_step_fraction=0.0).max_step_px() == 0.0


def test_single_frame_jump_without_image_support_is_held(image):
    """A coherent but wrong one-frame motion (here 60% of the rows displaced: a decoder
    glitch, a train, a minority on repetitive texture) moves the estimate 12 px while the
    rest of the frame still supports the last pose: held and unhealthy, never accepted."""
    glitch = image.copy()
    glitch[: int(0.6 * H)] = shifted(image, 12, 0)[: int(0.6 * H)]
    stab = make()
    stab.initialize(image, 0.0)
    assert stab.update(image, 1.0 / 30.0).healthy
    state = stab.update(glitch, 2.0 / 30.0)
    assert state.method == "held" and state.healthy is False
    assert np.array_equal(state.polygon, np.int32(POLY))
    assert np.allclose(state.to_reference, np.eye(3), atol=0.02)  # last accepted value
    assert "unconfirmed (1/3)" in stab.health()["reason"]
    assert stab.health_changed is True
    back = stab.update(image, 3.0 / 30.0)
    assert back.healthy and back.method == "lk" and np.array_equal(back.polygon, np.int32(POLY))


def test_moving_occluder_never_moves_the_zone(image):
    """A textured strip sliding 15 px/frame over 60% of the frame (a train) must not drag
    the zone: every frame keeps the calibration polygon."""
    train = textured_image(11)
    stab = make()
    stab.initialize(image, 0.0)
    for i in range(1, 10):
        f = image.copy()
        f[: int(0.6 * H)] = np.roll(train, 15 * i, axis=1)[: int(0.6 * H)]
        state = stab.update(f, i / 30.0)
        assert np.array_equal(state.polygon, np.int32(POLY)), i


def test_genuine_bump_is_followed_at_once_when_the_frame_refutes_the_old_pose(image):
    stab = make()
    stab.initialize(image, 0.0)
    state = stab.update(shifted(image, 10, 4), 1.0 / 30.0)
    assert state.healthy and state.method == "lk"
    assert max_err(state.polygon, np.float32(POLY) + [10, 4]) <= 1.0
    assert stab.health()["offset_px"] == pytest.approx(math.hypot(10, 4), abs=1.0)


@pytest.mark.parametrize("confirm", [3, 5])
def test_bump_behind_a_static_overlay_is_followed_after_the_confirmation_frames(image, confirm):
    """A burned-in banner does not move with the camera, so after a real 10 px bump part
    of the frame still supports the old pose: the jump needs ``confirm_frames`` agreeing
    estimates, then it is followed (the zone is not left behind)."""
    banner = textured_image(7)

    def with_banner(f):
        f = f.copy()
        f[:140] = banner[:140]
        return f

    stab = make(confirm_frames=confirm)
    stab.initialize(with_banner(image), 0.0)
    bumped = with_banner(shifted(image, 10, 4))
    for i in range(1, confirm):
        state = stab.update(bumped, i / 30.0)
        assert state.method == "held" and not state.healthy, i
        assert np.array_equal(state.polygon, np.int32(POLY))
        assert f"unconfirmed ({i}/{confirm})" in stab.health()["reason"]
    state = stab.update(bumped, confirm / 30.0)
    assert state.healthy and state.method == "lk"
    assert max_err(state.polygon, np.float32(POLY) + [10, 4]) <= 1.0
    # confirm_frames = 1 turns the gate off: followed at once
    off = make(confirm_frames=1)
    off.initialize(with_banner(image), 0.0)
    assert off.update(bumped, 1.0 / 30.0).healthy


def test_a_frame_without_estimate_breaks_the_confirmation_run(image):
    banner = textured_image(7)

    def with_banner(f):
        f = f.copy()
        f[:140] = banner[:140]
        return f

    stab = make(confirm_frames=3)
    stab.initialize(with_banner(image), 0.0)
    bumped = with_banner(shifted(image, 10, 4))
    blank = np.full((H, W, 3), 128, dtype=np.uint8)
    assert not stab.update(bumped, 1 / 30.0).healthy  # 1/3
    assert not stab.update(blank, 2 / 30.0).healthy  # no estimate at all: run broken
    assert not stab.update(bumped, 3 / 30.0).healthy  # 1/3 again
    assert "unconfirmed (1/3)" in stab.health()["reason"]
    assert not stab.update(bumped, 4 / 30.0).healthy  # 2/3
    assert stab.update(bumped, 5 / 30.0).healthy  # 3/3


def uneven_exposure(img: np.ndarray) -> np.ndarray:
    """Gain varying across the frame (0.2-1.0): defeats LK, histogram-matched LK included."""
    gain = 0.6 + 0.4 * np.sin(np.linspace(0.0, 12.0, img.shape[1], dtype=np.float32))
    return np.clip(img.astype(np.float32) * gain[None, :, None], 0, 255).astype(np.uint8)


def test_orb_estimates_need_an_inlier_fraction_and_pass_the_same_gate(image):
    """ORB (the fallback when LK fails, here on a frame with uneven exposure) must be
    supported by orb_min_inlier_fraction of its matches, and its jumps go through the
    temporal gate."""
    H_true = similarity(2.0, 7.0, -4.0)
    uneven = uneven_exposure(cv2.warpPerspective(image, H_true, (W, H)))
    gray = cv2.cvtColor(uneven, cv2.COLOR_BGR2GRAY)
    picky = make(orb_min_inlier_fraction=0.999)
    picky.initialize(image, 0.0)
    est, reason = picky._orb_estimate(picky._keyframe, gray)
    assert est is None and "agree" in reason
    held = picky.update(uneven, 1.0 / 30.0)
    assert held.method == "held" and not held.healthy

    # An unevenly exposed frame (LK fails) whose top half is displaced 12 px: ORB's best
    # fit is a jump, but many matches still fit the last pose, so the frame does not
    # refute it and the ORB jump is held exactly like an LK one (GAEL-M2 / F11).
    glitch = uneven_exposure(image)
    glitch[: H // 2] = uneven_exposure(shifted(image, 12, 0))[: H // 2]
    stab = make()
    stab.initialize(image, 0.0)
    est, _reason = stab._orb_estimate(stab._keyframe, cv2.cvtColor(glitch, cv2.COLOR_BGR2GRAY))
    assert est is not None
    assert max_err(transform_points(np.float32(POLY), est.pose), POLY) > stab.max_step_px()  # ORB alone would jump
    state = stab.update(glitch, 1.0 / 30.0)
    assert state.method == "held" and not state.healthy
    assert stab.health()["reason"].startswith("ORB step")
    assert np.array_equal(state.polygon, np.int32(POLY))
    # the same glitch at half exposure: histogram-matched LK finds the 12 px jump and the
    # gate holds it all the same
    dark = cv2.convertScaleAbs(image, alpha=0.5)
    dark[: H // 2] = cv2.convertScaleAbs(shifted(image, 12, 0), alpha=0.5)[: H // 2]
    stab = make()
    stab.initialize(image, 0.0)
    state = stab.update(dark, 1.0 / 30.0)
    assert state.method == "held" and stab.health()["reason"].startswith("LK step")
    assert np.array_equal(state.polygon, np.int32(POLY))


def test_temporal_gate_can_be_disabled(image):
    glitch = image.copy()
    glitch[: int(0.6 * H)] = shifted(image, 12, 0)[: int(0.6 * H)]
    for overrides in ({"confirm_frames": 1}, {"max_step_px": 0.0, "max_step_fraction": 0.0}):
        stab = make(**overrides)
        stab.initialize(image, 0.0)
        assert stab.update(glitch, 1.0 / 30.0).healthy, overrides


# ----------------------------------------------------------------------------- to_reference
def test_to_reference_is_the_inverse_of_the_camera_motion_and_held_when_unhealthy(image):
    T = similarity(1.0, 6.0, -5.0)
    warped = cv2.warpPerspective(image, T, (W, H))
    stab = make()
    state = stab.initialize(image, 0.0)
    assert np.array_equal(state.to_reference, np.eye(3)) and state.to_reference.dtype == np.float64
    state = stab.update(warped, 1.0 / 30.0)
    assert state.to_reference.shape == (3, 3) and state.to_reference.dtype == np.float64
    # current-frame pixels -> reference pixels
    pts_ref = np.float32([[100, 100], [500, 120], [320, 400]])
    pts_cur = transform_points(pts_ref, T)
    assert np.allclose(transform_points(pts_cur, state.to_reference), pts_ref, atol=0.5)
    good = state.to_reference.copy()
    held = stab.update(np.full((H, W, 3), 128, np.uint8), 2.0 / 30.0)
    assert held.method == "held" and held.homography is None
    assert np.array_equal(held.to_reference, good)


def test_to_reference_is_composed_across_keyframe_re_anchors(image):
    """F05 CORRECTED: after a keyframe re-anchor the transform must still map to the
    ORIGINAL reference (the frame the debris background was learned on), not to the new
    keyframe; and the re-anchor is verified against a direct calibration estimate."""
    T1 = similarity(1.0, 6.0, -4.0)
    T2 = similarity(-0.5, 5.0, 3.0) @ T1
    first = cv2.warpPerspective(image, T1, (W, H))
    second = cv2.warpPerspective(image, T2, (W, H))
    stab = make(keyframe_refresh_static_seconds=0.2)
    stab.initialize(image, 0.0)
    t = 0.0
    while stab.health()["reanchors"] < 1:
        t += 0.1
        stab.update(first, t, scene_static=True)
        assert t < 2.0
    assert stab.health()["unverified_reanchors"] == 0
    assert max_err(stab.reference_polygon, transform_points(np.float32(POLY), T1)) <= 1.0
    state = stab.update(second, t + 0.1, scene_static=False)
    assert state.healthy
    pts_ref = np.float32([[100, 100], [500, 120], [320, 400]])
    assert np.allclose(transform_points(transform_points(pts_ref, T2), state.to_reference), pts_ref, atol=0.6)
    # relative to the keyframe (state.homography) the motion is only the second step
    assert np.allclose(state.homography[:2, 2], (T2 @ np.linalg.inv(T1))[:2, 2], atol=0.6)
    # a second re-anchor (keyframe != calibration frame) is checked with a direct estimate
    t += 0.1
    while stab.health()["reanchors"] < 2:
        t += 0.1
        stab.update(second, t, scene_static=True)
        assert t < 4.0
    health = stab.health()
    assert health["unverified_reanchors"] == 0 and health["drift_px"] is not None and health["drift_px"] < 1.0
    state = stab.update(second, t + 0.1)
    assert np.allclose(transform_points(transform_points(pts_ref, T2), state.to_reference), pts_ref, atol=0.6)


def _biased_fit(stab: ZoneStabilizer, bias_px: float):
    """Make every fit 0.4 px off in x: the systematic per-estimate error that chained
    re-anchors would accumulate into a random walk (F19)."""
    original = stab._fit

    def fit(src, dst):
        M, inliers, mask = original(src, dst)
        if M is not None:
            M = M.copy()
            M[0, 2] += bias_px
        return M, inliers, mask

    stab._fit = fit  # type: ignore[assignment]


def test_verified_re_anchors_do_not_random_walk(image):
    stab = make(keyframe_refresh_static_seconds=0.1, deadzone_px=0.0)
    stab.initialize(image, 0.0)
    _biased_fit(stab, 0.4)
    for i in range(1, 121):
        stab.update(image, i / 30.0, scene_static=True)
    health = stab.health()
    assert health["reanchors"] >= 20
    assert health["unverified_reanchors"] == 0 and health["refused_reanchors"] == 0
    assert health["drift_px"] == pytest.approx(0.4, abs=0.1)
    # bounded by one chained step plus one direct estimate, however many re-anchors
    assert health["offset_px"] <= 1.0


def test_unverifiable_re_anchors_are_counted_and_the_drift_shows(image):
    """When no direct estimate against the calibration frame is possible the chained pose
    is used, and health() says how many re-anchors are unverified."""
    stab = make(keyframe_refresh_static_seconds=0.1, deadzone_px=0.0)
    stab.initialize(image, 0.0)
    _biased_fit(stab, 0.4)
    original = stab._estimate

    def no_direct(kf, gray):
        if kf is stab._reference and stab._keyframe is not stab._reference:
            return None, 0, "direct estimate unavailable"
        return original(kf, gray)

    stab._estimate = no_direct  # type: ignore[assignment]
    for i in range(1, 61):
        stab.update(image, i / 30.0, scene_static=True)
    health = stab.health()
    assert health["reanchors"] >= 10
    assert health["unverified_reanchors"] == health["reanchors"] - 1  # the first one was direct
    assert health["offset_px"] >= 0.4 * health["reanchors"] - 0.5  # the chain has drifted


def test_re_anchor_that_disagrees_with_the_calibration_frame_is_refused(image):
    stab = make(keyframe_refresh_static_seconds=0.1, reanchor_max_drift_px=1.0)
    stab.initialize(image, 0.0)
    t = 0.0
    while stab.health()["reanchors"] < 1:  # the first re-anchor is direct by construction
        t += 1.0 / 30.0
        stab.update(image, t, scene_static=True)
    keyframe = stab._keyframe
    original = stab._estimate

    def wrong_direct(kf, gray):
        est, n, reason = original(kf, gray)
        if est is not None and kf is stab._reference:
            est.pose = similarity(0.0, 5.0, 0.0) @ est.pose
        return est, n, reason

    stab._estimate = wrong_direct  # type: ignore[assignment]
    for _ in range(6):
        t += 1.0 / 30.0
        state = stab.update(image, t, scene_static=True)
        assert state.healthy and np.array_equal(state.polygon, np.int32(POLY))
    health = stab.health()
    assert health["refused_reanchors"] >= 1 and health["drift_px"] == pytest.approx(5.0, abs=0.3)
    assert stab._keyframe is keyframe


# ----------------------------------------------------------------------------- start-up
@pytest.mark.parametrize("first", ["black", "grey", "dark_noise"])
def test_featureless_first_frame_stays_uninitialised_and_retries(image, first):
    rng = np.random.default_rng(2)
    blank = {
        "black": np.zeros((H, W, 3), np.uint8),
        "grey": np.full((H, W, 3), 128, np.uint8),
        "dark_noise": np.clip(rng.normal(8, 1.2, (H, W, 3)), 0, 255).astype(np.uint8),
    }[first]
    stab = make()
    state = stab.update(blank, 0.0)
    assert state.method == "uninitialized" and state.healthy is False
    assert state.to_reference is None and state.homography is None
    assert np.array_equal(state.polygon, np.int32(POLY)) and state.mask.shape == (H, W) and state.area_px > 0
    assert not stab.initialized and stab.health()["initialized"] is False
    assert stab.health_changed is True  # reported once ...
    assert "featureless" in stab.health()["reason"]
    state = stab.update(blank, 1.0 / 30.0)
    assert state.method == "uninitialized" and stab.health_changed is False  # ... not every frame
    # the first textured frame becomes the reference
    state = stab.update(image, 2.0 / 30.0)
    assert state.method == "identity" and state.healthy and stab.initialized
    assert stab.health_changed is True and np.array_equal(state.to_reference, np.eye(3))
    state = stab.update(shifted(image, 4, 2), 3.0 / 30.0)
    assert state.method == "lk" and state.healthy


def test_initialize_on_a_featureless_frame_defers_to_the_next_update(image):
    stab = make()
    assert stab.initialize(np.zeros((H, W, 3), np.uint8), 0.0).method == "uninitialized"
    assert stab.update(image, 1.0 / 30.0).method == "identity"
    with pytest.raises(ValueError):
        stab.update(cv2.resize(image, (320, 240)), 2.0 / 30.0)


def test_frame_size_is_pinned_while_uninitialised(image):
    stab = make()
    stab.update(np.zeros((H, W, 3), np.uint8), 0.0)
    with pytest.raises(ValueError):
        stab.update(np.zeros((240, 320, 3), np.uint8), 1.0 / 30.0)


def test_stabilize_false_initialises_on_anything(image):
    stab = make(stabilize=False)
    state = stab.update(np.zeros((H, W, 3), np.uint8), 0.0)
    assert state.method == "identity" and state.healthy and np.array_equal(state.to_reference, np.eye(3))
    state = stab.update(shifted(image, 9, 9), 1.0 / 30.0)
    assert state.method == "identity" and np.array_equal(state.polygon, np.int32(POLY))
