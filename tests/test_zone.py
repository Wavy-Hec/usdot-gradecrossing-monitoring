"""Tests for gradecrossing.zone.ZoneStabilizer on synthetic imagery.

Every image is generated from a seeded RNG so the tests are deterministic. The textured
image is seeded noise blurred with a 3x3 Gaussian plus high-contrast rectangles and
circles, which gives Shi-Tomasi hundreds of strong corners (like a real crossing with
ballast, sleepers and road markings).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from gradecrossing.config import ZoneConfig
from gradecrossing.geometry import point_in_polygon, transform_points
from gradecrossing.zone import ZoneStabilizer

W, H = 640, 480
POLY = [[150, 300], [450, 280], [520, 400], [120, 420]]


def textured_image(seed: int = 0) -> np.ndarray:
    """640x480 BGR image with rich, corner-heavy texture."""
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


def warp_matrix(angle_deg: float, tx: float, ty: float) -> np.ndarray:
    """3x3 homography: rotation about the image centre followed by a translation."""
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64)


def uneven_exposure(img: np.ndarray) -> np.ndarray:
    """Exposure varying across the frame (gain 0.2-1.0 in two waves: cloud shadows, a
    headlight). LK's brightness constancy fails even after a global histogram match;
    ORB's local binary tests do not care."""
    gain = 0.6 + 0.4 * np.sin(np.linspace(0.0, 12.0, img.shape[1], dtype=np.float32))
    return np.clip(img.astype(np.float32) * gain[None, :, None], 0, 255).astype(np.uint8)


def rms_error(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    a = np.asarray(poly_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(poly_b, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def make_stabilizer(**overrides) -> ZoneStabilizer:
    cfg = ZoneConfig(polygon=[list(p) for p in POLY], **overrides)
    return ZoneStabilizer(cfg)


@pytest.fixture(autouse=True)
def _seed_opencv():
    cv2.setRNGSeed(0)


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    return textured_image(0)


@pytest.fixture(scope="module")
def warped_pair(image):
    """(H_true, warped image) for a 2 degree rotation and a (7,-4) px shift."""
    H_true = warp_matrix(2.0, 7.0, -4.0)
    return H_true, cv2.warpPerspective(image, H_true, (W, H))


# ----------------------------------------------------------------------------- initialize
def test_initialize_returns_identity_state(image):
    stab = make_stabilizer()
    state = stab.initialize(image, 0.0)
    assert state.method == "identity"
    assert state.healthy is True
    assert state.homography is None
    assert state.inliers == 0
    assert state.polygon.dtype == np.int32 and state.polygon.shape == (4, 2)
    assert np.array_equal(state.polygon, np.int32(POLY))
    assert state.mask.shape == (H, W) and state.mask.dtype == np.uint8
    assert state.area_px == cv2.countNonZero(state.mask) > 0
    assert stab.reference_features.shape[0] > 100  # richly textured image gives many corners
    health = stab.health()
    # The original keys are kept; motion model, hold reason, offset and re-anchor drift
    # bookkeeping were added with the similarity model / temporal gate / to_reference.
    assert {"method", "inliers", "healthy", "keyframe_age_s", "consecutive_unhealthy"} <= set(health)
    assert {"initialized", "motion_model", "reason", "offset_px", "reanchors", "unverified_reanchors",
            "refused_reanchors", "drift_px"} <= set(health)
    assert health["method"] == "identity" and health["healthy"] is True
    assert health["initialized"] is True and health["motion_model"] == "similarity"
    assert health["offset_px"] == 0.0 and health["reanchors"] == 0 and health["drift_px"] is None
    assert np.array_equal(state.to_reference, np.eye(3))


def test_features_are_excluded_from_zone(image):
    stab = make_stabilizer(exclude_features_in_zone=True)
    stab.initialize(image, 0.0)
    inside = [point_in_polygon((x, y), np.float32(POLY)) for x, y in stab.reference_features]
    assert not any(inside)
    # and with exclusion off some corners do land inside (the polygon covers textured area)
    stab2 = make_stabilizer(exclude_features_in_zone=False)
    stab2.initialize(image, 0.0)
    inside2 = [point_in_polygon((x, y), np.float32(POLY)) for x, y in stab2.reference_features]
    assert any(inside2)


# ----------------------------------------------------------------------------- LK path
def test_lk_tracks_small_rotation_and_shift(image, warped_pair):
    H_true, warped = warped_pair
    stab = make_stabilizer()
    stab.initialize(image, 0.0)
    state = stab.update(warped, 1.0 / 30.0)
    expected = transform_points(np.float32(POLY), H_true)
    assert state.method == "lk"
    assert state.healthy is True
    assert state.inliers >= stab.cfg.min_inliers
    assert rms_error(state.polygon, expected) <= 1.5
    assert state.homography is not None and state.homography.shape == (3, 3)
    assert np.allclose(state.homography, H_true, atol=0.2)
    assert state.mask.shape == (H, W)
    assert state.area_px == cv2.countNonZero(state.mask)
    assert stab.health()["consecutive_unhealthy"] == 0
    assert stab.health()["method"] == "lk"
    # the reference polygon is untouched by a normal update
    assert np.allclose(stab.reference_polygon, np.float32(POLY))


def test_update_before_initialize_uses_first_frame_as_keyframe(image, warped_pair):
    H_true, warped = warped_pair
    stab = make_stabilizer()
    first = stab.update(image, 0.0)
    assert first.method == "identity" and first.healthy
    second = stab.update(warped, 1.0 / 30.0)
    assert second.method == "lk"
    assert rms_error(second.polygon, transform_points(np.float32(POLY), H_true)) <= 1.5


def test_min_inlier_fraction_rejects_a_minority_agreement(image, warped_pair):
    """LK can lock a minority of corners onto self-similar sleeper/ballast texture, all
    sliding by the same wrong amount, and still clear the absolute min_inliers. The
    estimate must also be supported by a *fraction* of the reference features."""
    H_true, warped = warped_pair
    stab = make_stabilizer(min_inliers=30, min_inlier_fraction=0.2)
    stab.initialize(image, 0.0)
    assert stab.reference_features.shape[0] == 400
    assert stab.min_lk_inliers() == 80  # max(30, 0.2 * 400)

    # End to end through update(), with the RANSAC fit faked to return the true (plausible)
    # transform backed by only 45 of the 400 features: refused, and held.
    stab._fit = lambda src, dst: (H_true, 45, None)  # type: ignore[assignment]
    stab.cfg.orb_fallback = False
    held = stab.update(warped, 1.0 / 30.0)
    assert held.method == "held" and held.healthy is False and held.inliers == 45
    assert "45 inliers < 80" in stab.health()["reason"]
    # ... and the same transform with 120 inliers is accepted.
    stab._fit = lambda src, dst: (H_true, 120, None)  # type: ignore[assignment]
    good = stab.update(warped, 2.0 / 30.0)
    assert good.method == "lk" and good.healthy is True
    assert rms_error(good.polygon, transform_points(np.float32(POLY), H_true)) <= 1.5

    # min_inlier_fraction = 0 restores the absolute-only rule (min_inliers alone).
    plain = make_stabilizer(min_inliers=30, min_inlier_fraction=0.0, orb_fallback=False)
    plain.initialize(image, 0.0)
    assert plain.min_lk_inliers() == 30
    plain._fit = lambda src, dst: (H_true, 45, None)  # type: ignore[assignment]
    assert plain.update(warped, 1.0 / 30.0).method == "lk"


def test_min_inlier_fraction_default_still_accepts_a_real_track(image, warped_pair):
    """Sanity: a genuine small camera move keeps most features, so the default 0.2 never
    stands in the way of a good estimate."""
    H_true, warped = warped_pair
    stab = make_stabilizer()
    assert stab.cfg.min_inlier_fraction == 0.2
    stab.initialize(image, 0.0)
    state = stab.update(warped, 1.0 / 30.0)
    assert state.method == "lk" and state.healthy
    assert state.inliers >= stab.min_lk_inliers()


# ----------------------------------------------------------------------------- hold
def test_uniform_frame_holds_last_healthy_polygon(image, warped_pair):
    H_true, warped = warped_pair
    stab = make_stabilizer()
    stab.initialize(image, 0.0)
    good = stab.update(warped, 1.0 / 30.0)
    assert good.healthy and good.method == "lk"

    blank = np.full((H, W, 3), 128, dtype=np.uint8)
    held = stab.update(blank, 2.0 / 30.0)
    assert held.method == "held"
    assert held.healthy is False
    assert held.homography is None
    # The polygon is the last *healthy* one (the warped estimate), NOT the raw reference.
    assert np.array_equal(held.polygon, good.polygon)
    assert not np.array_equal(held.polygon, np.int32(POLY))
    assert stab.health()["consecutive_unhealthy"] == 1
    assert stab.health()["healthy"] is False
    assert stab.health_changed is True

    held2 = stab.update(blank, 3.0 / 30.0)
    assert held2.method == "held"
    assert stab.health()["consecutive_unhealthy"] == 2
    assert stab.health_changed is False  # still unhealthy: no transition

    # recovering restores healthy and resets the unhealthy counter
    back = stab.update(warped, 4.0 / 30.0)
    assert back.healthy and back.method == "lk"
    assert stab.health()["consecutive_unhealthy"] == 0
    assert stab.health_changed is True


def test_hold_when_orb_fallback_disabled(image, warped_pair):
    _, warped = warped_pair
    stab = make_stabilizer(orb_fallback=False)
    stab.initialize(image, 0.0)
    stab.update(warped, 1.0 / 30.0)
    # exposure varying across the frame: LK fails (a global brightness change no longer
    # does - see the histogram-matched LK test), so only ORB could follow
    state = stab.update(uneven_exposure(warped), 2.0 / 30.0)
    assert state.method == "held"
    assert state.healthy is False


# ----------------------------------------------------------------------------- keyframe refresh
def test_keyframe_refresh_after_static_period(image, warped_pair):
    H_true, warped = warped_pair
    expected = transform_points(np.float32(POLY), H_true)
    stab = make_stabilizer(keyframe_refresh_static_seconds=2.0)
    stab.initialize(image, 0.0)

    # static frames for 1.75 s: not long enough, keyframe stays and its age grows
    t = 0.0
    for _ in range(7):
        t += 0.25
        state = stab.update(warped, t, scene_static=True)
        assert state.healthy and state.method == "lk"
        assert np.allclose(stab.reference_polygon, np.float32(POLY))
    assert stab.health()["keyframe_age_s"] == pytest.approx(1.75)

    # one non-static frame resets the static timer
    t += 0.25
    stab.update(warped, t, scene_static=False)
    assert np.allclose(stab.reference_polygon, np.float32(POLY))

    # static again: the streak restarts at t=2.25, so no refresh until t >= 4.25
    refreshed_at = None
    while t < 6.0:
        t += 0.25
        stab.update(warped, t, scene_static=True)
        if not np.allclose(stab.reference_polygon, np.float32(POLY)):
            refreshed_at = t
            break
    assert refreshed_at is not None
    assert refreshed_at >= 4.25 - 1e-9
    assert refreshed_at <= 4.5 + 1e-9
    # the new reference polygon is the warped estimate and the keyframe age reset
    assert rms_error(stab.reference_polygon, expected) <= 1.5
    assert stab.health()["keyframe_age_s"] == pytest.approx(0.0)
    assert stab.reference_features.shape[0] > 100

    # subsequent updates on the same frame are identity relative to the new keyframe
    state = stab.update(warped, t + 0.25, scene_static=True)
    assert state.healthy and state.method == "lk"
    assert rms_error(state.polygon, expected) <= 1.5
    assert np.allclose(state.homography, np.eye(3), atol=0.05)
    assert stab.health()["keyframe_age_s"] == pytest.approx(0.25)


def test_keyframe_refresh_disabled_when_zero(image, warped_pair):
    _, warped = warped_pair
    stab = make_stabilizer(keyframe_refresh_static_seconds=0.0)
    stab.initialize(image, 0.0)
    for i in range(1, 12):
        stab.update(warped, i * 1.0, scene_static=True)
    assert np.allclose(stab.reference_polygon, np.float32(POLY))
    assert stab.health()["keyframe_age_s"] == pytest.approx(11.0)


def test_keyframe_refresh_requires_healthy_estimate(image, warped_pair):
    _, warped = warped_pair
    stab = make_stabilizer(keyframe_refresh_static_seconds=1.0)
    stab.initialize(image, 0.0)
    blank = np.full((H, W, 3), 128, dtype=np.uint8)
    for i in range(1, 8):
        state = stab.update(blank, i * 0.5, scene_static=True)
        assert state.method == "held"
    # never re-anchored onto the blank frame
    assert np.allclose(stab.reference_polygon, np.float32(POLY))
    assert stab.reference_features.shape[0] > 100


# ----------------------------------------------------------------------------- ORB fallback
def test_global_exposure_change_is_tracked_by_histogram_matched_lk(image, warped_pair):
    """Halving the gain breaks LK's brightness constancy (it used to take ORB, ~1 px less
    precise and several times slower); LK on the frame remapped to the keyframe's
    histogram recovers it with LK precision."""
    H_true, warped = warped_pair
    dark = cv2.convertScaleAbs(warped, alpha=0.5, beta=0)
    stab = make_stabilizer()
    stab.initialize(image, 0.0)
    # sanity: plain LK really fails on this frame
    lk_est, lk_inliers, _reason = stab._lk_estimate(stab._keyframe, cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY))
    assert lk_est is None and lk_inliers < stab.cfg.min_inliers
    state = stab.update(dark, 1.0 / 30.0)
    assert state.method == "lk" and state.healthy is True
    assert rms_error(state.polygon, transform_points(np.float32(POLY), H_true)) <= 1.0
    # the remap is skipped when the exposures already agree
    assert stab._photometric_match(stab._keyframe, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) is None


def test_orb_fallback_on_uneven_exposure_change(image, warped_pair):
    """Exposure varying across the frame defeats LK even after the histogram match, but
    ORB descriptors survive."""
    H_true, warped = warped_pair
    uneven = uneven_exposure(warped)
    stab = make_stabilizer()
    stab.initialize(image, 0.0)
    # sanity: the LK paths alone really are unhealthy on this frame
    gray = cv2.cvtColor(uneven, cv2.COLOR_BGR2GRAY)
    lk_est, _lk_inliers, _reason = stab._lk_estimate(stab._keyframe, gray)
    assert lk_est is None
    matched = stab._photometric_match(stab._keyframe, gray)
    assert matched is None or stab._lk_estimate(stab._keyframe, matched)[0] is None

    state = stab.update(uneven, 1.0 / 30.0)
    expected = transform_points(np.float32(POLY), H_true)
    assert state.method == "orb"
    assert state.healthy is True
    assert state.inliers >= stab.cfg.orb_min_matches
    assert rms_error(state.polygon, expected) <= 2.0
    assert stab.health()["method"] == "orb"


def test_orb_helper_directly(image, warped_pair):
    H_true, warped = warped_pair
    stab = make_stabilizer()
    stab.initialize(image, 0.0)
    est, reason = stab._orb_estimate(stab._keyframe, cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    assert est is not None and reason == ""
    H_orb = est.M
    assert est.src.shape[0] >= est.inliers >= stab.cfg.orb_min_matches
    assert est.inliers >= stab.cfg.orb_min_inlier_fraction * est.src.shape[0]
    assert np.allclose(H_orb, H_true, atol=0.3)
    expected = transform_points(np.float32(POLY), H_true)
    assert rms_error(transform_points(np.float32(POLY), H_orb), expected) <= 2.0


def test_large_jump_beyond_lk_range_falls_back_to_orb(image):
    """A 120 px camera bump exceeds LK's pyramid capture range; ORB matching is global."""
    H_jump = warp_matrix(0.0, 120.0, 5.0)
    jumped = cv2.warpPerspective(image, H_jump, (W, H))
    stab = make_stabilizer()
    stab.initialize(image, 0.0)
    # sanity: LK alone cannot follow a jump this large
    assert stab._lk_estimate(stab._keyframe, cv2.cvtColor(jumped, cv2.COLOR_BGR2GRAY))[0] is None
    state = stab.update(jumped, 1.0 / 30.0)
    assert state.healthy is True
    assert state.method == "orb"
    assert state.inliers >= stab.cfg.orb_min_matches
    assert rms_error(state.polygon, transform_points(np.float32(POLY), H_jump)) <= 2.0


# ----------------------------------------------------------------------------- guards
def test_frame_size_mismatch_raises(image):
    stab = make_stabilizer()
    stab.initialize(image, 0.0)
    small = cv2.resize(image, (320, 240))
    with pytest.raises(ValueError):
        stab.update(small, 1.0 / 30.0)


def test_stabilize_disabled_returns_identity(image, warped_pair):
    _, warped = warped_pair
    stab = make_stabilizer(stabilize=False)
    stab.initialize(image, 0.0)
    state = stab.update(warped, 1.0 / 30.0)
    assert state.method == "identity"
    assert state.healthy is True
    assert state.homography is None
    assert np.array_equal(state.polygon, np.int32(POLY))


def test_sanity_check_rejects_implausible_homography(image):
    """A 2x zoom is outside max_scale_change, so even a perfect track must be refused."""
    H_zoom = np.array([[2.0, 0.0, -W / 2.0], [0.0, 2.0, -H / 2.0], [0.0, 0.0, 1.0]])
    zoomed = cv2.warpPerspective(image, H_zoom, (W, H))
    stab = make_stabilizer()
    stab.initialize(image, 0.0)
    state = stab.update(zoomed, 1.0 / 30.0)
    assert state.method == "held"
    assert state.healthy is False
    assert np.array_equal(state.polygon, np.int32(POLY))


def test_accepts_grayscale_frames(image, warped_pair):
    H_true, warped = warped_pair
    stab = make_stabilizer()
    stab.initialize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0.0)
    state = stab.update(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), 1.0 / 30.0)
    assert state.method == "lk"
    assert rms_error(state.polygon, transform_points(np.float32(POLY), H_true)) <= 1.5
