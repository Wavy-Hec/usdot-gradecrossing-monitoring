# Pipeline Helper Functions (legacy prototype used by notebooks/MOG2_YOLO_Detection.ipynb)
#
# The maintained implementation is the `gradecrossing/` package. These helpers are kept so
# the notebook and the stress-test write-up still run, with the two safety bugs found in the
# audit fixed in a backward-compatible way (see docs/AUDIT.md):
#   * check_vehicle_in_zoi() now only counts road-vehicle classes and tests whether the
#     bottom-center of the box (the ground contact point) is inside the polygon, instead of
#     "any pixel of any box of any class overlaps the zone".
#   * every detected box of any class is still subtracted from the MOG2 mask (exclusion).
import cv2
import numpy as np

# Class names (case-insensitive) that count as road vehicles for the vehicle-in-ZOI check.
# Covers COCO (yolo11n.pt) and the fine-tuned CCTV_Vehicles names. Trains and people are
# deliberately NOT vehicles: a train in the zone is normal, and a pedestrian needs its own rule.
DEFAULT_VEHICLE_CLASSES = frozenset(
    ["car", "truck", "bus", "motorcycle", "motorbike", "jeep", "van", "tricycle", "bicycle"]
)


def stabilize_zoi(gray, first_gray, first_points, lk_params, zoi_points):
    """Track ZOI from frame 1 to current frame using optical flow + RANSAC.

    Known limitations (fixed in gradecrossing.zone.ZoneStabilizer): anchored to frame 1
    forever, no inlier-count or geometry sanity check, and a silent fallback to the
    unstabilised polygon when the homography fails.
    """
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        first_gray, gray, first_points, None, **lk_params
    )
    good_first = first_points[status == 1]
    good_curr  = curr_points[status == 1]
    if len(good_first) >= 4:
        H, _ = cv2.findHomography(good_first, good_curr, cv2.RANSAC, 3.0)
        if H is not None:
            zoi_reshaped = zoi_points.astype(np.float32).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(zoi_reshaped, H).reshape(-1, 2).astype(np.int32)
    return zoi_points


def build_zoi_mask(frame_shape, zoi_stabilized):
    """Create a filled polygon mask for the ZOI."""
    zoi_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(zoi_mask, [zoi_stabilized], 255)
    return zoi_mask


def apply_mog2_mask(frame, mog2, learning_rate=-1):
    """Apply MOG2 and return a cleaned foreground mask.

    ``learning_rate`` is passed straight to ``mog2.apply``: -1 (default) keeps OpenCV's
    automatic rate, which absorbs a stationary object into the background in about
    ln(0.9)/ln(1-1/history) frames (~32 frames = 1 s for history=300); pass 0 to freeze the
    background while monitoring. The package does this automatically.
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = mog2.apply(frame, learningRate=learning_rate)
    fg_mask[fg_mask == 127] = 0
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask


def _polygon_from_mask(zoi_mask):
    """Recover the ZOI polygon from its mask (largest external contour)."""
    contours, _ = cv2.findContours(zoi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def box_in_zoi(x1, y1, x2, y2, zoi_polygon):
    """True if the bottom-center of the box (its ground contact point) is inside the polygon."""
    contour = np.asarray(zoi_polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, ((x1 + x2) / 2.0, float(y2)), False) >= 0


def check_vehicle_in_zoi(yolo_results, zoi_mask, frame_shape, zoi_polygon=None,
                         vehicle_classes=DEFAULT_VEHICLE_CLASSES):
    """Check whether any road *vehicle* is inside the ZOI.

    Returns: vehicle_in_zoi (bool), vehicle_labels (list), exclusion_mask (array).

    * Only boxes whose class is in ``vehicle_classes`` (case-insensitive) can set
      ``vehicle_in_zoi``. Trains, people, birds, signs etc. never do.
    * "Inside" means the bottom-center of the box is inside ``zoi_polygon`` (recovered from
      ``zoi_mask`` when not given). A box whose corner merely overlaps the polygon in image
      space, e.g. a car on the adjacent road, is not inside.
    * ``exclusion_mask`` still covers every detected box of every class so that known
      objects are removed from the MOG2 debris mask.
    """
    vehicle_in_zoi = False
    vehicle_labels = []
    exclusion_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if zoi_polygon is None:
        zoi_polygon = _polygon_from_mask(zoi_mask)
    wanted = {c.lower() for c in vehicle_classes}

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            cv2.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, -1)
            if label.lower() not in wanted or zoi_polygon is None:
                continue
            if box_in_zoi(x1, y1, x2, y2, zoi_polygon):
                vehicle_in_zoi = True
                vehicle_labels.append(label)

    return vehicle_in_zoi, vehicle_labels, exclusion_mask


def get_debris_mask(fg_mask, exclusion_mask, zoi_mask):
    """Subtract YOLO exclusion mask from MOG2 mask, isolate inside ZOI."""
    clean_mask  = cv2.bitwise_and(fg_mask, cv2.bitwise_not(exclusion_mask))
    debris_mask = cv2.bitwise_and(clean_mask, zoi_mask)
    return debris_mask


def evaluate_status(debris_time, vehicle_time, vehicle_in_zoi,
                    TIME_THRESHOLD, VEHICLE_THRESHOLD):
    """Return status text and color based on timers and vehicle presence.

    This is the single source of truth for the notebook's status: WARNING while a vehicle
    is inside the ZOI, ALARM once it has been there for VEHICLE_THRESHOLD seconds or debris
    has persisted for TIME_THRESHOLD seconds. Note the prototype has no tracking, so
    ``vehicle_time`` measures "some vehicle has been in the zone continuously", not "one
    vehicle has been stationary"; the package fixes this per tracked object.
    """
    if debris_time >= TIME_THRESHOLD:
        return "ALARM",   (0, 0, 255)
    elif vehicle_time >= VEHICLE_THRESHOLD:
        return "ALARM",   (0, 0, 255)
    elif vehicle_in_zoi:
        return "WARNING", (0, 165, 255)
    else:
        return "CLEAR",   (0, 255, 0)


def draw_frame(frame, zoi_stabilized, yolo_results, zoi_mask,
               status_text, status_color, frame_count, current_time,
               debris_count, vehicle_in_zoi, vehicle_time,
               vehicle_classes=DEFAULT_VEHICLE_CLASSES):
    """Draw ZOI, YOLO boxes, and status overlay on the frame."""
    frame_drawn = frame.copy()
    cv2.polylines(frame_drawn, [zoi_stabilized], isClosed=True, color=(0, 255, 0), thickness=5)
    wanted = {c.lower() for c in vehicle_classes}

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            inside = label.lower() in wanted and box_in_zoi(x1, y1, x2, y2, zoi_stabilized)
            color = (0, 165, 255) if inside else (160, 160, 160)
            cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), color, 3 if inside else 2)
            cv2.putText(frame_drawn, f"{label}{' IN ZOI' if inside else ''}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame_drawn, f"STATUS: {status_text}",
                (50, 80),  cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 4)
    cv2.putText(frame_drawn, f"Frame: {frame_count} | Time: {current_time:.1f}s | Debris px: {debris_count}",
                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_drawn, f"Vehicle in ZOI: {vehicle_in_zoi} | Vehicle time: {vehicle_time:.1f}s",
                (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    return frame_drawn
