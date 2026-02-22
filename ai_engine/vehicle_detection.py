import requests
import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================
BACKEND_URL = "http://127.0.0.1:8000"
INTERSECTION_ID = 1
INTERVAL = 2.0           # wall-clock seconds between cycles
GREEN_VIDEO_STEP = 4.0   # seconds of video the green lane advances per cycle
PRE_GREEN_THRESHOLD = 3  # seconds remaining before triggering a full re-scan

MODEL_NAME = "yolov8x.pt"
IMAGE_SIZE = 1280
CONFIDENCE = 0.15
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

FAILURE_THRESHOLD  = 3
RECOVERY_THRESHOLD = 2

# ==========================================
# DEVICE + MODEL
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model = YOLO(MODEL_NAME)
model.to(device)

# ==========================================
# VIDEOS
# ==========================================
videos = {
    "N": cv2.VideoCapture("../video/traffic1.mp4"),
    "S": cv2.VideoCapture("../video/traffic2.mp4"),
    "E": cv2.VideoCapture("../video/traffic3.mp4"),
    "W": cv2.VideoCapture("../video/traffic4.mp4")
}

last_frames        = {lane: None for lane in videos}
last_counts        = {lane: 0    for lane in videos}
frame_indices      = {lane: 0    for lane in videos}
video_fps          = {lane: videos[lane].get(cv2.CAP_PROP_FPS) or 25 for lane in videos}
video_total_frames = {lane: int(videos[lane].get(cv2.CAP_PROP_FRAME_COUNT)) for lane in videos}

# ==========================================
# CAMERA HEALTH
# ==========================================
consecutive_failures  = {lane: 0     for lane in videos}
consecutive_successes = {lane: 0     for lane in videos}
camera_is_down        = {lane: False for lane in videos}

# ==========================================
# SIGNAL STATE  (from bulk_update response)
# ==========================================
green_lane     = None
phase          = "ALL_RED"
remaining_time = 0
current_mode   = "AUTO"

# ==========================================
# SCAN STATE — explicit, not inferred from remaining_time
# ==========================================
# After a full scan we switch to green-only mode.
# We switch back to full-scan when:
#   (a) backend tells us remaining_time <= PRE_GREEN_THRESHOLD, OR
#   (b) green lane changes (new phase started)
# This means the script works correctly even when backend is temporarily down.

in_green_only_mode = False   # True = green lane advancing, reds frozen
pre_green_fired    = False   # True = we already fired the pre-green full scan
                             #        for the current green phase; don't repeat it

# ==========================================
# POLYGONS
# ==========================================
polygons = {
    "N": [[606, 286], [442, 708], [1244, 709], [994, 471], [682, 265], [608, 288]],
    "S": [[606, 302], [488, 390], [242, 712], [1029, 708], [744, 337], [709, 284], [621, 289], [605, 302]],
    "E": [[718, 20], [8, 225], [14, 1061], [1902, 1062], [1900, 547], [1132, 20], [718, 22]],
    "W": [[618, 204], [438, 708], [1027, 707], [974, 522], [742, 205], [619, 203]]
}

# ==========================================
# INITIAL SCAN — frame 0 of all videos
# ==========================================
print("Initial scan of all lanes...")
for lane, cap in videos.items():
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    last_frames[lane] = frame
    frame_indices[lane] = 0
print("Initial scan done.\n")


# ==========================================
# HELPERS
# ==========================================
def report_camera_status(lane: str, is_working: bool):
    try:
        requests.post(
            f"{BACKEND_URL}/traffic/camera_alert",
            json={"intersection_id": INTERSECTION_ID, "lane": lane, "is_working": is_working},
            timeout=2
        )
        print(f"[CAMERA ALERT] Lane {lane} -> {'RECOVERED' if is_working else 'DOWN'}")
    except Exception as e:
        print(f"[CAMERA ALERT] Failed to notify: {e}")


def read_frame_with_health_check(lane: str, cap: cv2.VideoCapture):
    if not cap.isOpened():
        read_ok, frame = False, None
    else:
        ret, frame = cap.read()
        read_ok = ret and frame is not None

    if not read_ok:
        consecutive_failures[lane] += 1
        consecutive_successes[lane] = 0
        if consecutive_failures[lane] >= FAILURE_THRESHOLD and not camera_is_down[lane]:
            camera_is_down[lane] = True
            report_camera_status(lane, is_working=False)
    else:
        consecutive_successes[lane] += 1
        consecutive_failures[lane] = 0
        if consecutive_successes[lane] >= RECOVERY_THRESHOLD and camera_is_down[lane]:
            camera_is_down[lane] = False
            report_camera_status(lane, is_working=True)

    return read_ok, frame


def advance_and_read(lane: str, cap: cv2.VideoCapture, step_secs: float):
    """Advance video by step_secs worth of frames, read + health-check."""
    frame_indices[lane] += int(step_secs * video_fps[lane])
    if frame_indices[lane] >= video_total_frames[lane]:
        frame_indices[lane] %= video_total_frames[lane]

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[lane])
    read_ok, frame = read_frame_with_health_check(lane, cap)

    if not read_ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret2, frame2 = cap.read()
        frame = frame2 if (ret2 and frame2 is not None) else last_frames[lane]
        if ret2 and frame2 is not None:
            frame_indices[lane] = 0

    if frame is not None:
        last_frames[lane] = frame

    return frame


def scan_lane(lane: str, step_secs: float) -> int:
    """Advance video, run YOLO, cache and return count."""
    frame = advance_and_read(lane, videos[lane], step_secs)
    count = detect_vehicles(frame, polygons[lane]) if frame is not None else last_counts[lane]
    last_counts[lane] = count
    return count


def detect_vehicles(frame, polygon) -> int:
    polygon_np = np.array(polygon, np.int32)
    results = model.predict(frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                            device=device, half=True, verbose=False)
    count = 0
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            if model.names[cls] not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cv2.pointPolygonTest(polygon_np, (cx, cy), False) >= 0:
                count += 1
    return count


# ==========================================
# MAIN LOOP
# ==========================================
first_update_done  = False
next_run_time      = time.time()
prev_green_lane    = None   # to detect when green lane changes

while True:
    now = time.time()
    if now < next_run_time:
        time.sleep(next_run_time - now)

    approach_counts = {}

    # ── Detect green lane change → new phase started → reset scan state ──
    green_lane_changed = (green_lane != prev_green_lane) and (green_lane is not None)
    if green_lane_changed:
        in_green_only_mode = True
        pre_green_fired    = False   # reset so we fire pre-green for the new phase
        prev_green_lane    = green_lane

    # ── Decide this cycle's scan mode ────────────────────────────────────
    #
    # FULL SCAN (all 4 lanes, 2s video step) when:
    #   1. First cycle ever                           → bootstrap
    #   2. Backend not yet contacted successfully     → no signal info yet
    #   3. remaining_time ≤ PRE_GREEN_THRESHOLD AND   → pre-green scan
    #      we haven't fired it yet for this phase
    #   4. SEMI-AUTO / MANUAL mode                    → need all counts
    #
    # GREEN-ONLY (green advances 4s, reds frozen) otherwise.
    #
    is_pre_green = (
        first_update_done                      # have signal info
        and remaining_time <= PRE_GREEN_THRESHOLD
        and remaining_time > 0                 # not just "backend down = 0"
        and not pre_green_fired
        and phase == "GREEN"                   # only meaningful during green phase
    )

    do_full_scan = (
        not first_update_done
        or is_pre_green
        or current_mode in ("SEMI-AUTO", "MANUAL")
        or not in_green_only_mode
        or green_lane is None
    )

    if is_pre_green:
        pre_green_fired = True   # don't fire again until next green phase

    # ── Execute scans ─────────────────────────────────────────────────────
    for lane in videos:
        if do_full_scan:
            # All lanes: advance 2s, run YOLO
            count = scan_lane(lane, step_secs=INTERVAL)
            ts    = frame_indices[lane] / video_fps[lane]
            tag   = f" [DOWN]" if camera_is_down[lane] else f" [SCAN  @ {ts:.1f}s]"

        elif lane == green_lane:
            # Green lane only: advance 4s, run YOLO
            count = scan_lane(lane, step_secs=GREEN_VIDEO_STEP)
            ts    = frame_indices[lane] / video_fps[lane]
            tag   = f" [GREEN @ {ts:.1f}s]"

        else:
            # Red lane: frozen — no video advance, no YOLO
            count = last_counts[lane]
            ts    = frame_indices[lane] / video_fps[lane]
            tag   = f" [HELD  @ {ts:.1f}s]"

        approach_counts[lane] = count
        print(f"Lane {lane}{tag} - Vehicles: {count}")

    # ── Send to backend, read signal state from response ─────────────────
    max_video_time = max(frame_indices[l] / video_fps[l] for l in videos)
    try:
        resp = requests.post(
            f"{BACKEND_URL}/traffic/bulk_update",
            json={
                "intersection_id": INTERSECTION_ID,
                "counts":          approach_counts,
                "video_time":      max_video_time,
                "camera_health":   {l: not camera_is_down[l] for l in videos}
            },
            timeout=3
        )
        resp.raise_for_status()
        data = resp.json()

        sig            = data.get("signal", {})
        new_green      = sig.get("active_lane")
        phase          = sig.get("phase", "ALL_RED")
        remaining_time = sig.get("remaining_time", 0)
        current_mode   = sig.get("mode", "AUTO")
        first_update_done = True

        # Apply green lane update — change detected at top of next cycle
        green_lane = new_green

    except Exception as e:
        print(f"Backend error: {e}")
        # Don't touch remaining_time — keep it as-is so pre_green logic
        # doesn't falsely trigger. We stay in whatever scan mode we're in.

    # ── Console summary ───────────────────────────────────────────────────
    down_lanes = [l for l in videos if camera_is_down[l]]
    if down_lanes:
        print(f"\nCAMERA FAILURE - Lanes: {', '.join(down_lanes)} | Mode: {current_mode}\n")

    if not first_update_done:
        scan_label = "ALL (waiting for backend)"
    elif do_full_scan and is_pre_green:
        scan_label = f"ALL (pre-green, {remaining_time}s left)"
    elif do_full_scan:
        scan_label = "ALL (bootstrap / semi / manual)"
    else:
        scan_label = f"GREEN={green_lane} only  |  reds frozen"

    print(f"""
========== SIGNAL SYNC ==========
Mode        : {current_mode}
Green Lane  : {green_lane}
Phase       : {phase}
Remaining   : {remaining_time}s
This cycle  : {scan_label}
Counts      : {approach_counts}
Camera Down : {down_lanes if down_lanes else 'None'}
=================================
""")

    next_run_time += INTERVAL
