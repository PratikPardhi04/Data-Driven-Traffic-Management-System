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
INTERVAL = 2.0  # seconds between sending counts

MODEL_NAME = "yolov8x.pt"
IMAGE_SIZE = 1280
CONFIDENCE = 0.25
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

# ==========================================
# DEVICE
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================
# LOAD MODEL
# ==========================================
model = YOLO(MODEL_NAME)
model.to(device)

# ==========================================
# LOAD VIDEOS
# ==========================================
videos = {
    "N": cv2.VideoCapture("../video/traffic1.mp4"),
    "S": cv2.VideoCapture("../video/traffic2.mp4"),
    "E": cv2.VideoCapture("../video/traffic3.mp4"),
    "W": cv2.VideoCapture("../video/traffic4.mp4")
}

last_frames = {lane: None for lane in videos.keys()}
frame_indices = {lane: 0 for lane in videos.keys()}
video_fps = {lane: videos[lane].get(cv2.CAP_PROP_FPS) or 25 for lane in videos.keys()}
video_total_frames = {lane: int(videos[lane].get(cv2.CAP_PROP_FRAME_COUNT)) for lane in videos.keys()}

# ==========================================
# UPDATED POLYGONS (YOUR NEW VALUES)
# ==========================================
polygons = {
    "N": [[489, 719], [1274, 718], [687, 278], [606, 280], [485, 715]],

    "S": [[326, 717], [544, 349], [600, 308], [612, 285],
          [705, 284], [769, 380], [862, 601], [914, 718], [326, 717]],

    "E": [[132, 719], [1279, 716], [1278, 368], [855, 136],
          [760, 65], [566, 58], [348, 376], [131, 716]],

    "W": [[466, 718], [1031, 719], [841, 335], [748, 220],
          [623, 218], [554, 371], [507, 553], [467, 714], [465, 718]]
}

# ==========================================
# INITIAL SCAN
# ==========================================
print("Initial scan of all lanes...")
for lane, cap in videos.items():
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    last_frames[lane] = frame
    frame_indices[lane] = 0
print("Initial scan done.\n")

# ==========================================
# DETECTION FUNCTION
# ==========================================
def detect_vehicles(frame, polygon):
    polygon_np = np.array(polygon, np.int32)

    results = model.predict(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=device,
        half=True,
        verbose=False
    )

    count = 0
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])
            if model.names[cls] not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cv2.pointPolygonTest(polygon_np, (cx, cy), False) >= 0:
                count += 1

    return count

# ==========================================
# MAIN LOOP
# ==========================================
next_run_time = time.time()

while True:
    now = time.time()
    if now < next_run_time:
        time.sleep(next_run_time - now)

    # GET SIGNAL STATUS
    try:
        response = requests.get(
            f"{BACKEND_URL}/traffic/status/{INTERSECTION_ID}",
            timeout=2
        )
        signal = response.json()
        green_lane = signal.get("active_lane")
        phase = signal.get("phase")
        remaining_time = signal.get("remaining_time", 0)
    except:
        green_lane = None
        phase = "ALL_RED"
        remaining_time = 0

    pre_green_scan = remaining_time <= 3
    approach_counts = {}

    for lane, cap in videos.items():

        process_lane = (
            (lane == green_lane and phase == "GREEN")
            or pre_green_scan
        )

        if process_lane:
            frame_indices[lane] += int(INTERVAL * video_fps[lane])

            if frame_indices[lane] >= video_total_frames[lane]:
                frame_indices[lane] = (
                    frame_indices[lane] % video_total_frames[lane]
                )

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[lane])
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            last_frames[lane] = frame
        else:
            frame = last_frames[lane]

        count = detect_vehicles(frame, polygons[lane])
        approach_counts[lane] = count

        video_sec = frame_indices[lane] / video_fps[lane]
        print(f"Lane {lane} - Time: {video_sec:.2f}s - Vehicles: {count}")

    # SEND COUNTS TO BACKEND
    try:
        requests.post(
            f"{BACKEND_URL}/traffic/bulk_update",
            json={
                "intersection_id": INTERSECTION_ID,
                "counts": approach_counts
            },
            timeout=3
        )
    except Exception as e:
        print("Backend error:", e)

    print(f"""
========== SIGNAL SYNC ==========
Green Lane : {green_lane}
Phase      : {phase}
Remaining  : {remaining_time}s
Counts     : {approach_counts}
=================================
""")

    next_run_time += INTERVAL
    