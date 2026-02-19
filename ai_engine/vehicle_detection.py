import requests
import cv2
import torch
import time
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================
BACKEND_URL = "http://127.0.0.1:8000"
INTERSECTION_ID = 1
INTERVAL = 2.0

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

print("Vehicle Detection Started...\n")

next_run_time = time.time()

while True:

    now = time.time()
    if now < next_run_time:
        time.sleep(next_run_time - now)

    # ======================================
    # GET SIGNAL STATE
    # ======================================
    try:
        response = requests.get(
            f"{BACKEND_URL}/traffic/status/{INTERSECTION_ID}",
            timeout=2
        )
        signal = response.json()
        green_lane = signal.get("active_lane")
        phase = signal.get("phase")
    except:
        green_lane = None
        phase = "ALL_RED"

    approach_counts = {}

    # ======================================
    # PROCESS LANES
    # ======================================
    for direction, cap in videos.items():

        if direction == green_lane and phase == "GREEN":
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            last_frames[direction] = frame
        else:
            frame = last_frames[direction]

            if frame is None:
                ret, frame = cap.read()
                last_frames[direction] = frame

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
                if model.names[cls] in VEHICLE_CLASSES:
                    count += 1

        approach_counts[direction] = count

    # ======================================
    # SEND COUNTS
    # ======================================
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

    # ======================================
    # TERMINAL LOG
    # ======================================
    print("\n========== SIGNAL SYNC ==========")
    print("Phase:", phase)
    print("Green Lane:", green_lane)
    print("Counts:", approach_counts)
    print("=================================\n")

    next_run_time += INTERVAL
