import cv2
import torch
import time
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================
INTERVAL = 2
IMAGE_SIZE = 1280     # Good balance
CONFIDENCE = 0.30

MIN_GREEN = 20
MAX_GREEN = 90

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# LOAD MODEL (Use l instead of x)
# =========================
model = YOLO("yolov8l.pt")   # Better speed/accuracy balance
model.to(device)

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

# =========================
# LOAD VIDEOS
# =========================
videos = {
    "N": cv2.VideoCapture("../video/traffic1.mp4"),
    "S": cv2.VideoCapture("../video/traffic2.mp4"),
    "E": cv2.VideoCapture("../video/traffic3.mp4"),
    "W": cv2.VideoCapture("../video/traffic4.mp4")
}

def calculate_green_time(vehicle_count):
    green = 25 + vehicle_count * 0.8
    return int(max(MIN_GREEN, min(MAX_GREEN, green)))

print("\n===== FAST 2-SECOND TRAFFIC MODE =====\n")

# =========================
# MAIN LOOP
# =========================
while True:

    cycle_start = time.time()
    approach_counts = {}

    for direction, cap in videos.items():

        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        results = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=device,
            half=True,          # Faster on GPU
            verbose=False
        )

        count = 0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name in VEHICLE_CLASSES:
                    count += 1

        approach_counts[direction] = count

    active_lane = max(approach_counts, key=approach_counts.get)
    vehicle_count = approach_counts[active_lane]
    green_time = calculate_green_time(vehicle_count)

    print("\n========== SIGNAL DECISION ==========")
    print("Lane Counts:", approach_counts)
    print("Active Lane:", active_lane)
    print("Vehicle Count:", vehicle_count)
    print("Green Time:", green_time, "seconds")
    print("=====================================\n")

    elapsed = time.time() - cycle_start
    if elapsed < INTERVAL:
        time.sleep(INTERVAL - elapsed)
