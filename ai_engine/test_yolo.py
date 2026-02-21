import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================
MODEL_NAME = "yolov8x.pt"
IMAGE_SIZE = 1280
CONFIDENCE = 0.15
VIDEO_PATH = "../video/traffic2.mp4"
VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

# ==========================================
# DEVICE & MODEL
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = YOLO(MODEL_NAME)
model.to(device)

# ==========================================
# LOAD VIDEO
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get original resolution
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Original Resolution:", orig_width, "x", orig_height)

# ==========================================
# DRAW POLYGON INTERACTIVELY
# ==========================================
polygon_points = []
drawing_done = False

def draw_polygon(event, x, y, flags, param):
    global polygon_points, drawing_done
    if drawing_done:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing_done = True

cv2.namedWindow("Draw Polygon", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Draw Polygon", draw_polygon)

print("Draw polygon points with left click. Right click to finish.")

ret, frame = cap.read()
if not ret:
    print("Error reading video.")
    exit()

while not drawing_done:
    temp_frame = frame.copy()

    for pt in polygon_points:
        cv2.circle(temp_frame, tuple(pt), 5, (0, 0, 255), -1)

    if len(polygon_points) > 1:
        cv2.polylines(
            temp_frame,
            [np.array(polygon_points, np.int32)],
            isClosed=False,
            color=(0, 255, 0),
            thickness=2
        )

    cv2.imshow("Draw Polygon", temp_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("Polygon drawing canceled.")
        exit()

cv2.destroyWindow("Draw Polygon")
print(f"Polygon points: {polygon_points}")

# ==========================================
# VEHICLE DETECTION LOOP
# ==========================================
drag_point = None

def mouse_event(event, x, y, flags, param):
    global drag_point, polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(polygon_points):
            if abs(x - pt[0]) < 10 and abs(y - pt[1]) < 10:
                drag_point = i
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_point is not None:
            polygon_points[drag_point] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        drag_point = None

cv2.namedWindow("Vehicle Detection & Counting", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Vehicle Detection & Counting", mouse_event)

print("Vehicle Detection Started...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    polygon = np.array(polygon_points, np.int32)
    cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)

    results = model.predict(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=device,
        half=True,
        verbose=False
    )

    vehicle_count = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                vehicle_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    cv2.putText(frame, f"Count: {vehicle_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Vehicle Detection & Counting", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
