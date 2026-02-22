import time
from typing import Dict, List
from learning_storage import LearningStorage

LANES = ["N", "E", "S", "W"]


class FailureManager:

    def __init__(self):

        self.storage = LearningStorage()

        self.camera_status: Dict[int, Dict[str, bool]] = {}

        # VIDEO HEARTBEAT STORAGE
        self.last_video_seen: Dict[int, Dict[str, float]] = {}

        # timeout seconds
        self.failure_timeout = 4

        self.learning_data: Dict[int, Dict[str, List[Dict]]] = self.storage.load()

    # =========================
    # INIT
    # =========================
    def ensure_intersection(self, intersection_id: int):

        if intersection_id not in self.camera_status:
            self.camera_status[intersection_id] = {
                l: True for l in LANES
            }

        if intersection_id not in self.last_video_seen:
            self.last_video_seen[intersection_id] = {}

        if intersection_id not in self.learning_data:
            self.learning_data[intersection_id] = {
                l: [] for l in LANES
            }

    # =========================
    # VIDEO HEARTBEAT
    # =========================
    def update_video_heartbeat(
        self,
        intersection_id: int,
        lane: str,
        video_detected: bool
    ):

        self.ensure_intersection(intersection_id)

        now = time.time()

        if video_detected:

            # camera alive
            self.last_video_seen[intersection_id][lane] = now
            self.camera_status[intersection_id][lane] = True

        else:

            last_seen = self.last_video_seen[intersection_id].get(lane)

            if last_seen is None:
                self.last_video_seen[intersection_id][lane] = now
                return

            # FAIL only after timeout
            if now - last_seen > self.failure_timeout:
                self.camera_status[intersection_id][lane] = False

    # =========================
    # CAMERA STATUS
    # =========================
    def get_working_lanes(self, intersection_id: int):

        self.ensure_intersection(intersection_id)

        return [

            lane

            for lane, status in
            self.camera_status[intersection_id].items()

            if status

        ]

    def all_cameras_failed(self, intersection_id: int):

        return len(

            self.get_working_lanes(intersection_id)

        ) == 0

    # =========================
    # STORE LEARNING DATA
    # =========================
    def store_learning_data(
        self,
        intersection_id,
        lane,
        video_time,
        vehicle_count,
        green_time
    ):

        self.ensure_intersection(intersection_id)

        record = {

            "timestamp": time.time(),
            "video_time": video_time,
            "vehicle_count": vehicle_count,
            "green_time": green_time

        }

        self.learning_data[intersection_id][lane].append(record)

        if len(self.learning_data[intersection_id][lane]) > 100:
            self.learning_data[intersection_id][lane].pop(0)

        self.storage.save(self.learning_data)

    # =========================
    # FALLBACK LEARNING
    # =========================
    def get_fallback_time(
        self,
        intersection_id,
        lane,
        vehicle_count,
        manual_value
    ):

        self.ensure_intersection(intersection_id)

        records = self.learning_data[intersection_id][lane]

        if not records:
            return manual_value

        closest = min(

            records,

            key=lambda r:

            abs(r["vehicle_count"] - vehicle_count)

        )

        return closest["green_time"]