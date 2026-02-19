from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Traffic Engine Data Driven")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBAL MEMORY
# =========================
traffic_counts = {1: {"N": 0, "S": 0, "E": 0, "W": 0}}
signal_state = {1: {
    "counts": {"N": 0, "S": 0, "E": 0, "W": 0},
    "active_lane": None,
    "phase": "ALL_RED",
    "remaining_time": 0,
    "vehicle_count": 0
}}

phase_end_time = {1: 0}
current_lane = {1: None}
cycle_served = {1: set()}

MIN_GREEN = 20
MAX_GREEN = 90
YELLOW_TIME = 3
ALL_RED_TIME = 3


class BulkUpdate(BaseModel):
    intersection_id: int
    counts: Dict[str, int]


def calculate_green_time(vehicle_count):
    green = 25 + vehicle_count * 0.8
    return int(max(MIN_GREEN, min(MAX_GREEN, green)))


@app.post("/traffic/bulk_update")
def bulk_update(data: BulkUpdate):

    intersection_id = data.intersection_id
    counts = data.counts
    now = time.time()

    traffic_counts[intersection_id] = counts

    phase = signal_state[intersection_id]["phase"]

    # =========================
    # IF PHASE STILL ACTIVE
    # =========================
    if now < phase_end_time[intersection_id]:

        remaining = int(phase_end_time[intersection_id] - now)

        signal_state[intersection_id] = {
            "counts": counts,
            "active_lane": current_lane[intersection_id],
            "phase": phase,
            "remaining_time": remaining,
            "vehicle_count": counts.get(current_lane[intersection_id], 0)
        }

        return signal_state[intersection_id]

    # =========================
    # TRANSITIONS
    # =========================

    # GREEN → YELLOW
    if phase == "GREEN":
        phase = "YELLOW"
        phase_end_time[intersection_id] = now + YELLOW_TIME

    # YELLOW → ALL_RED
    elif phase == "YELLOW":
        phase = "ALL_RED"
        phase_end_time[intersection_id] = now + ALL_RED_TIME

    # ALL_RED → NEW GREEN
    else:

        # Cycle enforcement
        if len(cycle_served[intersection_id]) == 4:
            cycle_served[intersection_id] = set()

        eligible = {
            k: v for k, v in counts.items()
            if k not in cycle_served[intersection_id]
        }

        if not eligible:
            eligible = counts

        lane = max(eligible, key=eligible.get)
        vehicle_count = counts[lane]
        green_time = calculate_green_time(vehicle_count)

        current_lane[intersection_id] = lane
        cycle_served[intersection_id].add(lane)

        phase = "GREEN"
        phase_end_time[intersection_id] = now + green_time

    # =========================
    # UPDATE STATE
    # =========================

    remaining = int(phase_end_time[intersection_id] - now)

    signal_state[intersection_id] = {
        "counts": counts,
        "active_lane": current_lane[intersection_id],
        "phase": phase,
        "remaining_time": remaining,
        "vehicle_count": counts.get(current_lane[intersection_id], 0)
    }

    return signal_state[intersection_id]


@app.get("/traffic/status/{intersection_id}")
def get_status(intersection_id: int):
    return signal_state.get(
        intersection_id,
        {
            "counts": {"N": 0, "S": 0, "E": 0, "W": 0},
            "active_lane": None,
            "phase": "ALL_RED",
            "remaining_time": 0,
            "vehicle_count": 0
        }
    )
