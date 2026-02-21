from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Smart Traffic Control System")

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONSTANTS
# =========================
LANES = ["N", "E", "S", "W"]

MIN_GREEN = 20
MAX_GREEN = 90
YELLOW_TIME = 3
ALL_RED_TIME = 3

# =========================
# MEMORY STORAGE
# =========================
traffic_counts: Dict[int, Dict[str, int]] = {}
signal_state: Dict[int, Dict] = {}
phase_end_time: Dict[int, float] = {}
current_lane: Dict[int, str] = {}
control_mode: Dict[int, str] = {}
manual_times_store: Dict[int, Dict[str, int]] = {}
manual_index: Dict[int, int] = {}
cycle_served: Dict[int, set] = {}
pending_mode_switch: Dict[int, bool] = {}
emergency_request: Dict[int, Dict] = {}

# =========================
# INIT
# =========================
def ensure_intersection(i: int):
    if i not in traffic_counts:
        traffic_counts[i] = {l: 0 for l in LANES}
        signal_state[i] = {
            "counts": traffic_counts[i],
            "active_lane": None,
            "phase": "ALL_RED",
            "remaining_time": 0,
            "vehicle_count": 0,
        }
        phase_end_time[i] = 0
        current_lane[i] = None
        control_mode[i] = "AUTO"
        manual_times_store[i] = {l: 30 for l in LANES}
        manual_index[i] = -1
        cycle_served[i] = set()
        pending_mode_switch[i] = False
        emergency_request[i] = None


def calculate_green_time(vehicle_count: int):
    green = 25 + vehicle_count * 0.8
    return int(max(MIN_GREEN, min(MAX_GREEN, green)))


# =========================
# CONTROL MODE SWITCH
# =========================
class ControlPayload(BaseModel):
    intersection_id: int
    mode: str
    manual_times: Dict[str, int] = {}


@app.post("/traffic/control")
def control_mode_switch(data: ControlPayload):
    ensure_intersection(data.intersection_id)

    now = time.time()

    # Trigger yellow before switching
    if current_lane[data.intersection_id]:
        signal_state[data.intersection_id]["phase"] = "YELLOW"
        phase_end_time[data.intersection_id] = now + YELLOW_TIME
        pending_mode_switch[data.intersection_id] = True
        print("Switching mode → Yellow phase")

    control_mode[data.intersection_id] = data.mode.upper()

    if data.mode.upper() == "MANUAL":
        manual_times_store[data.intersection_id] = data.manual_times
        manual_index[data.intersection_id] = -1
        cycle_served[data.intersection_id] = set()
        print("Mode changed to MANUAL")

    elif data.mode.upper() == "AUTO":
        cycle_served[data.intersection_id] = set()
        print("Mode changed to AUTO")

    return {"status": "ok"}


# =========================
# EMERGENCY OVERRIDE
# =========================
class EmergencyPayload(BaseModel):
    intersection_id: int
    lane: str
    green_time: int


@app.post("/traffic/emergency")
def emergency_override(data: EmergencyPayload):
    ensure_intersection(data.intersection_id)

    now = time.time()

    if current_lane[data.intersection_id]:
        signal_state[data.intersection_id]["phase"] = "YELLOW"
        phase_end_time[data.intersection_id] = now + YELLOW_TIME
        pending_mode_switch[data.intersection_id] = True

    emergency_request[data.intersection_id] = {
        "lane": data.lane,
        "duration": max(5, min(180, data.green_time))
    }

    control_mode[data.intersection_id] = "EMERGENCY"
    print(f"Emergency override requested → {data.lane}")

    return {"status": "ok"}


# =========================
# BULK UPDATE (CORE ENGINE)
# =========================
class BulkUpdate(BaseModel):
    intersection_id: int
    counts: Dict[str, int]


@app.post("/traffic/bulk_update")
def bulk_update(data: BulkUpdate):

    ensure_intersection(data.intersection_id)
    now = time.time()

    traffic_counts[data.intersection_id] = data.counts
    remaining = phase_end_time[data.intersection_id] - now
    phase = signal_state[data.intersection_id]["phase"]

    # =========================
    # ACTIVE PHASE
    # =========================
    if remaining > 0:

        if phase == "GREEN" and remaining <= YELLOW_TIME:
            phase = "YELLOW"

        signal_state[data.intersection_id] = {
            "counts": data.counts,
            "active_lane": current_lane[data.intersection_id],
            "phase": phase,
            "remaining_time": int(remaining),
            "vehicle_count": data.counts.get(current_lane[data.intersection_id], 0)
        }

        print(f"{phase} | Lane {current_lane[data.intersection_id]} | {int(remaining)}s")
        return signal_state[data.intersection_id]

    # =========================
    # TRANSITIONS
    # =========================
    if phase == "GREEN":
        signal_state[data.intersection_id]["phase"] = "YELLOW"
        phase_end_time[data.intersection_id] = now + YELLOW_TIME
        return signal_state[data.intersection_id]

    if phase == "YELLOW":
        signal_state[data.intersection_id]["phase"] = "ALL_RED"
        phase_end_time[data.intersection_id] = now + ALL_RED_TIME
        return signal_state[data.intersection_id]

    # =========================
    # NEW GREEN PHASE
    # =========================
    mode = control_mode[data.intersection_id]

    # EMERGENCY
    if mode == "EMERGENCY" and emergency_request[data.intersection_id]:
        lane = emergency_request[data.intersection_id]["lane"]
        duration = emergency_request[data.intersection_id]["duration"]
        emergency_request[data.intersection_id] = None

    # AUTO
    elif mode == "AUTO":
        if len(cycle_served[data.intersection_id]) == 4:
            cycle_served[data.intersection_id] = set()

        eligible = {
            k: v for k, v in data.counts.items()
            if k not in cycle_served[data.intersection_id]
        }

        if not eligible:
            eligible = data.counts

        lane = max(eligible, key=eligible.get)
        duration = calculate_green_time(data.counts[lane])
        cycle_served[data.intersection_id].add(lane)

    # MANUAL
    else:
        manual_index[data.intersection_id] = (
            manual_index[data.intersection_id] + 1
        ) % 4

        lane = LANES[manual_index[data.intersection_id]]
        duration = manual_times_store[data.intersection_id].get(lane, 30)

    current_lane[data.intersection_id] = lane
    phase_end_time[data.intersection_id] = now + duration

    signal_state[data.intersection_id] = {
        "counts": data.counts,
        "active_lane": lane,
        "phase": "GREEN",
        "remaining_time": duration,
        "vehicle_count": data.counts.get(lane, 0)
    }

    print(f"SWITCH → {lane} for {duration}s")

    return signal_state[data.intersection_id]


# =========================
# STATUS API
# =========================
@app.get("/traffic/status/{intersection_id}")
def get_status(intersection_id: int):
    ensure_intersection(intersection_id)

    now = time.time()
    remaining = max(0, int(phase_end_time[intersection_id] - now))
    signal_state[intersection_id]["remaining_time"] = remaining

    return signal_state[intersection_id]
