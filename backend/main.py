from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, List, Optional
import time
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from failure_manager import FailureManager

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

LANES = ["N", "S", "E", "W"]

MIN_GREEN = 20
MAX_GREEN = 90
DEFAULT_FALLBACK_GREEN = 30  # used for dead-camera lanes in MANUAL fallback

YELLOW_TIME = 3
ALL_RED_TIME = 3

SCAN_WINDOW = 8

failure_manager = FailureManager()

traffic_counts: Dict[int, Dict[str, int]] = {}
display_counts: Dict[int, Dict[str, int]] = {}
scan_start_time: Dict[int, float] = {}

signal_state: Dict[int, Dict] = {}
phase_end_time: Dict[int, float] = {}
control_mode: Dict[int, str] = {}

manual_times_store: Dict[int, Dict[str, int]] = {}
manual_index: Dict[int, int] = {}
cycle_served: Dict[int, set] = {}

emergency_active: Dict[int, bool] = {}
emergency_data: Dict[int, Dict] = {}

# Tracks if MANUAL mode was triggered by camera failure (not by operator)
camera_failure_mode: Dict[int, bool] = {}

# Once cameras recover, switch back to AUTO after the current signal finishes
pending_auto_restore: Dict[int, bool] = {}

# Tracks which lanes are currently flagged as camera-down (for dashboard notifications)
camera_down_lanes: Dict[int, set] = {}

# Tracks lanes manually reported as offline from the frontend toggle
frontend_camera_offline: Dict[int, set] = {}

dashboard_clients: List[WebSocket] = []


# =========================
# HEATMAP
# =========================
def generate_heatmap(counts):
    if not counts:
        return {}
    mx = max(counts.values())
    if mx == 0:
        mx = 1
    return {l: round(c / mx, 2) for l, c in counts.items()}


# =========================
# DASHBOARD WS
# =========================
@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    dashboard_clients.append(websocket)
    try:
        while True:
            await websocket.send_text("ping")
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        dashboard_clients.remove(websocket)


async def notify_dashboard(intersection_id: int):
    state = signal_state[intersection_id]
    down = list(camera_down_lanes.get(intersection_id, set()))

    payload = {
        "intersection_id": intersection_id,
        "counts": traffic_counts[intersection_id],
        "heatmap": generate_heatmap(traffic_counts[intersection_id]),
        "active_lane": state["active_lane"],
        "phase": state["phase"],
        "remaining_time": state["remaining_time"],
        "vehicle_count": state["vehicle_count"],
        "mode": state["mode"],
        # NEW: camera health info baked into every update
        "camera_down_lanes": down,
        "has_camera_failure": len(down) > 0,
    }

    for ws in dashboard_clients.copy():
        try:
            await ws.send_json(payload)
        except Exception:
            dashboard_clients.remove(ws)


async def push_camera_alert(intersection_id: int, lane: str, is_working: bool):
    """Push a real-time camera failure/recovery notification to the dashboard."""
    down = list(camera_down_lanes.get(intersection_id, set()))
    alert_type = "camera_recovery" if is_working else "camera_failure"

    notification = {
        "type": "alert",
        "alert_type": alert_type,
        "intersection_id": intersection_id,
        "lane": lane,
        "down_lanes": down,
        "message": (
            f"✅ Camera {lane} recovered at Intersection {intersection_id}."
            if is_working
            else f"⚠️ Camera {lane} is NOT RESPONDING at Intersection {intersection_id}!"
        ),
        "severity": "info" if is_working else "critical",
        "timestamp": time.time(),
        # Also send updated mode so dashboard can update badge immediately
        "current_mode": control_mode.get(intersection_id, "AUTO"),
    }

    for ws in dashboard_clients.copy():
        try:
            await ws.send_json(notification)
        except Exception:
            dashboard_clients.remove(ws)


# =========================
# INIT
# =========================
def ensure_intersection(i: int):
    if i not in traffic_counts:
        traffic_counts[i] = {l: 0 for l in LANES}
        display_counts[i] = {l: 0 for l in LANES}
        scan_start_time[i] = time.time()

        signal_state[i] = {
            "counts": traffic_counts[i],
            "active_lane": None,
            "phase": "INIT",
            "remaining_time": 0,
            "vehicle_count": 0,
            "mode": "AUTO",
            "system_ready": False,
        }

        phase_end_time[i] = 0
        control_mode[i] = "AUTO"
        manual_times_store[i] = {l: DEFAULT_FALLBACK_GREEN for l in LANES}
        manual_index[i] = -1
        cycle_served[i] = set()
        emergency_active[i] = False
        emergency_data[i] = {}
        camera_failure_mode[i] = False
        pending_auto_restore[i] = False
        camera_down_lanes[i] = set()
        frontend_camera_offline[i] = set()

        failure_manager.ensure_intersection(i)


# =========================
# GREEN TIME
# =========================
def calculate_green_time(vehicle_count):
    g = 25 + vehicle_count * 0.8
    return int(max(MIN_GREEN, min(MAX_GREEN, g)))


# =========================
# SEMI-AUTO MODE
# Working lanes → dynamic green time based on vehicle count
# Dead camera lanes → fixed DEFAULT_FALLBACK_GREEN, still get a rotation slot
# =========================
def run_semi_auto(intersection_id: int, now: float, counts: dict, working_lanes: list):
    """
    Like AUTO but dead-camera lanes get a fixed green time instead of
    count-based time, so no lane ever starves and no decision is made
    on stale/fake data.
    """
    if len(cycle_served[intersection_id]) == 4:
        cycle_served[intersection_id] = set()

    # All unserved lanes are eligible
    unserved = [l for l in LANES if l not in cycle_served[intersection_id]]
    if not unserved:
        unserved = LANES[:]

    # Split into live (can use counts) vs dead (fixed time)
    live_unserved = [l for l in unserved if l in working_lanes]
    dead_unserved = [l for l in unserved if l not in working_lanes]

    if live_unserved:
        # Pick the live lane with the most vehicles
        lane = max(live_unserved, key=lambda l: counts.get(l, 0))
        duration = calculate_green_time(counts.get(lane, 0))
    elif dead_unserved:
        # All remaining unserved lanes are dead — round-robin through them with fixed time
        lane = dead_unserved[0]
        duration = DEFAULT_FALLBACK_GREEN
    else:
        lane = LANES[0]
        duration = DEFAULT_FALLBACK_GREEN

    cycle_served[intersection_id].add(lane)
    phase_end_time[intersection_id] = now + duration

    signal_state[intersection_id] = {
        "counts": counts,
        "active_lane": lane,
        "phase": "GREEN",
        "remaining_time": duration,
        "vehicle_count": counts.get(lane, 0),
        "mode": "SEMI-AUTO",
        "system_ready": True,
    }
    return signal_state[intersection_id]


# =========================
# MANUAL MODE
# Owns the full cycle: GREEN → YELLOW(3s) → ALL_RED(3s) → next lane → repeat
# =========================
def run_manual_mode(intersection_id: int, now: float, counts: dict):
    phase = signal_state[intersection_id]["phase"]
    remaining = phase_end_time[intersection_id] - now

    # ── Phase still running: tick remaining time ──
    if remaining > 0:
        signal_state[intersection_id]["remaining_time"] = int(remaining)
        signal_state[intersection_id]["mode"] = "MANUAL"
        return signal_state[intersection_id]

    # ── GREEN expired → YELLOW for 3s ──
    if phase == "GREEN":
        signal_state[intersection_id]["phase"] = "YELLOW"
        signal_state[intersection_id]["remaining_time"] = YELLOW_TIME
        signal_state[intersection_id]["mode"] = "MANUAL"
        phase_end_time[intersection_id] = now + YELLOW_TIME
        return signal_state[intersection_id]

    # ── YELLOW expired → ALL_RED for 3s ──
    if phase == "YELLOW":
        signal_state[intersection_id]["phase"] = "ALL_RED"
        signal_state[intersection_id]["remaining_time"] = ALL_RED_TIME
        signal_state[intersection_id]["mode"] = "MANUAL"
        phase_end_time[intersection_id] = now + ALL_RED_TIME
        return signal_state[intersection_id]

    # ── ALL_RED (or INIT) expired → advance to next lane and go GREEN ──
    manual_index[intersection_id] = (manual_index[intersection_id] + 1) % len(LANES)
    lane = LANES[manual_index[intersection_id]]

    duration = max(MIN_GREEN, min(MAX_GREEN, manual_times_store[intersection_id].get(lane, DEFAULT_FALLBACK_GREEN)))

    signal_state[intersection_id] = {
        "counts": counts,
        "active_lane": lane,
        "phase": "GREEN",
        "remaining_time": duration,
        "vehicle_count": counts.get(lane, 0),
        "mode": "MANUAL",
        "system_ready": True,
    }
    phase_end_time[intersection_id] = now + duration
    return signal_state[intersection_id]


# =========================
# MANUAL OVERRIDE ENDPOINT
# =========================
class ManualOverride(BaseModel):
    intersection_id: int
    lane_times: Optional[Dict[str, int]] = None
    start_lane: Optional[str] = None
    lane: Optional[str] = None
    green_time: Optional[int] = None


@app.post("/traffic/manual_override")
def manual_override(data: ManualOverride):
    ensure_intersection(data.intersection_id)
    now = time.time()

    if data.lane_times:
        update = {
            lane: max(MIN_GREEN, min(MAX_GREEN, t))
            for lane, t in data.lane_times.items()
            if lane in LANES
        }
    elif data.lane and data.green_time is not None:
        if data.lane not in LANES:
            return {"error": f"Invalid lane: {data.lane}"}
        update = {data.lane: max(MIN_GREEN, min(MAX_GREEN, data.green_time))}
    else:
        return {"error": "Provide either lane_times or lane+green_time"}

    manual_times_store[data.intersection_id].update(update)

    # Operator-chosen MANUAL — cameras recovering should NOT auto-restore to SEMI-AUTO
    camera_failure_mode[data.intersection_id] = False
    pending_auto_restore[data.intersection_id] = False

    current_phase = signal_state[data.intersection_id].get("phase", "ALL_RED")

    if current_phase == "GREEN":
        signal_state[data.intersection_id]["phase"] = "YELLOW"
        phase_end_time[data.intersection_id] = now + YELLOW_TIME
    elif current_phase not in ("YELLOW", "ALL_RED"):
        signal_state[data.intersection_id]["phase"] = "ALL_RED"
        phase_end_time[data.intersection_id] = now + ALL_RED_TIME

    control_mode[data.intersection_id] = "MANUAL"
    signal_state[data.intersection_id]["mode"] = "MANUAL"

    start = data.start_lane if data.start_lane and data.start_lane in LANES else LANES[0]
    manual_index[data.intersection_id] = LANES.index(start) - 1

    return {
        "status": "Manual override activated — YELLOW transition started",
        "lane_times": manual_times_store[data.intersection_id],
        "phase": signal_state[data.intersection_id]["phase"],
        "mode": "MANUAL",
    }


# =========================
# CONTROL (AUTO / MANUAL toggle from frontend)
# =========================
class ControlPayload(BaseModel):
    intersection_id: int
    mode: str  # "AUTO" or "MANUAL"
    manual_times: Optional[Dict[str, int]] = None


@app.post("/traffic/control")
def set_control(data: ControlPayload):
    ensure_intersection(data.intersection_id)

    if data.manual_times:
        manual_times_store[data.intersection_id].update({
            lane: max(MIN_GREEN, min(MAX_GREEN, t))
            for lane, t in data.manual_times.items()
            if lane in LANES
        })

    mode = data.mode.upper()

    if mode == "AUTO":
        # Respect the 1-camera = SEMI-AUTO, 2+ cameras = MANUAL rule
        working = failure_manager.get_working_lanes(data.intersection_id)
        all_offline = frontend_camera_offline.get(data.intersection_id, set())
        effective_working = [l for l in working if l not in all_offline]
        ctrl_down = len(LANES) - len(effective_working)

        if ctrl_down >= 2:
            # 2+ cameras down → MANUAL
            control_mode[data.intersection_id] = "MANUAL"
            signal_state[data.intersection_id]["mode"] = "MANUAL"
            camera_failure_mode[data.intersection_id] = True
            manual_times_store[data.intersection_id] = {l: DEFAULT_FALLBACK_GREEN for l in LANES}
            manual_index[data.intersection_id] = -1
        elif ctrl_down == 1:
            # 1 camera down → SEMI-AUTO
            control_mode[data.intersection_id] = "SEMI-AUTO"
            signal_state[data.intersection_id]["mode"] = "SEMI-AUTO"
            camera_failure_mode[data.intersection_id] = True
            cycle_served[data.intersection_id] = set()
        else:
            control_mode[data.intersection_id] = "AUTO"
            signal_state[data.intersection_id]["mode"] = "AUTO"
            cycle_served[data.intersection_id] = set()
    else:
        control_mode[data.intersection_id] = "MANUAL"
        signal_state[data.intersection_id]["mode"] = "MANUAL"
        # Don't treat operator manual as camera-failure manual
        camera_failure_mode[data.intersection_id] = False
        pending_auto_restore[data.intersection_id] = False

    return {"ok": True, "mode": control_mode[data.intersection_id]}


# =========================
# CAMERA STATUS (from frontend toggle button)
# =========================
class CameraStatusPayload(BaseModel):
    intersection_id: int
    lane: str
    status: bool  # True = working, False = offline


@app.post("/traffic/camera_status")
async def set_camera_status(data: CameraStatusPayload):
    ensure_intersection(data.intersection_id)

    i = data.intersection_id
    lane = data.lane

    if lane not in LANES:
        return {"error": f"Invalid lane: {lane}"}

    # Apply operator toggle — this is the AUTHORITATIVE source of truth.
    # Nothing else in the system can undo what the operator sets here.
    if not data.status:
        # Operator is DISABLING this camera — lock it down
        frontend_camera_offline[i].add(lane)
        failure_manager.camera_status[i][lane] = False
        camera_down_lanes[i].add(lane)
    else:
        # Operator is RE-ENABLING this camera — only operator can do this
        frontend_camera_offline[i].discard(lane)
        # Hardware state is assumed working when operator re-enables
        failure_manager.camera_status[i][lane] = True
        camera_down_lanes[i].discard(lane)

    # Recalculate effective working lanes AFTER applying toggle
    # effective_working excludes ALL frontend-disabled lanes
    effective_working = [
        l for l in LANES
        if l not in frontend_camera_offline[i]
        and failure_manager.camera_status[i].get(l, True)
    ]
    down_count = len(LANES) - len(effective_working)
    all_down = len(effective_working) == 0

    # Apply mode rules based on how many lanes are actually usable
    if all_down or down_count >= 2:
        # 2 or more cameras disabled/down → full MANUAL, fixed rotation
        camera_failure_mode[i] = True
        pending_auto_restore[i] = False
        control_mode[i] = "MANUAL"
        signal_state[i]["mode"] = "MANUAL"
        manual_times_store[i] = {l: DEFAULT_FALLBACK_GREEN for l in LANES}
        manual_index[i] = -1
    elif down_count == 1:
        # Exactly 1 camera disabled/down → SEMI-AUTO
        camera_failure_mode[i] = True
        pending_auto_restore[i] = False
        control_mode[i] = "SEMI-AUTO"
        signal_state[i]["mode"] = "SEMI-AUTO"
        cycle_served[i] = set()
    else:
        # All cameras enabled and working → restore AUTO
        # Only reaches here when operator re-enables the last disabled camera
        camera_failure_mode[i] = False
        pending_auto_restore[i] = True

    # Notify dashboard of the toggle
    await push_camera_alert(i, lane, data.status)

    return {
        "ok": True,
        "lane": lane,
        "status": data.status,
        "mode": control_mode[i],
        "down_lanes": list(camera_down_lanes[i]),
    }


# =========================
# CAMERA ALERT (from vehicle_detection.py)
# Real hardware camera failure/recovery
# =========================
class CameraAlert(BaseModel):
    intersection_id: int
    lane: str
    is_working: bool


@app.post("/traffic/camera_alert")
async def camera_alert(data: CameraAlert):
    """
    Called by vehicle_detection.py when a hardware camera fails or recovers.
    Rule: operator-disabled lanes (frontend_camera_offline) are COMPLETELY IGNORED here.
    Hardware recovery can NEVER override an operator's manual disable decision.
    """
    ensure_intersection(data.intersection_id)

    i = data.intersection_id
    lane = data.lane

    # ABSOLUTE RULE: if operator disabled this lane, ignore all hardware signals for it.
    if lane in frontend_camera_offline.get(i, set()):
        return {
            "status": "skipped_operator_lock",
            "lane": lane,
            "mode": control_mode[i],
            "down_lanes": list(camera_down_lanes[i]),
        }

    # Update hardware state for this non-operator-locked lane
    if not data.is_working:
        camera_down_lanes[i].add(lane)
        failure_manager.camera_status[i][lane] = False
    else:
        camera_down_lanes[i].discard(lane)
        failure_manager.camera_status[i][lane] = True

    # effective_working = hardware-working lanes that are NOT operator-disabled
    all_offline = frontend_camera_offline.get(i, set())
    effective_working = [
        l for l in LANES
        if l not in all_offline and failure_manager.camera_status[i].get(l, True)
    ]
    down_count = len(LANES) - len(effective_working)
    all_down = len(effective_working) == 0

    # Apply mode rules — same logic as set_camera_status
    if all_down or down_count >= 2:
        camera_failure_mode[i] = True
        pending_auto_restore[i] = False
        if control_mode[i] != "MANUAL":
            control_mode[i] = "MANUAL"
            signal_state[i]["mode"] = "MANUAL"
            manual_times_store[i] = {l: DEFAULT_FALLBACK_GREEN for l in LANES}
            manual_index[i] = -1
    elif down_count == 1:
        camera_failure_mode[i] = True
        pending_auto_restore[i] = False
        if control_mode[i] == "AUTO":
            control_mode[i] = "SEMI-AUTO"
            signal_state[i]["mode"] = "SEMI-AUTO"
            cycle_served[i] = set()
    else:
        # All effective cameras working AND no operator-disabled lanes remain
        if len(all_offline) == 0 and camera_failure_mode[i]:
            camera_failure_mode[i] = False
            pending_auto_restore[i] = True

    await push_camera_alert(i, lane, data.is_working)

    return {
        "status": "alert_sent",
        "lane": lane,
        "is_working": data.is_working,
        "mode": control_mode[i],
        "down_lanes": list(camera_down_lanes[i]),
    }


# =========================
# EMERGENCY
# =========================
class EmergencyOverride(BaseModel):
    intersection_id: int
    lane: str
    green_time: int


@app.post("/traffic/emergency")
def emergency_override(data: EmergencyOverride):
    ensure_intersection(data.intersection_id)
    now = time.time()

    emergency_data[data.intersection_id] = {
        "lane": data.lane,
        "time": max(5, min(180, data.green_time)),
        "stage": "FORCE_YELLOW",
    }

    emergency_active[data.intersection_id] = True
    signal_state[data.intersection_id]["phase"] = "YELLOW"
    phase_end_time[data.intersection_id] = now + YELLOW_TIME

    return {"status": "Emergency Triggered"}


# =========================
# BULK UPDATE RESPONSE BUILDER
# Always includes signal info so vehicle_detection.py never needs a separate GET
# =========================
def build_bulk_response(intersection_id: int) -> dict:
    """
    Wraps signal_state with an explicit 'signal' block containing exactly
    what vehicle_detection.py needs to know: active lane, phase, remaining time,
    and current mode. The detection script reads this from the POST response
    instead of making a separate GET /traffic/status call.
    """
    state = signal_state[intersection_id]
    now = time.time()
    remaining = max(0, phase_end_time[intersection_id] - now)
    return {
        **state,
        "signal": {
            "active_lane":    state.get("active_lane"),
            "phase":          state.get("phase"),
            "remaining_time": int(remaining),
            "mode":           state.get("mode", "AUTO"),
        }
    }


# =========================
# BULK UPDATE (called every 2s by vehicle_detection.py)
# =========================
class BulkUpdate(BaseModel):
    intersection_id: int
    counts: Dict[str, int]
    video_time: float
    # NEW: optional per-lane camera health from vehicle_detection.py
    camera_health: Optional[Dict[str, bool]] = None


@app.post("/traffic/bulk_update")
async def bulk_update(data: BulkUpdate):
    ensure_intersection(data.intersection_id)

    now = time.time()
    traffic_counts[data.intersection_id] = data.counts
    phase = signal_state[data.intersection_id]["phase"]
    remaining = phase_end_time[data.intersection_id] - now

    # ======================
    # CAMERA HEARTBEAT
    # Only update heartbeat for lanes NOT manually disabled by the frontend operator.
    # Frontend-disabled lanes must stay offline regardless of detection script reports.
    # ======================
    _frontend_offline = frontend_camera_offline.get(data.intersection_id, set())
    for lane in LANES:
        if lane in _frontend_offline:
            # Keep the failure_manager permanently aware this lane is down
            failure_manager.camera_status[data.intersection_id][lane] = False
            continue
        failure_manager.update_video_heartbeat(
            data.intersection_id,
            lane,
            data.video_time > 0
        )

    # camera_health from vehicle_detection.py updates internal state only.
    # We NEVER push WS alerts from here (only /camera_alert does that).
    # We NEVER touch frontend_camera_offline lanes — operator lock is absolute.
    if data.camera_health:
        for lane, is_working in data.camera_health.items():
            if lane not in LANES:
                continue
            if lane in _frontend_offline:
                # Hard lock: keep this lane marked down no matter what detection says
                failure_manager.camera_status[data.intersection_id][lane] = False
                camera_down_lanes[data.intersection_id].add(lane)
                continue
            failure_manager.camera_status[data.intersection_id][lane] = is_working
            if not is_working:
                camera_down_lanes[data.intersection_id].add(lane)
            else:
                camera_down_lanes[data.intersection_id].discard(lane)

    # ======================
    # DETERMINE WORKING LANES
    # effective_working = hardware-working lanes that are NOT operator-disabled.
    # Uses camera_status directly — same source of truth as all other endpoints.
    # ======================
    effective_working = [
        l for l in LANES
        if l not in _frontend_offline
        and failure_manager.camera_status[data.intersection_id].get(l, True)
    ]

    down_count = len(LANES) - len(effective_working)
    all_down = len(effective_working) == 0
    some_down = down_count > 0

    # ======================
    # CAMERA FAILURE → MODE SWITCH
    # ======================
    if all_down or down_count >= 2:
        # 2+ cameras down → full MANUAL
        if not camera_failure_mode[data.intersection_id] or control_mode[data.intersection_id] != "MANUAL":
            camera_failure_mode[data.intersection_id] = True
            pending_auto_restore[data.intersection_id] = False
            control_mode[data.intersection_id] = "MANUAL"
            signal_state[data.intersection_id]["mode"] = "MANUAL"
            manual_times_store[data.intersection_id] = {l: DEFAULT_FALLBACK_GREEN for l in LANES}
            manual_index[data.intersection_id] = -1

    elif down_count == 1:
        # Exactly 1 camera down → SEMI-AUTO
        if control_mode[data.intersection_id] in ("AUTO",):
            camera_failure_mode[data.intersection_id] = True
            control_mode[data.intersection_id] = "SEMI-AUTO"
            signal_state[data.intersection_id]["mode"] = "SEMI-AUTO"
            cycle_served[data.intersection_id] = set()

    elif camera_failure_mode[data.intersection_id]:
        # Only restore AUTO if operator has no manually disabled lanes AND
        # down_count==0 (all hardware cameras working).
        if len(_frontend_offline) == 0 and down_count == 0:
            camera_failure_mode[data.intersection_id] = False
            if not emergency_active[data.intersection_id]:
                pending_auto_restore[data.intersection_id] = True
        # While any frontend-offline lanes exist: keep camera_failure_mode=True

    # ======================
    # ACTIVE PHASE RUNNING
    # ======================
    if remaining > 0:
        if phase == "GREEN" and remaining <= YELLOW_TIME:
            phase = "YELLOW"

        current_mode = control_mode[data.intersection_id]
        if emergency_active[data.intersection_id]:
            current_mode = "EMERGENCY"

        signal_state[data.intersection_id] = {
            "counts": data.counts,
            "active_lane": signal_state[data.intersection_id]["active_lane"],
            "phase": phase,
            "remaining_time": int(remaining),
            "vehicle_count": data.counts.get(signal_state[data.intersection_id]["active_lane"], 0),
            "mode": current_mode,
            "system_ready": True,
        }

        await notify_dashboard(data.intersection_id)
        return build_bulk_response(data.intersection_id)

    # ======================
    # MANUAL MODE
    # ======================
    if control_mode[data.intersection_id] == "MANUAL":
        run_manual_mode(data.intersection_id, now, data.counts)
        await notify_dashboard(data.intersection_id)
        return build_bulk_response(data.intersection_id)

    # ======================
    # TRANSITIONS (AUTO / EMERGENCY only — MANUAL handled above)
    # ======================
    if phase == "GREEN":
        signal_state[data.intersection_id]["phase"] = "YELLOW"
        phase_end_time[data.intersection_id] = now + YELLOW_TIME
        await notify_dashboard(data.intersection_id)
        return build_bulk_response(data.intersection_id)

    if phase == "YELLOW":
        signal_state[data.intersection_id]["phase"] = "ALL_RED"
        phase_end_time[data.intersection_id] = now + ALL_RED_TIME
        await notify_dashboard(data.intersection_id)
        return build_bulk_response(data.intersection_id)

    # ======================
    # EMERGENCY SAFE FLOW
    # ======================
    if emergency_active[data.intersection_id]:
        edata = emergency_data[data.intersection_id]

        if edata["stage"] == "FORCE_YELLOW":
            if remaining > 0:
                signal_state[data.intersection_id]["remaining_time"] = int(remaining)
                await notify_dashboard(data.intersection_id)
                return build_bulk_response(data.intersection_id)
            signal_state[data.intersection_id]["phase"] = "ALL_RED"
            phase_end_time[data.intersection_id] = now + ALL_RED_TIME
            edata["stage"] = "FORCE_RED"
            await notify_dashboard(data.intersection_id)
            return build_bulk_response(data.intersection_id)

        if edata["stage"] == "FORCE_RED":
            lane = edata["lane"]
            duration = edata["time"]
            phase_end_time[data.intersection_id] = now + duration
            signal_state[data.intersection_id] = {
                "counts": data.counts,
                "active_lane": lane,
                "phase": "GREEN",
                "remaining_time": duration,
                "vehicle_count": data.counts.get(lane, 0),
                "mode": "EMERGENCY",
                "system_ready": True,
            }
            edata["stage"] = "RUNNING"
            await notify_dashboard(data.intersection_id)
            return build_bulk_response(data.intersection_id)

        if edata["stage"] == "RUNNING":
            if remaining > 0:
                signal_state[data.intersection_id]["remaining_time"] = int(remaining)
                await notify_dashboard(data.intersection_id)
                return build_bulk_response(data.intersection_id)
            emergency_active[data.intersection_id] = False
            emergency_data[data.intersection_id] = {}

    # ======================
    # PENDING AUTO RESTORE
    # ======================
    if pending_auto_restore[data.intersection_id]:
        pending_auto_restore[data.intersection_id] = False

        if down_count >= 2 or all_down:
            # Still 2+ cameras down — keep MANUAL
            control_mode[data.intersection_id] = "MANUAL"
            signal_state[data.intersection_id]["mode"] = "MANUAL"
            manual_times_store[data.intersection_id] = {l: DEFAULT_FALLBACK_GREEN for l in LANES}
            manual_index[data.intersection_id] = -1
        elif down_count == 1:
            # Still 1 camera down — drop to SEMI-AUTO
            control_mode[data.intersection_id] = "SEMI-AUTO"
            signal_state[data.intersection_id]["mode"] = "SEMI-AUTO"
            cycle_served[data.intersection_id] = set()
        else:
            # All cameras back → full AUTO
            control_mode[data.intersection_id] = "AUTO"
            signal_state[data.intersection_id]["mode"] = "AUTO"
            cycle_served[data.intersection_id] = set()

    # ======================
    # SEMI-AUTO MODE (exactly 1 camera down)
    # ======================
    if control_mode[data.intersection_id] == "SEMI-AUTO":
        run_semi_auto(data.intersection_id, now, data.counts, effective_working)
        await notify_dashboard(data.intersection_id)
        return build_bulk_response(data.intersection_id)

    # ======================
    # AUTO MODE
    # ======================
    if len(cycle_served[data.intersection_id]) == 4:
        cycle_served[data.intersection_id] = set()

    eligible = {
        k: v
        for k, v in data.counts.items()
        if k not in cycle_served[data.intersection_id]
        and k in effective_working
    }

    if not eligible:
        # Reset cycle but still respect effective_working (never serve offline lanes)
        cycle_served[data.intersection_id] = set()
        eligible = {
            k: v
            for k, v in data.counts.items()
            if k in effective_working
        }
    if not eligible:
        # Absolute fallback: all lanes offline, serve any lane (MANUAL will handle it)
        eligible = data.counts

    lane = max(eligible, key=eligible.get)
    duration = calculate_green_time(data.counts[lane])
    cycle_served[data.intersection_id].add(lane)
    phase_end_time[data.intersection_id] = now + duration

    signal_state[data.intersection_id] = {
        "counts": data.counts,
        "active_lane": lane,
        "phase": "GREEN",
        "remaining_time": duration,
        "vehicle_count": data.counts.get(lane, 0),
        "mode": "AUTO",
        "system_ready": True,
    }

    await notify_dashboard(data.intersection_id)
    return build_bulk_response(data.intersection_id)


# =========================
# STATUS
# =========================
@app.get("/traffic/status/{intersection_id}")
def get_status(intersection_id: int):
    ensure_intersection(intersection_id)
    now = time.time()
    remaining = phase_end_time[intersection_id] - now
    if remaining < 0:
        remaining = 0
    signal_state[intersection_id]["remaining_time"] = int(remaining)

    # Include camera info in status response too
    down = list(camera_down_lanes.get(intersection_id, set()))
    signal_state[intersection_id]["camera_down_lanes"] = down
    signal_state[intersection_id]["has_camera_failure"] = len(down) > 0

    return signal_state[intersection_id]
