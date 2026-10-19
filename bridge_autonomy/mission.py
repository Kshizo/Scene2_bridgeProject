#!/usr/bin/env python3
"""Fly a plan.npz (from plan_survey.py) with ArduCopter in GUIDED mode.

Streams position + velocity feed-forward + yaw at 20 Hz; the reference only advances while
the vehicle (EKF estimate - no ground truth) keeps up. Navigation source management:
  * before the path enters the GPS-denied zone (switch_lookahead metres ahead) the EKF is
    switched to source set 2 (external nav = VIO);
  * after leaving it, once GPS has been healthy for gps_recover_hold seconds, back to set 1.

Usage: mission.py <config.yaml>"""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_model import load_config
from pymavlink import mavutil

cfg = load_config(sys.argv[1]); sc, gd, mo = cfg["scene"], cfg["gps_denied"], cfg["motion"]
plan = np.load(os.path.join(sc["out_dir"], "plan.npz"))
P, V, T, YAW, S, ZONE = plan["P"], plan["V"], plan["T"], plan["yaw"], plan["S"], plan["zone"]
print(f"plan: {S[-1]:.0f} m, {len(P)} samples, {100*ZONE.mean():.0f}% GPS-denied", flush=True)
# distance (along the path) to the next GPS-denied sample, for the look-ahead switch
nxt = np.full(len(P), np.inf); d = np.inf
for i in range(len(P) - 1, -1, -1):
    d = 0.0 if ZONE[i] else d + (S[i + 1] - S[i] if i + 1 < len(P) else 0)
    nxt[i] = d

m = mavutil.mavlink_connection(cfg["mavlink"]["mission"], source_system=250)
m.wait_heartbeat(); print("heartbeat", flush=True)
for mid, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 20), (mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 2),
                (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 5)):
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, mid, 1e6 / hz, 0, 0, 0, 0, 0)
st = {"pos": np.zeros(3), "gps_ok": False, "gps_ok_since": None}


def update(timeout=0.02):
    msg = m.recv_match(type=["LOCAL_POSITION_NED", "STATUSTEXT", "GPS_RAW_INT"], blocking=True, timeout=timeout)
    if msg is None:
        return
    t = msg.get_type()
    if t == "STATUSTEXT":
        print("  [ap]", msg.text, flush=True)
    elif t == "LOCAL_POSITION_NED":
        st["pos"] = np.array([msg.y, msg.x, -msg.z])
    else:
        ok = msg.fix_type >= 3 and msg.satellites_visible >= 6
        if ok and not st["gps_ok"]:
            st["gps_ok_since"] = time.time()
        st["gps_ok"] = ok


def set_source(k):
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_SET_EKF_SOURCE_SET, 0, k, 0, 0, 0, 0, 0, 0)
    print(f"  EKF source set -> {k} ({'GPS' if k == 1 else 'VIO'})", flush=True)


t0 = time.time()
while True:
    e = m.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=2)
    if e is not None and (e.flags & 0x18) == 0x18 and not (e.flags & 0x400):
        break
    if time.time() - t0 > 300:
        raise SystemExit("EKF never ready")
print("EKF ready", flush=True); time.sleep(3)
set_source(1); source = 1
for _ in range(40):
    update(0.1)
z0 = st["pos"][2]
# NAV_TAKEOFF checks the altitude against the EKF origin but targets it above home, and home
# is reset to wherever we arm (e.g. a bridge deck). Re-home to the EKF origin after arming
# so both agree, then take off to (current height above origin + climb).
m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                        mavutil.mavlink.MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN, 0, 0, 0, 0, 0, 0)
org = m.recv_match(type="GPS_GLOBAL_ORIGIN", blocking=True, timeout=10)
if org is None:
    raise SystemExit("no GPS_GLOBAL_ORIGIN")
print(f"EKF origin {org.latitude*1e-7:.7f}, {org.longitude*1e-7:.7f}, {org.altitude/1000:.2f} m; "
      f"we are {z0:.2f} m above it", flush=True)


def rehome():
    m.mav.command_int_send(m.target_system, m.target_component, mavutil.mavlink.MAV_FRAME_GLOBAL, mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                           0, 0, 0, 0, 0, 0, org.latitude, org.longitude, org.altitude / 1000.0)
    a = m.recv_match(type="COMMAND_ACK", condition="COMMAND_ACK.command==179", blocking=True, timeout=3)
    return a is not None and a.result == 0
for attempt in range(10):
    m.set_mode("GUIDED")
    for _ in range(30):
        m.arducopter_arm(); m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        if m.motors_armed():
            break
        time.sleep(2)
    if not rehome():
        print("set-home not acknowledged", flush=True)
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0,
                            z0 + 2.0)
    t1 = time.time()
    while time.time() - t1 < 20 and st["pos"][2] < z0 + 1.0:
        update(0.2)
    if st["pos"][2] >= z0 + 1.0:
        break
    print("takeoff did not start, retrying", flush=True)
print("airborne, following plan", flush=True)

VC = cfg["vio"]
status_path = os.path.join(sc["out_dir"], "vio_status.json")
vio = {"healthy": False, "since": None, "read": 0.0}


def vio_ok_for(seconds):
    """VIO has been continuously healthy for at least `seconds`."""
    now = time.time()
    if now - vio["read"] > 0.2:
        vio["read"] = now
        try:
            h = json.load(open(status_path))["healthy"]
        except (OSError, ValueError, KeyError):
            h = False
        if h and not vio["healthy"]:
            vio["since"] = now
        vio["healthy"] = h
    return vio["healthy"] and now - vio["since"] >= seconds


MASK = 0b0000100111000000
# join the plan at the closest sample of its take-off column (the first few metres of path)
col = np.where(S <= S[0] + cfg["vehicle"]["takeoff_climb"] + 1.0)[0]
i = int(col[np.argmin(np.linalg.norm(P[col] - st["pos"], axis=1))])
s_ref, last_t, last_print = S[i], time.time(), 0
direction, unhealthy_since, state = 1, None, "fly"
while i < len(P) - 1:
    update()
    now = time.time(); dt = now - last_t; last_t = now
    err = np.linalg.norm(st["pos"] - P[i])
    want = 2 if nxt[i] <= gd["switch_lookahead"] or ZONE[i] else 1
    advance = err < (0.4 if V[i] < 1.0 else 1.0)
    if source == 2 and not vio_ok_for(0):
        unhealthy_since = unhealthy_since or now
        advance = False
        if now - unhealthy_since > gd["vio_lost_retreat"]:
            if state != "retreat":
                print("  VIO unhealthy in the GPS-denied zone - retracing the path out", flush=True)
            state, direction, advance = "retreat", -1, err < 1.0
    else:
        unhealthy_since = None
    if state == "retreat" and not ZONE[i] and nxt[i] > gd["switch_lookahead"]:
        state, direction = "wait", 1                       # out of the zone: hold until GPS + VIO are back
    if want == 2 and source == 1:
        if vio_ok_for(VC["healthy_hold"]) and state != "wait":
            set_source(2); source = 2
        else:
            advance = False                                # hold short of the zone until VIO is trusted
    elif source == 2 and (want == 1 or state == "wait") and st["gps_ok"] and st["gps_ok_since"] \
            and now - st["gps_ok_since"] > gd["gps_recover_hold"]:
        set_source(1); source = 1
    if state == "wait" and source == 1 and vio_ok_for(VC["healthy_hold"]):
        state = "fly"; print("  GPS + VIO healthy again - resuming", flush=True)
    if state == "wait":
        advance = False
    if advance:
        s_ref += direction * max(V[i], 0.3 if direction < 0 else 0) * dt
        while direction > 0 and i < len(P) - 1 and S[i + 1] <= s_ref:
            i += 1
        while direction < 0 and i > 0 and S[i] > s_ref:
            i -= 1
    p = P[i]; vel = direction * T[i] * (V[i] if (err < 1.0 and advance) else 0.0)
    m.mav.set_position_target_local_ned_send(0, m.target_system, m.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                                             MASK, p[1], p[0], -p[2], vel[1], vel[0], -vel[2], 0, 0, 0, math.pi / 2 - YAW[i], 0)
    if now - last_print > 10:
        last_print = now
        print(f"{100*S[i]/S[-1]:5.1f}%  ref ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})  err {err:.2f} m  nav {'VIO' if source == 2 else 'GPS'}"
              f"  vio {'ok' if vio['healthy'] else 'LOST'}  {state}"
              f"{'  [GPS-denied zone]' if ZONE[i] else ''}", flush=True)
    time.sleep(0.03)

while np.linalg.norm(st["pos"] - P[-1]) > 0.5:
    update(0.1)
if source == 2:
    set_source(1)
m.set_mode("LAND")
while m.motors_armed():                                   # armed state comes from HEARTBEAT messages
    m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
print("landed, mission done", flush=True)
