#!/usr/bin/env python3
"""Autonomous lidar survey of the Scene-2 bridge with ArduCopter SITL (GUIDED mode).

Covers: both sides of the bridge at 1.5 / 5 / 10 / 14 m, full 360 deg orbits of every pier
at 5 m and 1.5 m (r = 9 m), stops between the columns to see their inner faces, sweeps along
both abutment faces, and a pass over the deck. Waypoints are in world ENU metres (same frame
as scene2_parts/parts.json) and converted to ArduPilot local NED about the spawn point.

Usage: fly_scene2_survey.py [--conn udpin:0.0.0.0:14560] [--speed 2.5] [--dry]"""
import argparse, math, time
from pymavlink import mavutil

ap = argparse.ArgumentParser()
ap.add_argument("--conn", default="udpin:0.0.0.0:14560")
ap.add_argument("--speed", type=float, default=2.5)
ap.add_argument("--dry", action="store_true", help="print waypoints and exit")
args = ap.parse_args()

SPAWN = (-3.3, 47.5, 10.795)            # iris spawn on the deck mid-span, clear of the centre-line dashes
CX = -4.8                               # bridge centreline x
PIERS = [20.475, 38.475, 56.475, 74.475]
R_ORBIT = 9.0
COL_GAPS = (-6.35, -3.25)               # x midway between the 3 columns of each pier

WP = []                                 # (x, y, z, yaw_enu_deg or None, accept_radius)


def add(x, y, z, yaw=None, acc=0.6):
    WP.append((x, y, z, yaw, acc))


def face_bridge(x, y):
    """Yaw (ENU deg) pointing at the nearest point of the bridge centreline."""
    if abs(x - CX) < 1.0:
        return 90.0
    ty = min(max(y, 0.0), 95.0)
    return math.degrees(math.atan2(ty - y, CX - x))


def leg(a, b, step=6.0, yaw=None):
    n = max(1, int(math.dist(a, b) / step))
    for k in range(1, n + 1):
        p = [a[i] + (b[i] - a[i]) * k / n for i in range(3)]
        add(*p, yaw if yaw is not None else face_bridge(p[0], p[1]), acc=1.0)


def orbit(yc, z, start_deg=-90.0, sweep=360.0, step=15.0):
    n = int(abs(sweep) / step)
    for k in range(1, n + 1):
        th = math.radians(start_deg + sweep * k / n)
        x, y = CX + R_ORBIT * math.cos(th), yc + R_ORBIT * math.sin(th)
        add(x, y, z, math.degrees(math.atan2(yc - y, CX - x)), acc=1.0)


# --- plan --------------------------------------------------------------------------
add(SPAWN[0], 47.5, 16.0, 90)                      # straight up off the deck
add(20, 47.5, 16.0, 180)                           # out over the river, east of the bridge
add(20, 47.5, 4.0, 180)                            # down over open water
add(20, 10, 4.0)
# A. east side, low
leg((20, 10, 4.0), (8, 10, 1.5)); leg((8, 10, 1.5), (8, 85, 1.5)); add(8, 85, 5.0, 180)
leg((8, 85, 5.0), (8, 10, 5.0))
# B. south abutment face (under the deck, below the girders at 8.25 m)
leg((8, 10, 5.0), (8, 10, 3.0), yaw=-90); leg((8, 10, 3.0), (-18, 10, 3.0), yaw=-90)
add(-18, 10, 6.0, -90); leg((-18, 10, 6.0), (8, 10, 6.0), yaw=-90)
# C. piers: inner faces from the south, two full orbits, inner faces from the north
add(CX, PIERS[0] - R_ORBIT, 5.0, 90, acc=1.0)
for i, yc in enumerate(PIERS):
    for gx in COL_GAPS:
        add(gx, yc - 4.5, 3.0, 90)
    add(CX, yc - R_ORBIT, 5.0, 90)
    orbit(yc, 5.0)                                 # full circle at 5 m
    add(CX, yc - R_ORBIT, 1.5, 90)
    orbit(yc, 1.5)                                 # full circle at 1.5 m
    orbit(yc, 1.5, start_deg=-90, sweep=-180)      # west half again to reach the north side
    for gx in reversed(COL_GAPS):
        add(gx, yc + 4.5, 3.0, -90)
    add(CX, yc + R_ORBIT, 5.0, -90)
# D. north abutment face
leg((CX, PIERS[-1] + R_ORBIT, 5.0), (8, 85, 3.0), yaw=90); leg((8, 85, 3.0), (-18, 85, 3.0), yaw=90)
add(-18, 85, 6.0, 90); leg((-18, 85, 6.0), (8, 85, 6.0), yaw=90)
# E. west side, low (cross under the deck mid-span)
leg((8, 85, 6.0), (8, 83.5, 5.0)); leg((8, 83.5, 5.0), (-18, 83.5, 5.0), yaw=180)
leg((-18, 83.5, 5.0), (-18, 10, 5.0)); add(-18, 10, 1.5, 0); leg((-18, 10, 1.5), (-18, 85, 1.5))
# F. deck level, both sides, over the river only
add(-18, 85, 10.0, 0); leg((-18, 85, 10.0), (-18, 10, 10.0))
add(-18, 10, 14.0, 0)
# G. high passes past both ends + over the deck
leg((-18, 10, 14.0), (-18, -15, 14.0)); leg((-18, -15, 14.0), (-18, 110, 14.0))
leg((-18, 110, 14.0), (CX, 110, 16.0), yaw=-90); leg((CX, 110, 16.0), (CX, -15, 16.0), yaw=-90)
leg((CX, -15, 16.0), (8, -15, 14.0)); leg((8, -15, 14.0), (8, 110, 14.0))
add(8, 85, 14.0, 180); add(8, 85, 10.0, 180); leg((8, 85, 10.0), (8, 10, 10.0))
add(8, 10, 14.0, 180); leg((8, 10, 14.0), (8, 47.5, 16.0)); add(SPAWN[0], 47.5, 16.0, 90)   # back over the spawn

print(f"{len(WP)} waypoints, path {sum(math.dist(WP[i][:3], WP[i+1][:3]) for i in range(len(WP)-1)):.0f} m")
if args.dry:
    for w in WP:
        print("  %7.2f %7.2f %6.2f  yaw %s" % (w[0], w[1], w[2], "-" if w[3] is None else f"{w[3]:.0f}"))
    raise SystemExit


# --- fly ---------------------------------------------------------------------------
def to_ned(x, y, z):
    # SITL home = the world's <spherical_coordinates> origin, so ArduPilot local NED is just
    # the world ENU frame rotated: N = y, E = x, D = -z
    return y, x, -z


m = mavutil.mavlink_connection(args.conn, source_system=250)
m.wait_heartbeat()
print("heartbeat from", m.target_system, flush=True)


def msg_interval(mid, hz):
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                            0, mid, 1e6 / hz, 0, 0, 0, 0, 0)


msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10)
for name, val in (("WPNAV_SPEED", args.speed * 100), ("WPNAV_SPEED_UP", 150), ("WPNAV_SPEED_DN", 100),
                  ("WPNAV_ACCEL", 150)):
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(), val,
                         mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

pos = [0, 0, 0]


def update():
    msg = m.recv_match(type=["LOCAL_POSITION_NED", "STATUSTEXT"], blocking=True, timeout=1)
    if msg is None:
        return
    if msg.get_type() == "STATUSTEXT":
        print("  [ap]", msg.text, flush=True)
    else:
        pos[:] = [msg.x, msg.y, msg.z]


def ekf_ready():
    """EKF has an absolute horizontal position (GPS) and vertical position."""
    msg = m.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=2)
    return msg is not None and (msg.flags & 0x10) and (msg.flags & 0x08) and not (msg.flags & 0x400)


msg_interval(mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 2)
t0 = time.time()
while not ekf_ready():
    if time.time() - t0 > 300:
        raise SystemExit("EKF never got a GPS position")
print("EKF ready", flush=True)
time.sleep(3)

for _ in range(20):
    update()
z_ground = pos[2]
for attempt in range(10):                         # arm + takeoff, retry if it auto-disarms
    m.set_mode("GUIDED")
    for _ in range(30):                           # GPS may still be "configuring"
        m.arducopter_arm()
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        if m.motors_armed():
            break
        update(); time.sleep(2)
    if not m.motors_armed():
        continue
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                            0, 0, 0, 0, 0, 0, 0, 3.0)
    t0 = time.time()
    while time.time() - t0 < 20 and pos[2] > z_ground - 1.0:
        update()
    if pos[2] <= z_ground - 1.0:
        break
    print("takeoff did not start, retrying", flush=True)
print("climbing", flush=True)
while pos[2] > z_ground - 2.6:
    update()
print("airborne", flush=True)

for i, (x, y, z, yaw, acc) in enumerate(WP):
    n, e, d = to_ned(x, y, z)
    yaw_ned = math.radians(90.0 - (yaw if yaw is not None else 90.0))
    t0 = time.time(); best = 1e9; bumped = False
    while True:
        m.mav.set_position_target_local_ned_send(0, m.target_system, m.target_component,
                                                 mavutil.mavlink.MAV_FRAME_LOCAL_NED, 0b0000100111111000,
                                                 n, e, d, 0, 0, 0, 0, 0, 0, yaw_ned, 0)
        for _ in range(3):
            update()
        err = math.dist(pos, (n, e, d))
        if err < acc:
            break
        if time.time() - t0 > 12 and err > best - 0.3 and not bumped:
            print(f"  wp {i}: no progress (err {err:.1f} m) - climbing 2 m", flush=True)
            for _ in range(40):
                m.mav.set_position_target_local_ned_send(0, m.target_system, m.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED, 0b0000100111111000, pos[0], pos[1], pos[2] - 2.0,
                    0, 0, 0, 0, 0, 0, yaw_ned, 0)
                update()
            bumped = True
        if time.time() - t0 > 12:
            best, t0b = min(best, err), t0
        else:
            best = min(best, err)
        if time.time() - t0 > 120:
            print(f"  wp {i} timeout at {pos}", flush=True)
            break
    if i % 10 == 0:
        print(f"wp {i}/{len(WP)}  world ({x:.1f}, {y:.1f}, {z:.1f})", flush=True)

m.set_mode("LAND")
while m.motors_armed():
    update()
print("landed, survey done", flush=True)
