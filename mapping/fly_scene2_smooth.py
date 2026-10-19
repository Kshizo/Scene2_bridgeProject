#!/usr/bin/env python3
"""Smooth lidar survey of the Scene-2 bridge (ArduCopter SITL, GUIDED, pos+vel streaming).

The path is built from lines, arcs and helices, sampled every 0.1 m, time-parameterised
with speed limits (open air / under the deck / around the columns), and streamed at 20 Hz
as position + velocity feed-forward, so the drone flies through it without stopping.
The reference waits for the drone if the tracking error grows (tight near the columns).

Coverage: both sides low/mid/deck-level/high, both abutment faces, under-deck passes
threaded through the pier column gaps, a rising helix around every pier and a figure-8
weave that loops each individual column (twice, climbing), then a pass over the deck.
World ENU metres = scene2_parts/parts.json frame. SITL local NED: N=y, E=x, D=-z.

Usage: fly_scene2_smooth.py [--conn udpin:0.0.0.0:14560] [--check] [--plot out.png]"""
import argparse, math, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--conn", default="udpin:0.0.0.0:14560")
ap.add_argument("--check", action="store_true", help="build + clearance-check the path, don't fly")
ap.add_argument("--plot", default=None)
args = ap.parse_args()

SPAWN = (-3.3, 47.5)                      # on the deck pad, z 10.9
CX = -4.8
PIERS = [20.475, 38.475, 56.475, 74.475]
COLS = [-7.902, -4.802, -1.702]           # column centres (1 m square, z -1..7.15)
GAPS = (-6.352, -3.252)
R_COL, R_HELIX = 1.55, 7.0
V_OPEN, V_UNDER, V_COL = 2.0, 1.2, 0.8    # m/s
ACC = 0.6                                 # m/s^2

segs = []                                 # list of (points Nx3, vmax, yaw_mode, yaw_arg)


def line(a, b, v=V_OPEN, yaw=("bridge", None)):
    a, b = np.array(a, float), np.array(b, float)
    n = max(2, int(np.linalg.norm(b - a) / 0.1) + 1)
    segs.append((a + (b - a) * np.linspace(0, 1, n)[:, None], v, yaw))


def arc(c, r, a0, a1, z0, z1, v, yaw):
    """Circle/helix about centre c (x, y) from angle a0 to a1 (deg, sign = direction)."""
    n = max(2, int(abs(math.radians(a1 - a0)) * r / 0.1) + 1)
    th = np.radians(np.linspace(a0, a1, n))
    P = np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th), np.linspace(z0, z1, n)], 1)
    segs.append((P, v, yaw))


def go(b, v=V_OPEN, yaw=("bridge", None)):
    line(last(), b, v, yaw)


def last():
    return tuple(segs[-1][0][-1])


# ---- plan --------------------------------------------------------------------------
line((SPAWN[0], SPAWN[1], 11.2), (SPAWN[0], SPAWN[1], 16), 1.0, ("fixed", 90))
go((20, 47.5, 16)); go((20, 47.5, 4)); go((8, 10, 1.5))
# a. east side
go((8, 85, 1.5)); go((8, 85, 5)); go((8, 10, 5))
# b. south abutment face, under the deck
go((-18, 10, 5), V_UNDER, ("fixed", -90))
# c. west side, north abutment, low sweeps
go((-18, 85, 5)); go((8, 85, 5), V_UNDER, ("fixed", 90)); go((8, 85, 3), V_UNDER, ("fixed", 90))
go((-18, 85, 3), V_UNDER, ("fixed", 90)); go((-18, 85, 1.5)); go((-18, 10, 1.5))
go((-18, 10, 3)); go((8, 10, 3), V_UNDER, ("fixed", -90))
# d. under the deck, threaded through the column gaps (below the pier caps at 7.15)
go((GAPS[0], 10, 5.5), V_UNDER, ("along", None)); go((GAPS[0], 87, 5.5), V_UNDER, ("along", None))
go((GAPS[1], 87, 5.5), V_UNDER, ("along", None)); go((GAPS[1], 10, 5.5), V_UNDER, ("along", None))
# e. every pier: rising helix around the group, then a figure-8 weave around each column
for yc in PIERS:
    go((CX, yc - R_HELIX, 1.0), V_UNDER, ("along", None))
    arc((CX, yc), R_HELIX, -90, 630, 1.0, 7.0, V_UNDER, ("centre", (CX, yc)))     # 2 turns, 1 -> 7 m
    go((GAPS[0], yc - 4.0, 2.0), V_UNDER, ("along", None))
    go((GAPS[0], yc, 2.0), V_COL, ("fixed", 90))
    for k in range(2):                                     # two weaves, climbing 2 -> 5.5 m
        z = [2.0 + 3.5 * (k * 4 + i) / 8 for i in range(5)]
        arc((COLS[0], yc), R_COL, 0, 360, z[0], z[1], V_COL, ("centre", (COLS[0], yc)))    # col 1 CCW
        arc((COLS[1], yc), R_COL, 180, 0, z[1], z[2], V_COL, ("centre", (COLS[1], yc)))    # col 2 CW, north side
        arc((COLS[2], yc), R_COL, 180, 540, z[2], z[3], V_COL, ("centre", (COLS[2], yc)))  # col 3 CCW
        arc((COLS[1], yc), R_COL, 0, -180, z[3], z[4], V_COL, ("centre", (COLS[1], yc)))   # col 2 CW, south side
    go((GAPS[0], yc + 4.0, 5.5), V_COL, ("fixed", 90))
# f. out west, deck-level and high passes, over the deck, back to the pad
go((-18, 85, 5), V_UNDER, ("along", None)); go((-18, 85, 10)); go((-18, 10, 10))
go((-18, 10, 14)); go((-18, -15, 14)); go((-18, 110, 14))
go((CX, 110, 16), V_OPEN, ("fixed", -90)); go((CX, -15, 16), V_OPEN, ("fixed", -90))
go((8, -15, 14)); go((8, 110, 14)); go((8, 85, 10)); go((8, 10, 10)); go((8, 10, 16))
go((SPAWN[0], SPAWN[1], 16), V_OPEN, ("fixed", 90))

# ---- sample + time-parameterise ----------------------------------------------------
P, VMAX, YAW = [], [], []
for pts, v, (mode, arg) in segs:
    for i, p in enumerate(pts):
        if P and i == 0:
            continue
        P.append(p); VMAX.append(v)
        if mode == "fixed":
            YAW.append(math.radians(arg))
        elif mode == "centre":
            YAW.append(math.atan2(arg[1] - p[1], arg[0] - p[0]))
        elif mode == "along":
            q = pts[min(i + 1, len(pts) - 1)] if i + 1 < len(pts) else p + (p - pts[i - 1])
            d = q[:2] - p[:2]
            YAW.append(math.atan2(d[1], d[0]) if np.hypot(*d) > 1e-6 else (YAW[-1] if YAW else 0))
        else:                                              # face the bridge centreline
            ty = min(max(p[1], 0), 95)
            YAW.append(math.atan2(ty - p[1], CX - p[0]) if abs(p[0] - CX) > 1 else math.radians(90))
P = np.array(P); VMAX = np.array(VMAX); YAW = np.unwrap(np.array(YAW))
D = np.linalg.norm(np.diff(P, axis=0), axis=1)
keep = np.concatenate([[True], D > 1e-6]); P, VMAX, YAW = P[keep], VMAX[keep], YAW[keep]
D = np.linalg.norm(np.diff(P, axis=0), axis=1)
S = np.concatenate([[0], np.cumsum(D)])
# corner slow-down: turn angle over +-0.5 m
T = np.gradient(P, S, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
k = 5
turn = np.zeros(len(P))
turn[k:-k] = np.degrees(np.arccos(np.clip(np.sum(T[2 * k:] * T[:-2 * k], 1), -1, 1)))
VMAX = np.minimum(VMAX, np.where(turn > 60, 0.3, np.where(turn > 25, 0.6, 9)))
# yaw-rate limit (40 deg/s)
dyaw = np.abs(np.diff(YAW, append=YAW[-1]))
VMAX = np.minimum(VMAX, np.where(dyaw > 0, math.radians(40) * np.maximum(D.mean(), 0.1) / np.maximum(dyaw, 1e-6), 9))
V = VMAX.copy(); V[0] = V[-1] = 0
for i in range(1, len(V)):
    V[i] = min(V[i], math.sqrt(V[i - 1] ** 2 + 2 * ACC * D[i - 1]))
for i in range(len(V) - 2, -1, -1):
    V[i] = min(V[i], math.sqrt(V[i + 1] ** 2 + 2 * ACC * D[i]))
V = np.maximum(V, 0.05)
TT = np.concatenate([[0], np.cumsum(D / ((V[1:] + V[:-1]) / 2))])
print(f"path {S[-1]:.0f} m, est. {TT[-1]/60:.1f} min of sim time, {len(P)} samples", flush=True)

if args.check or args.plot:
    import glob, os
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0); pts = []
    for f in glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scene2_parts", "*.obj")):
        if "water" in f:
            continue
        V_, F_ = [], []
        for l in open(f):
            t = l.split()
            if t and t[0] == "v": V_.append([float(x) for x in t[1:4]])
            elif t and t[0] == "f":
                ii = [int(x.split("/")[0]) - 1 for x in t[1:]]; F_ += [[ii[0], ii[j], ii[j + 1]] for j in range(1, len(ii) - 1)]
        Tr = np.array(V_)[np.array(F_)]
        A = np.linalg.norm(np.cross(Tr[:, 1] - Tr[:, 0], Tr[:, 2] - Tr[:, 0]), axis=1) / 2
        for t, a in zip(Tr, A):
            n = min(int(a / 0.01) + 3, 300000)
            r = rng.random((n, 2)); m = r.sum(1) > 1; r[m] = 1 - r[m]
            q = t[0] + r[:, :1] * (t[1] - t[0]) + r[:, 1:] * (t[2] - t[0])
            pts.append(q[(q[:, 0] > -40) & (q[:, 0] < 40) & (q[:, 1] > -40) & (q[:, 1] < 130)])
    Q = np.concatenate(pts); tree = cKDTree(Q)
    d, _ = tree.query(P)
    order = np.argsort(d)
    print("closest approaches (m):", flush=True)
    seen = []
    for i in order:
        if all(abs(i - j) > 50 for j in seen):
            seen.append(i); print(f"  {d[i]:.2f} at ({P[i,0]:.2f}, {P[i,1]:.2f}, {P[i,2]:.2f}) v={V[i]:.2f}")
        if len(seen) >= 8:
            break
    if args.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 2, 1, projection="3d"); sub = Q[rng.choice(len(Q), 60000)]
        ax.scatter(*sub.T, s=0.1, c="0.7"); ax.plot(*P.T, lw=0.8, c="r"); ax.view_init(25, -50)
        ax.set_xlim(-60, 60); ax.set_ylim(-15, 105); ax.set_zlim(-10, 30)
        ax2 = fig.add_subplot(1, 2, 2); m = (P[:, 1] > 12) & (P[:, 1] < 30)
        ax2.scatter(sub[:, 0], sub[:, 1], s=0.1, c="0.7"); ax2.plot(P[m, 0], P[m, 1], lw=0.6, c="r")
        ax2.set_xlim(-14, 5); ax2.set_ylim(12, 30); ax2.set_aspect("equal"); ax2.set_title("pier 1 (plan)")
        plt.savefig(args.plot, dpi=80)
    raise SystemExit

# ---- fly ---------------------------------------------------------------------------
from pymavlink import mavutil
m = mavutil.mavlink_connection(args.conn, source_system=250)
m.wait_heartbeat(); print("heartbeat", flush=True)
for mid, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 20), (mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 2)):
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, mid, 1e6 / hz, 0, 0, 0, 0, 0)
pos = [0.0, 0.0, 0.0]


def update(timeout=0.05):
    msg = m.recv_match(type=["LOCAL_POSITION_NED", "STATUSTEXT"], blocking=True, timeout=timeout)
    if msg is None:
        return
    if msg.get_type() == "STATUSTEXT":
        print("  [ap]", msg.text, flush=True)
    else:
        pos[:] = [msg.y, msg.x, -msg.z]                   # -> world ENU


t0 = time.time()
while True:
    e = m.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=2)
    if e is not None and (e.flags & 0x18) == 0x18 and not (e.flags & 0x400):
        break
    if time.time() - t0 > 300:
        raise SystemExit("EKF never ready")
print("EKF ready", flush=True); time.sleep(3)
for _ in range(40):
    update(0.1)
z_ground = pos[2]
for attempt in range(10):
    m.set_mode("GUIDED")
    for _ in range(30):
        m.arducopter_arm(); m.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        if m.motors_armed():
            break
        time.sleep(2)
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 2.0)
    t1 = time.time()
    while time.time() - t1 < 20 and pos[2] < z_ground + 1.0:
        update(0.2)
    if pos[2] >= z_ground + 1.0:
        break
    print("takeoff did not start, retrying", flush=True)
while pos[2] < z_ground + 1.7:
    update(0.2)
print("airborne, following path", flush=True)

MASK = 0b0000100111000000                                  # pos + vel + yaw
i_ref, s_ref, last_t, last_print = 0, 0.0, time.time(), 0
P0 = np.array(pos)
lead = np.linalg.norm(P[0] - P0)
while i_ref < len(P) - 1:
    update(0.02)
    now = time.time(); dt = now - last_t; last_t = now
    err = np.linalg.norm(np.array(pos) - P[i_ref])
    tight = V[i_ref] <= V_COL + 1e-6
    if err < (0.35 if tight else 1.0):                    # reference only advances when we keep up
        s_ref += V[i_ref] * dt
        while i_ref < len(P) - 1 and S[i_ref + 1] <= s_ref:
            i_ref += 1
    p = P[i_ref]; t = T[i_ref] * (V[i_ref] if err < 1.0 else 0)
    m.mav.set_position_target_local_ned_send(0, m.target_system, m.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                                             MASK, p[1], p[0], -p[2], t[1], t[0], -t[2], 0, 0, 0,
                                             math.pi / 2 - YAW[i_ref], 0)
    if now - last_print > 10:
        last_print = now
        print(f"{100*S[i_ref]/S[-1]:5.1f}%  ref ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})  err {err:.2f} m", flush=True)
    time.sleep(0.03)

while np.linalg.norm(np.array(pos) - P[-1]) > 0.5:
    update(0.1)
m.set_mode("LAND")
while m.motors_armed():
    update(0.2)
print("landed, survey done", flush=True)
