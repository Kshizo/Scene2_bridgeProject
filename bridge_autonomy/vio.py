#!/usr/bin/env python3
"""Visual-inertial odometry for the bridge drone -> ArduPilot external navigation.

Front end : Shi-Tomasi corners on the RGB image, tracked frame to frame with pyramidal KLT
            (forward-backward checked). New corners get a 3-D world landmark from the depth
            image and the current pose; each frame is localised against the tracked
            landmarks with PnP + RANSAC, seeded by the IMU prediction.
Inertial  : gyro integration predicts attitude between frames and a constant-velocity model
            predicts position; implausible PnP jumps are rejected; the PnP pose corrects the
            prediction (complementary filter). ArduPilot's EKF3 then fuses this with its own
            IMU (accelerometers) as external navigation.
Health    : only sent to ArduPilot while tracking; status in <out_dir>/vio_status.json.
            With GPS available a lost VIO re-initialises from GPS (xy) + EKF (z, attitude).
Frames    : estimates live in the world ENU frame. While GPS is healthy the output is
            slowly aligned (yaw + translation) to the EKF so switching sources is seamless;
            in the GPS-denied zone the alignment freezes and the VIO dead-reckons.
Mounting  : camera pose comes from the drone model SDF, IMU orientation is calibrated from
            gravity at rest, intrinsics from camera_info.
Ground truth (gz model pose) is only logged for evaluation, never used by the estimator.

Usage: vio.py <config.yaml>
"""
import json, math, os, subprocess, sys, threading, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_model import load_config
from sdf_utils import model_sdf_for_include, link_pose
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.imu_pb2 import IMU
from gz.msgs10.camera_info_pb2 import CameraInfo
from gz.msgs10.pose_v_pb2 import Pose_V
from pymavlink import mavutil

cfg = load_config(sys.argv[1]); vc = cfg["vio"]; sc = cfg["scene"]
G = 9.80665


def stamp(h):
    return h.stamp.sec + h.stamp.nsec * 1e-9


def expm(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def logm(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


# ---- mounting from the model SDF -----------------------------------------------------
model_sdf = model_sdf_for_include(sc["world_sdf"], sc["vehicle_model"])
R_bl, t_bl, sensor = link_pose(model_sdf, vc["camera_link"])           # camera link in body (FLU)
R_lo = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)             # optical (x right, y down, z fwd) in link
R_bc, t_bc = R_bl @ R_lo, t_bl                                          # optical camera in body
print(f"camera in body: t={t_bc.round(3)} from {model_sdf}", flush=True)


def find_topic(*parts):
    tl = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True).stdout.split()
    hits = [t for t in tl if all(p in t for p in parts)]
    if not hits:
        raise RuntimeError(f"no gz topic containing {parts}")
    return sorted(hits, key=len)[0]


imu_topic = find_topic(f"/model/{sc['vehicle_model']}/", "imu")
print("imu topic", imu_topic, flush=True)

lock = threading.Lock()
imu_buf, frames, depths, truth = [], {}, {}, {}
K = [None]


def on_imu(m):
    with lock:
        imu_buf.append((stamp(m.header), np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]),
                        np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])))
        if len(imu_buf) > 20000:
            del imu_buf[:10000]


def on_img(m):
    with lock:
        frames[stamp(m.header)] = m
        for k in sorted(frames)[:-5]:
            del frames[k]


def on_depth(m):
    with lock:
        depths[stamp(m.header)] = m
        for k in sorted(depths)[:-5]:
            del depths[k]


def on_info(m):
    if K[0] is None:
        k = list(m.intrinsics.k)
        K[0] = np.array(k, float).reshape(3, 3)
        print("intrinsics", K[0].round(1).tolist(), flush=True)


def on_truth(m):
    for p in m.pose:
        if p.name == sc["vehicle_model"]:
            truth["p"] = np.array([p.position.x, p.position.y, p.position.z]); truth["t"] = stamp(m.header)


node = Node()
node.subscribe(IMU, imu_topic, on_imu)
node.subscribe(Image, vc["rgbd_topic"] + "/image", on_img)
node.subscribe(Image, vc["rgbd_topic"] + "/depth_image", on_depth)
node.subscribe(CameraInfo, vc["rgbd_topic"] + "/camera_info", on_info)
node.subscribe(Pose_V, f"/world/{sc['world_name']}/dynamic_pose/info", on_truth)

mav = mavutil.mavlink_connection(vc["mavlink"], source_system=1, source_component=197)
mav.wait_heartbeat(); print("mavlink up", flush=True)
for mid, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 20), (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 20),
                (mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 10)):
    mav.mav.command_long_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, mid, 1e6 / hz, 0, 0, 0, 0, 0)
ekf = {}


def poll_mav():
    while True:
        m = mav.recv_match(blocking=False)
        if m is None:
            return
        t = m.get_type()
        if t == "LOCAL_POSITION_NED":
            ekf["p"] = np.array([m.y, m.x, -m.z]); ekf["v"] = np.array([m.vy, m.vx, -m.vz])
        elif t == "ATTITUDE":
            ekf["yaw"] = math.pi / 2 - m.yaw; ekf["roll"], ekf["pitch"] = m.roll, m.pitch
        elif t == "GPS_RAW_INT":
            ekf["gps_ok"] = m.fix_type >= 3 and m.satellites_visible >= 6
            if ekf["gps_ok"] and "origin" in ekf:
                la0, lo0 = ekf["origin"]
                ekf["gps_p"] = np.array([math.radians(m.lon * 1e-7 - lo0) * 6378137.0 * math.cos(math.radians(la0)),
                                         math.radians(m.lat * 1e-7 - la0) * 6378137.0])
        elif t == "GPS_GLOBAL_ORIGIN":
            ekf["origin"] = (m.latitude * 1e-7, m.longitude * 1e-7)


def ekf_R():
    """EKF attitude (NED/FRD euler) as body FLU -> world ENU."""
    r, p, y = ekf["roll"], ekf["pitch"], math.pi / 2 - ekf["yaw"]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    R_nf = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    M = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float); F = np.diag([1, -1, -1])
    return M @ R_nf @ F


def send(p, R, v, reset):
    M = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float); F = np.diag([1, -1, -1])
    Rn = M @ R @ F
    roll = math.atan2(Rn[2, 1], Rn[2, 2]); pitch = -math.asin(np.clip(Rn[2, 0], -1, 1)); yaw = math.atan2(Rn[1, 0], Rn[0, 0])
    us = int(time.time() * 1e6)
    cov = [float("nan")] * 21
    mav.mav.vision_position_estimate_send(us, p[1], p[0], -p[2], roll, pitch, yaw, cov, reset)
    mav.mav.vision_speed_estimate_send(us, v[1], v[0], -v[2], [float("nan")] * 9, reset)


# ---- wait for sensors + EKF, calibrate IMU orientation -----------------------------
while K[0] is None or not frames or "roll" not in ekf or "p" not in ekf or "origin" not in ekf:
    if "origin" not in ekf:
        mav.mav.command_long_send(mav.target_system, mav.target_component, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                                  mavutil.mavlink.MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN, 0, 0, 0, 0, 0, 0)
    poll_mav(); time.sleep(0.2)
t0 = time.time(); acc = []
while time.time() - t0 < 1.0:
    with lock:
        acc += [a for _, _, a in imu_buf[-50:]]
    time.sleep(0.1)
a_mean = np.mean(acc, 0)
R_imu = np.eye(3) if a_mean[2] > 0 else np.diag([1.0, -1.0, -1.0])       # sensor mounted upright or flipped (roll 180)
print(f"imu accel at rest {a_mean.round(2)} -> {'upright' if a_mean[2] > 0 else 'flipped'} mounting", flush=True)

# state (world ENU, body FLU), initialised from the EKF while GPS is good
R = ekf_R(); p = ekf["p"].copy(); v = np.zeros(3)
A_yaw, A_t = 0.0, np.zeros(3)                  # output alignment: out = Rz(A_yaw) * internal + A_t
lm_pts, lm_xyz = np.zeros((0, 1, 2), np.float32), np.zeros((0, 3))
prev_gray, t_prev, reset_counter = None, None, 0
lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
log, last_print, last_good = [], 0, None
dmin, dmax = vc["depth_range"]
period = 1.0 / vc["process_hz"]
out_dir = sc["out_dir"]; os.makedirs(out_dir, exist_ok=True)
status_path = os.path.join(out_dir, "vio_status.json")


def to_img(m):
    return np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)


def to_depth(m):
    return np.frombuffer(m.data, np.float32).reshape(m.height, m.width)


while True:
    tl = time.time()
    poll_mav()
    with lock:
        common = sorted(set(frames) & set(depths))
        if not common or (t_prev is not None and common[-1] <= t_prev):
            pair = None
        else:
            tf = common[-1]; pair = (frames[tf], depths[tf])
            imu = [x for x in imu_buf if (t_prev is None or x[0] > t_prev) and x[0] <= tf]
    if pair is None:
        time.sleep(0.005); continue
    img, dep = to_img(pair[0]), to_depth(pair[1])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # -- IMU prediction ---------------------------------------------------------------
    dt = 0.0 if t_prev is None else tf - t_prev
    R_pred, v_pred, p_pred = R.copy(), v.copy(), p.copy()
    tp = t_prev
    for (ti, w, a) in imu:
        if tp is None:
            tp = ti; continue
        h = ti - tp; tp = ti
        R_pred = R_pred @ expm((R_imu @ w) * h)
    p_pred = p + v * dt

    # -- track landmarks ----------------------------------------------------------------
    inl_n = 0
    if prev_gray is not None and len(lm_pts):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, lm_pts, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **lk)
        good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((back - lm_pts)[:, 0], axis=1) < 1.0)
        lm_pts, lm_xyz = nxt[good], lm_xyz[good]
    if len(lm_pts) >= vc["min_inliers"]:
        R_cw = (R_pred @ R_bc).T; t_cw = -R_cw @ (p_pred + R_pred @ t_bc)
        rvec, _ = cv2.Rodrigues(R_cw)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(lm_xyz.astype(np.float64), lm_pts.astype(np.float64), K[0], None,
                                                  rvec, t_cw.reshape(3, 1).copy(), useExtrinsicGuess=True,
                                                  iterationsCount=100, reprojectionError=2.0, confidence=0.99,
                                                  flags=cv2.SOLVEPNP_ITERATIVE)
        if ok and inl is not None and len(inl) >= vc["min_inliers"]:
            inl_n = len(inl); keep = np.zeros(len(lm_pts), bool); keep[inl[:, 0]] = True
            lm_pts, lm_xyz = lm_pts[keep], lm_xyz[keep]
            R_cw, _ = cv2.Rodrigues(rvec); R_wc = R_cw.T; p_c = -R_wc @ tvec[:, 0]
            R_vo = R_wc @ R_bc.T; p_vo = p_c - R_vo @ t_bc
            jump = np.linalg.norm(p_vo - p_pred); rjump = math.degrees(np.linalg.norm(logm(R_pred.T @ R_vo)))
            if jump > vc["max_jump"] + vc["max_speed"] * dt or rjump > vc["max_rot_jump_deg"]:
                inl_n = 0                                   # implausible: treat as a lost frame
                lm_pts, lm_xyz = lm_pts[:0], lm_xyz[:0]
            else:
                # complementary fusion: VO corrects the gyro / constant-velocity prediction
                R = R_pred @ expm(0.6 * logm(R_pred.T @ R_vo))
                p_new = p_pred + 0.85 * (p_vo - p_pred)
                if dt > 0:
                    v = 0.7 * v + 0.3 * (p_new - p) / dt
                p = p_new; last_good = tf
    if inl_n == 0:
        R, p, v = R_pred, p_pred, v * 0.8                 # dead-reckon briefly, velocity decays

    # -- replenish landmarks from depth --------------------------------------------------
    if len(lm_pts) < vc["keyframe_min_tracked"]:
        mask = np.full(gray.shape, 255, np.uint8)
        for q in lm_pts[:, 0]:
            cv2.circle(mask, (int(q[0]), int(q[1])), 10, 0, -1)
        new = cv2.goodFeaturesToTrack(gray, vc["max_features"] - len(lm_pts), 0.01, 10, mask=mask)
        if new is not None:
            uv = new[:, 0]; d = dep[np.clip(uv[:, 1].astype(int), 0, dep.shape[0] - 1), np.clip(uv[:, 0].astype(int), 0, dep.shape[1] - 1)]
            ok = np.isfinite(d) & (d > dmin) & (d < dmax)
            uv, d = uv[ok], d[ok]
            Kinv = np.linalg.inv(K[0])
            rays = (Kinv @ np.column_stack([uv, np.ones(len(uv))]).T).T * d[:, None]       # optical frame
            Xw = (R @ R_bc @ rays.T).T + p + R @ t_bc
            lm_pts = np.concatenate([lm_pts, uv[:, None, :].astype(np.float32)])
            lm_xyz = np.concatenate([lm_xyz, Xw])

    # -- align output to the GPS/EKF frame while GPS is good --------------------------------
    gps_ok = ekf.get("gps_ok", False) and "gps_p" in ekf
    healthy = last_good is not None and tf - last_good < vc["unhealthy_after"]
    if gps_ok:
        ref = np.array([*ekf["gps_p"], ekf["p"][2]])
        if not healthy and (last_good is None or tf - last_good > vc["reinit_after"]):
            # lost with GPS available: restart from the GPS / EKF state
            R = Rz(-A_yaw) @ ekf_R(); p = Rz(-A_yaw) @ (ref - A_t); v = np.zeros(3)
            lm_pts, lm_xyz = lm_pts[:0], lm_xyz[:0]; reset_counter = (reset_counter + 1) % 256; last_good = tf
            print(f"t {tf:7.1f}  VIO re-initialised from GPS", flush=True)
        g = vc["align_gain"]
        e_yaw = math.atan2(math.sin(ekf["yaw"] - (yaw_of(R) + A_yaw)), math.cos(ekf["yaw"] - (yaw_of(R) + A_yaw)))
        A_yaw += g * e_yaw
        A_t += g * (ref - (Rz(A_yaw) @ p + A_t))
    Ro, po, vo = Rz(A_yaw) @ R, Rz(A_yaw) @ p + A_t, Rz(A_yaw) @ v
    if healthy:
        send(po, Ro, vo, reset_counter)
    status = {"t": tf, "healthy": bool(healthy), "inliers": int(inl_n), "tracks": int(len(lm_pts)), "gps_ok": bool(gps_ok)}
    with open(status_path + ".tmp", "w") as f:
        json.dump(status, f)
    os.replace(status_path + ".tmp", status_path)

    tr = truth.get("p", np.full(3, np.nan))
    log.append([tf, *po, *tr, *(ekf.get("p", np.full(3, np.nan))), float(gps_ok), inl_n, len(lm_pts)])
    if time.time() - last_print > 5:
        last_print = time.time()
        err = np.linalg.norm(po - tr)
        print(f"t {tf:7.1f}  vio ({po[0]:.1f},{po[1]:.1f},{po[2]:.1f})  err vs truth {err:.2f} m  inliers {inl_n:3d}  "
              f"tracks {len(lm_pts):3d}  gps {'ok' if gps_ok else 'DENIED'}", flush=True)
        np.save(os.path.join(out_dir, "vio_log.npy"), np.array(log))
    prev_gray, t_prev = gray, tf
    time.sleep(max(0.0, period - (time.time() - tl)))
