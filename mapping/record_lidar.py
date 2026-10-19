#!/usr/bin/env python3
"""Record the iris_mapper lidar into a world-frame point map.

Each /mapper/lidar/points scan is placed in the world with the drone's ground-truth pose
(model pose from /world/<world>/dynamic_pose/info, interpolated to the scan stamp) and
merged into a 3 cm voxel map. Writes <out>/map_points.npy (Nx3 float32, world ENU m) and
<out>/trajectory.npy (t, x, y, z, qx, qy, qz, qw) every few seconds and on Ctrl-C/SIGTERM.

With --ros it also publishes for RViz2 (frame "map" = world ENU):
  /bridge/map   PointCloud2 xyz+rgb, the growing map coloured by bridge component class
  /bridge/scan  PointCloud2, the latest scan in the world frame
  /bridge/path  nav_msgs/Path of the drone, plus TF map -> iris

Usage: GZ_PARTITION=scene2 record_lidar.py <world_name> <out_dir> [model] [--ros --parts <parts_dir>]"""
import os, signal, sys, threading, time
import numpy as np
from gz.transport13 import Node
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
from gz.msgs10.pose_v_pb2 import Pose_V

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("world"); _ap.add_argument("out"); _ap.add_argument("model", nargs="?", default="iris")
_ap.add_argument("--ros", action="store_true"); _ap.add_argument("--parts", default=None)
_ap.add_argument("--world-sdf", default=None, help="read the lidar mounting pose from the drone model")
_ap.add_argument("--lidar-link", default="lidar_link"); _ap.add_argument("--topic", default="/mapper/lidar/points")
_a = _ap.parse_args()
world, out, model = _a.world, _a.out, _a.model
LIDAR_IN_MODEL = np.zeros(3)                    # lidar_link offset in the drone model (from its SDF)
LIDAR_R = np.eye(3)
if _a.world_sdf:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge_autonomy"))
    from sdf_utils import model_sdf_for_include, link_pose
    LIDAR_R, LIDAR_IN_MODEL, _ = link_pose(model_sdf_for_include(os.path.expanduser(_a.world_sdf), model), _a.lidar_link)
    print("lidar mount", LIDAR_IN_MODEL.round(3).tolist(), flush=True)
VOX = 0.03
os.makedirs(out, exist_ok=True)

lock = threading.Lock()
poses = []          # (t, p[3], q[xyzw])
chunks = []         # per-scan voxel-downsampled world points, merged periodically
merged = np.zeros((0, 3), np.float32)
traj = []
scans = 0
latest = {}


def stamp(h):
    return h.stamp.sec + h.stamp.nsec * 1e-9


def on_pose(msg):
    t = stamp(msg.header)
    for p in msg.pose:
        if p.name == model:
            with lock:
                poses.append((t, np.array([p.position.x, p.position.y, p.position.z]),
                              np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])))
                if len(poses) > 4000:
                    del poses[:2000]
            return


def rotmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_at(t):
    with lock:
        if len(poses) < 2 or t < poses[0][0] or t > poses[-1][0] + 0.05:
            return None
        i = next((k for k in range(len(poses) - 1, -1, -1) if poses[k][0] <= t), 0)
        a, b = poses[i], poses[min(i + 1, len(poses) - 1)]
    s = 0.0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
    p = a[1] + s * (b[1] - a[1])
    q = a[2] + s * (b[2] - a[2]) if np.dot(a[2], b[2]) >= 0 else a[2] - s * (b[2] + a[2])
    return p, q / np.linalg.norm(q)


def on_cloud(msg):
    global scans
    t = stamp(msg.header)
    for _ in range(20):                       # pose stream may lag the scan slightly
        pq = pose_at(t)
        if pq is not None:
            break
        time.sleep(0.01)
    if pq is None:
        return
    off = {f.name: f.offset for f in msg.field}
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                   "offsets": [off["x"], off["y"], off["z"]], "itemsize": msg.point_step})
    a = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
    P = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    r = np.linalg.norm(P, axis=1)
    P = P[np.isfinite(r) & (r > 0.6) & (r < 39.5)]
    if not len(P):
        return
    p, q = pq
    R = rotmat(q)
    W = (P @ LIDAR_R.T + LIDAR_IN_MODEL) @ R.T + p
    W = vox_down(W.astype(np.float32))
    with lock:
        chunks.append(W)
        traj.append([t, *p, *q])
        latest["scan"], latest["pose"], latest["t"] = W, (p, q), t
        scans += 1


def vox_down(W):
    k = np.floor(W / VOX).astype(np.int64) + (1 << 20)
    k = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    _, first = np.unique(k, return_index=True)
    return W[first]


def save():
    global merged
    with lock:
        new, chunks[:] = chunks[:], []
        tr = np.array(traj, np.float64)
    merged = vox_down(np.concatenate([merged] + new)) if new else merged
    pts = merged
    np.save(os.path.join(out, "map_points.npy.tmp.npy"), pts)
    os.replace(os.path.join(out, "map_points.npy.tmp.npy"), os.path.join(out, "map_points.npy"))
    np.save(os.path.join(out, "trajectory.npy"), tr)
    return len(pts)


class RosOut:
    """Publishes the map / scan / path / TF to ROS 2 for RViz2."""
    PALETTE = {1: (150, 150, 160), 2: (230, 120, 40), 3: (240, 200, 40), 4: (200, 60, 200), 5: (60, 140, 240),
               6: (120, 80, 40), 7: (170, 110, 80), 8: (240, 240, 240), 9: (250, 40, 40), 10: (40, 220, 220),
               11: (180, 180, 60), 20: (70, 110, 50), 21: (40, 80, 160), 24: (60, 60, 60), 30: (255, 0, 120)}

    def __init__(self, parts_dir):
        import rclpy
        from rclpy.node import Node as RNode
        from sensor_msgs.msg import PointCloud2
        from nav_msgs.msg import Path
        from tf2_ros import TransformBroadcaster
        rclpy.init()
        self.rclpy = rclpy
        self.n = RNode("bridge_mapper")
        self.pub_map = self.n.create_publisher(PointCloud2, "/bridge/map", 1)
        self.pub_scan = self.n.create_publisher(PointCloud2, "/bridge/scan", 1)
        self.pub_path = self.n.create_publisher(Path, "/bridge/path", 1)
        self.tfb = TransformBroadcaster(self.n)
        self.path = Path(); self.path.header.frame_id = "map"
        self.last_scan_t = None
        self.tree = None
        if parts_dir:
            self._load_parts(parts_dir)

    def _load_parts(self, d):
        import json
        from scipy.spatial import cKDTree
        meta = json.load(open(os.path.join(d, "parts.json")))
        rng = np.random.default_rng(0); S, L = [], []
        for p in meta["parts"]:
            V, F = [], []
            for l in open(os.path.join(d, p["name"] + ".obj")):
                t = l.split()
                if t and t[0] == "v": V.append([float(x) for x in t[1:4]])
                elif t and t[0] == "f":
                    ii = [int(x.split("/")[0]) - 1 for x in t[1:]]; F += [[ii[0], ii[j], ii[j + 1]] for j in range(1, len(ii) - 1)]
            if not F:
                continue
            T = np.array(V)[np.array(F)]
            A = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
            n = np.minimum(rng.poisson(A / 0.15 ** 2) + 1, 200000)
            ti = np.repeat(np.arange(len(T)), n); r = rng.random((len(ti), 2)); f = r.sum(1) > 1; r[f] = 1 - r[f]
            q = T[ti, 0] + r[:, :1] * (T[ti, 1] - T[ti, 0]) + r[:, 1:] * (T[ti, 2] - T[ti, 0])
            keep = (np.abs(q[:, 0]) < 120) & (q[:, 1] > -150) & (q[:, 1] < 250)
            S.append(q[keep]); L.append(np.full(keep.sum(), p["label"]))
        self.tree, self.lab = cKDTree(np.concatenate(S)), np.concatenate(L)
        print(f"ros: class colouring from {len(self.lab)} part samples", flush=True)

    def _cloud(self, P, rgb=None):
        from sensor_msgs.msg import PointCloud2, PointField
        msg = PointCloud2(); msg.header.frame_id = "map"; msg.header.stamp = self.n.get_clock().now().to_msg()
        msg.height, msg.width = 1, len(P)
        names = ["x", "y", "z"] + (["rgb"] if rgb is not None else [])
        msg.fields = [PointField(name=k, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, k in enumerate(names)]
        msg.point_step = 4 * len(names); msg.row_step = msg.point_step * len(P); msg.is_dense = True
        buf = np.zeros((len(P), len(names)), np.float32); buf[:, :3] = P
        if rgb is not None:
            packed = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
            buf[:, 3] = packed.view(np.float32)
        msg.data = buf.tobytes()
        return msg

    def publish_map(self, M):
        if not len(M):
            return
        k = np.floor(M / 0.08).astype(np.int64) + (1 << 20)            # thin to 8 cm for display
        _, first = np.unique((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2], return_index=True)
        M = M[first]
        rgb = np.tile(np.array([[110, 110, 110]], np.uint8), (len(M), 1))
        if self.tree is not None:
            d, i = self.tree.query(M, distance_upper_bound=0.25)
            ok = np.isfinite(d)
            pal = np.zeros((256, 3), np.uint8); pal[:] = (110, 110, 110)
            for l, c in self.PALETTE.items():
                pal[l] = c
            rgb[ok] = pal[self.lab[i[ok]]]
        self.pub_map.publish(self._cloud(M, rgb))

    def tick(self):
        from geometry_msgs.msg import PoseStamped, TransformStamped
        with lock:
            W, pq, t = latest.get("scan"), latest.get("pose"), latest.get("t")
        if W is None or t == self.last_scan_t:
            self.rclpy.spin_once(self.n, timeout_sec=0); return
        self.last_scan_t = t
        p, q = pq; now = self.n.get_clock().now().to_msg()
        self.pub_scan.publish(self._cloud(W))
        tf = TransformStamped(); tf.header.stamp = now; tf.header.frame_id = "map"; tf.child_frame_id = "iris"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, p)
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
        self.tfb.sendTransform(tf)
        ps = PoseStamped(); ps.header.frame_id = "map"; ps.header.stamp = now
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
        ps.pose.orientation = tf.transform.rotation
        self.path.poses.append(ps); self.path.header.stamp = now
        self.pub_path.publish(self.path)
        self.rclpy.spin_once(self.n, timeout_sec=0)


stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *a: stop.set())
signal.signal(signal.SIGINT, lambda *a: stop.set())
node = Node()
node.subscribe(Pose_V, f"/world/{world}/dynamic_pose/info", on_pose)
node.subscribe(PointCloudPacked, _a.topic, on_cloud)
print(f"recording {world} -> {out}", flush=True)
ros = RosOut(_a.parts) if _a.ros else None
last = last_map = time.time()
while not stop.is_set():
    time.sleep(0.2)
    if ros:
        ros.tick()
    if time.time() - last > 10:
        n = save(); last = time.time()
        print(f"scans {scans}  map points {n}", flush=True)
    if ros and time.time() - last_map > 2:
        with lock:
            new = list(chunks)
        ros.publish_map(np.concatenate([merged] + new) if new else merged); last_map = time.time()
print(f"final: scans {scans}  map points {save()}", flush=True)
