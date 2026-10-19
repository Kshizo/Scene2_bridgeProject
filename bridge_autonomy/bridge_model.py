"""Bridge geometry derived from a scene's parts.json + part meshes + world SDF.

Nothing here is scene specific: the bridge axis comes from a PCA of the deck, heights from
the part classes, piers from clustering pier columns along the axis, the floor (terrain /
water) from the environment meshes, and the drone spawn from the world SDF.
"""
import json, math, os, re, xml.etree.ElementTree as ET
import numpy as np
import yaml
from scipy import ndimage
from scipy.spatial import cKDTree

SUPER = {"deck", "girder", "cross_beam", "bearing", "railing", "arch_truss"}
SUB = {"pier_cap", "pier_column", "footing", "abutment"}
ENV_SOLID = {"terrain", "road", "vegetation", "vehicle", "other"}


def load_config(path):
    """Load a scene config. Relative paths are resolved against `root` (itself relative to the
    config file's directory; default: the project root two levels up), '~' paths are expanded."""
    path = os.path.abspath(os.path.expanduser(path))
    cfg = yaml.safe_load(open(path))
    root = os.path.normpath(os.path.join(os.path.dirname(path), cfg.get("root", "../..")))
    cfg["root"] = root

    def res(p):
        p = os.path.expanduser(p)
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))
    for k in ("world_sdf", "parts_dir", "out_dir"):
        cfg["scene"][k] = res(cfg["scene"][k])
    if "mapping" in cfg and "rviz_config" in cfg["mapping"]:
        cfg["mapping"]["rviz_config"] = res(cfg["mapping"]["rviz_config"])
    cfg["sitl"]["ardupilot_dir"] = res(cfg["sitl"]["ardupilot_dir"])
    cfg["sitl"]["param_files"] = [res(p) for p in cfg["sitl"]["param_files"]]
    return cfg


def load_obj(path):
    V, F = [], []
    for l in open(path):
        t = l.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            idx = [int(x.split("/")[0]) - 1 for x in t[1:]]
            F += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.array(V)[np.array(F)] if F else np.zeros((0, 3, 3))


def sample_tris(T, spacing, rng):
    if not len(T):
        return np.zeros((0, 3))
    A = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
    n = rng.poisson(A / spacing ** 2) + 1
    ti = np.repeat(np.arange(len(T)), n)
    r = rng.random((len(ti), 2)); f = r.sum(1) > 1; r[f] = 1 - r[f]
    t = T[ti]
    return t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0])


def spawn_pose(world_sdf, model_name):
    """(x, y, z, yaw_rad) of <include><name>model_name</name> in the world."""
    root = ET.parse(world_sdf).getroot()
    for inc in root.iter("include"):
        n = inc.find("name")
        if n is not None and n.text.strip() == model_name:
            pe = inc.find("pose")
            v = [float(x) for x in pe.text.split()]
            if pe.get("degrees") == "true":
                v[3:] = [math.radians(a) for a in v[3:]]
            return np.array(v[:3]), v[5]
    raise KeyError(f"no <include> named {model_name} in {world_sdf}")


class Bridge:
    def __init__(self, cfg, rng_seed=0):
        sc = cfg["scene"]
        self.cfg = cfg
        rng = np.random.default_rng(rng_seed)
        meta = json.load(open(os.path.join(sc["parts_dir"], "parts.json")))
        self.labels = meta["labels"]
        self.parts = meta["parts"]
        self.tris = {p["name"]: load_obj(os.path.join(sc["parts_dir"], p["name"] + ".obj")) for p in self.parts}
        cls = lambda c: [p for p in self.parts if p["class"] == c]
        bridge_parts = [p for p in self.parts if p["class"] in SUPER | SUB]

        # --- axis (PCA of deck vertices in xy) ---------------------------------------
        D = np.concatenate([self.tris[p["name"]].reshape(-1, 3) for p in cls("deck")])
        self.origin = D[:, :2].mean(0)
        w, V = np.linalg.eigh(np.cov((D[:, :2] - self.origin).T))
        self.u = V[:, np.argmax(w)]                      # along the bridge
        if self.u[np.argmax(np.abs(self.u))] < 0:
            self.u = -self.u
        self.v = np.array([-self.u[1], self.u[0]])       # across (left of u)
        sd, cd = self.to_sc(D[:, :2]).T
        self.s_range = (sd.min(), sd.max())
        self.c_range = (np.percentile(cd, 1), np.percentile(cd, 99))
        self.c_mid = sum(self.c_range) / 2
        self.half_width = (self.c_range[1] - self.c_range[0]) / 2

        # --- heights -----------------------------------------------------------------
        self.deck_top = float(np.percentile(D[:, 2], 99))
        self.deck_bottom = float(min(p["bbox_min"][2] for p in cls("deck")))
        soff = [p["bbox_min"][2] for p in self.parts if p["class"] in ("girder", "cross_beam", "deck")]
        self.soffit = float(min(soff))
        rails = cls("railing")
        self.railing_top = float(max([p["bbox_max"][2] for p in rails] + [self.deck_top]))

        # --- floor: highest of terrain / road / water under each xy ------------------
        floor_parts = [p for p in self.parts if p["class"] in ENV_SOLID | {"water"}]
        self._floor_tris = np.concatenate([self.tris[p["name"]] for p in floor_parts]) if floor_parts else np.zeros((0, 3, 3))
        water = cls("water")
        self.water_z = max(p["bbox_max"][2] for p in water) if water else None

        # --- piers: cluster columns along the axis -------------------------------------
        gap = cfg["survey"]["pier_cluster_gap"]
        cols = []
        for p in cls("pier_column"):
            T = self.tris[p["name"]].reshape(-1, 3)
            s, c = self.to_sc(T[:, :2]).T
            cols.append({"name": p["name"], "s": (s.min() + s.max()) / 2, "c": (c.min() + c.max()) / 2,
                         "hs": (s.max() - s.min()) / 2, "hc": (c.max() - c.min()) / 2,
                         "zmin": T[:, 2].min(), "zmax": T[:, 2].max()})
        cols.sort(key=lambda k: k["s"])
        self.piers = []
        for col in cols:
            if self.piers and col["s"] - self.piers[-1]["cols"][-1]["s"] < gap:
                self.piers[-1]["cols"].append(col)
            else:
                self.piers.append({"cols": [col]})
        caps = cls("pier_cap")
        for pr in self.piers:
            cs = pr["cols"]
            pr["s"] = float(np.mean([k["s"] for k in cs]))
            pr["s_lo"] = min(k["s"] - k["hs"] for k in cs); pr["s_hi"] = max(k["s"] + k["hs"] for k in cs)
            pr["c_lo"] = min(k["c"] - k["hc"] for k in cs); pr["c_hi"] = max(k["c"] + k["hc"] for k in cs)
            pr["c"] = (pr["c_lo"] + pr["c_hi"]) / 2
            pr["zmin"] = min(k["zmin"] for k in cs)
            top = [k["zmax"] for k in cs]
            for cp in caps:
                T = self.tris[cp["name"]].reshape(-1, 3); s = self.to_sc(T[:, :2])[:, 0]
                if s.min() - 0.5 <= pr["s"] <= s.max() + 0.5:
                    top.append(T[:, 2].min())
            pr["ztop"] = float(min(top))
            pr["cols"] = sorted(cs, key=lambda k: k["c"])

        # --- abutments + supports ------------------------------------------------------
        self.abutments = []
        for p in cls("abutment"):
            T = self.tris[p["name"]].reshape(-1, 3); s = self.to_sc(T[:, :2])[:, 0]
            # the face toward the spans: the side of the abutment nearer the bridge middle
            mid = sum(self.s_range) / 2
            face = s.max() if abs(s.max() - mid) < abs(s.min() - mid) else s.min()
            self.abutments.append({"name": p["name"], "face_s": float(face), "dir": float(np.sign(mid - face))})
        sup = sorted([a["face_s"] for a in self.abutments] + [p["s"] for p in self.piers])
        self.supports = sup if sup else list(self.s_range)
        self.midspans = [(a + b) / 2 for a, b in zip(self.supports, self.supports[1:])]

        # --- structure samples (for look-at and clearance) ----------------------------
        res = cfg["safety"]["grid_resolution"]
        self.struct_pts = np.concatenate([sample_tris(self.tris[p["name"]], res / 2.5, rng) for p in bridge_parts])
        self.struct_tree = cKDTree(self.struct_pts)
        env_pts = [sample_tris(self.tris[p["name"]], res / 2.5, rng) for p in self.parts
                   if p["class"] in ENV_SOLID and p["name"] in self.tris]
        self.env_pts = np.concatenate(env_pts) if env_pts else np.zeros((0, 3))
        self.spawn, self.spawn_yaw = spawn_pose(sc["world_sdf"], sc["vehicle_model"])

    # --- frames ------------------------------------------------------------------------
    def to_sc(self, xy):
        d = np.atleast_2d(xy) - self.origin
        return np.stack([d @ self.u, d @ self.v], 1)

    def to_xy(self, s, c):
        return self.origin + np.multiply.outer(s, self.u) + np.multiply.outer(c, self.v)

    def sc_point(self, s, c, z):
        x, y = self.to_xy(s, c)
        return np.array([x, y, z])

    # --- floor height --------------------------------------------------------------------
    def floor_at(self, xy):
        xy = np.atleast_2d(xy); out = np.full(len(xy), -np.inf)
        for v0, v1, v2 in self._floor_tris:
            d = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
            if abs(d) < 1e-9:
                continue
            l0 = ((v1[1] - v2[1]) * (xy[:, 0] - v2[0]) + (v2[0] - v1[0]) * (xy[:, 1] - v2[1])) / d
            l1 = ((v2[1] - v0[1]) * (xy[:, 0] - v2[0]) + (v0[0] - v2[0]) * (xy[:, 1] - v2[1])) / d
            l2 = 1 - l0 - l1
            ins = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            z = l0 * v0[2] + l1 * v1[2] + l2 * v2[2]
            out = np.where(ins & (z > out), z, out)
        return out

    # --- GPS-denied zone ---------------------------------------------------------------
    def in_gps_denied_zone(self, P):
        g = self.cfg["gps_denied"]
        P = np.atleast_2d(P)
        s, c = self.to_sc(P[:, :2]).T
        m = g["footprint_margin"]
        return ((s >= self.s_range[0] - m) & (s <= self.s_range[1] + m) &
                (c >= self.c_range[0] - m) & (c <= self.c_range[1] + m) &
                (P[:, 2] < self.deck_bottom + g["top_offset"]))

    def summary(self):
        return {"axis": self.u.round(3).tolist(), "s_range": np.round(self.s_range, 2).tolist(),
                "deck_width": round(2 * self.half_width, 2), "deck_top": round(self.deck_top, 2),
                "soffit": round(self.soffit, 2), "railing_top": round(self.railing_top, 2),
                "water_z": self.water_z, "piers": [{"s": round(p["s"], 2), "columns": len(p["cols"]),
                                                   "top": round(p["ztop"], 2)} for p in self.piers],
                "abutment_faces_s": [round(a["face_s"], 2) for a in self.abutments],
                "spawn": self.spawn.round(2).tolist()}


class Occupancy:
    """Voxel occupancy (structure + terrain + everything below the floor) and clearance field."""

    def __init__(self, br):
        cfg = br.cfg; self.res = r = cfg["safety"]["grid_resolution"]; mg = cfg["safety"]["grid_margin"]
        P = np.concatenate([br.struct_pts, br.env_pts]) if len(br.env_pts) else br.struct_pts
        lo = br.struct_pts.min(0) - mg; hi = br.struct_pts.max(0) + mg
        lo[2] = br.struct_pts[:, 2].min() - 2.0; hi[2] = br.railing_top + mg / 2
        self.lo, self.shape = lo, np.ceil((hi - lo) / r).astype(int) + 1
        occ = np.zeros(self.shape, bool)
        idx = np.floor((P - lo) / r).astype(int)
        ok = np.all((idx >= 0) & (idx < self.shape), axis=1)
        occ[tuple(idx[ok].T)] = True
        # everything at or below the floor (terrain / water surface) is solid
        gx = lo[0] + r * (np.arange(self.shape[0]) + 0.5); gy = lo[1] + r * (np.arange(self.shape[1]) + 0.5)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        fl = br.floor_at(np.stack([X.ravel(), Y.ravel()], 1)).reshape(X.shape)
        fl[~np.isfinite(fl)] = lo[2]
        kz = np.clip(np.floor((fl - lo[2]) / r).astype(int), -1, self.shape[2] - 1)
        zz = np.arange(self.shape[2])[None, None, :]
        occ |= zz <= kz[:, :, None]
        self.occ = occ
        self.clear = ndimage.distance_transform_edt(~occ) * r
        self.floor = fl

    def idx(self, P):
        return np.clip(np.floor((np.atleast_2d(P) - self.lo) / self.res).astype(int), 0, self.shape - 1)

    def clearance(self, P):
        P = np.atleast_2d(P)
        i = np.floor((P - self.lo) / self.res).astype(int)
        inside = np.all((i >= 0) & (i < self.shape), axis=1)
        out = np.full(len(P), 99.0)
        out[inside] = self.clear[tuple(i[inside].T)]
        return out

    def center(self, i):
        return self.lo + (np.asarray(i) + 0.5) * self.res
