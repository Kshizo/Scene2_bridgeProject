#!/usr/bin/env python3
"""Score a lidar map against the ground-truth bridge parts.

Samples every part surface (~SPACING m), drops samples nothing could ever see (under the
water, buried in terrain/embankment, or pressed against another face: girder tops under
the deck, column tops under caps ...), and calls a sample covered when a map point lies
within TOL m. Writes coverage.json, labelled map (map_labelled.npz / map.ply) and renders.

Usage: eval_coverage.py <parts_dir> <map_dir> [--water-z 0.0]"""
import argparse, json, os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("parts"); ap.add_argument("map")
ap.add_argument("--water-z", type=float, default=0.0)
ap.add_argument("--spacing", type=float, default=0.10)
ap.add_argument("--tol", type=float, default=0.10)
a = ap.parse_args()
rng = np.random.default_rng(0)

meta = json.load(open(os.path.join(a.parts, "parts.json")))
LABELS = meta["labels"]


def load_obj(p):
    V, F = [], []
    for l in open(p):
        t = l.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            idx = [int(x.split("/")[0]) - 1 for x in t[1:]]
            F += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    V = np.array(V)
    return V[np.array(F)] if F else np.zeros((0, 3, 3))


parts = [p for p in meta["parts"] if not p["name"].startswith("env_")]
env = [p for p in meta["parts"] if p["name"].startswith("env_") and p["class"] != "water"]
TRI = {p["name"]: load_obj(os.path.join(a.parts, p["name"] + ".obj")) for p in parts + env}

# ---- samples ----------------------------------------------------------------------
S, N, OWNER, TID = [], [], [], []
all_tris, tri_owner = [], []
for pi, p in enumerate(parts):
    T = TRI[p["name"]]
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = np.linalg.norm(nrm, axis=1) / 2
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    k = rng.poisson(area / a.spacing ** 2) + (area > 1e-4)
    ti = np.repeat(np.arange(len(T)), k)
    r = rng.random((len(ti), 2)); f = r.sum(1) > 1; r[f] = 1 - r[f]
    t = T[ti]
    S.append(t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0]))
    N.append(nrm[ti]); OWNER.append(np.full(len(ti), pi)); TID.append(ti + sum(len(x) for x in all_tris))
    all_tris.append(T); tri_owner.append(np.full(len(T), pi))
S, N, OWNER, TID = map(np.concatenate, (S, N, OWNER, TID))
ALL = np.concatenate(all_tris + [TRI[p["name"]] for p in env])
n_bridge_tris = sum(len(x) for x in all_tris)
print(f"{len(parts)} parts, {len(S)} surface samples")

hidden = np.zeros(len(S), bool)
why = np.zeros(len(S), "U8")
# 1. under water
m = S[:, 2] < a.water_z + 0.02
hidden |= m; why[m] = "water"
# 2. buried: below the terrain/embankment surface at that xy
E = np.concatenate([TRI[p["name"]] for p in env])
ground = np.full(len(S), -np.inf)
for t in E:
    v0, v1, v2 = t
    d = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if abs(d) < 1e-9:
        continue
    l0 = ((v1[1] - v2[1]) * (S[:, 0] - v2[0]) + (v2[0] - v1[0]) * (S[:, 1] - v2[1])) / d
    l1 = ((v2[1] - v0[1]) * (S[:, 0] - v2[0]) + (v0[0] - v2[0]) * (S[:, 1] - v2[1])) / d
    l2 = 1 - l0 - l1
    inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
    z = l0 * v0[2] + l1 * v1[2] + l2 * v2[2]
    ground = np.where(inside & (z > ground), z, ground)
m = ~hidden & (S[:, 2] < ground - 0.03)
hidden |= m; why[m] = "buried"
# 3. contact: another face lies on this one with opposite normal
tree = cKDTree(S)
for ti, t in enumerate(ALL):
    n = np.cross(t[1] - t[0], t[2] - t[0]); ar = np.linalg.norm(n)
    if ar < 1e-9:
        continue
    n /= ar
    c = t.mean(0); rad = np.linalg.norm(t - c, axis=1).max() + 0.02
    idx = np.array(tree.query_ball_point(c, rad), int)
    if not len(idx):
        continue
    idx = idx[(TID[idx] != ti) & (N[idx] @ n < -0.5)]
    if not len(idx):
        continue
    P = S[idx]
    dist = (P - t[0]) @ n
    idx, P = idx[np.abs(dist) < 0.03], P[np.abs(dist) < 0.03]
    if not len(idx):
        continue
    v0, v1, v2 = t[1] - t[0], t[2] - t[0], P - t[0]
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
    ok = (v >= -0.01) & (w >= -0.01) & (v + w <= 1.01)
    sel = idx[ok & ~hidden[idx]]
    hidden[sel] = True; why[sel] = "contact"
print("excluded:", {k: int((why == k).sum()) for k in ("water", "buried", "contact")})

# ---- coverage ---------------------------------------------------------------------
M = np.load(os.path.join(a.map, "map_points.npy")).astype(np.float64)
mt = cKDTree(M)
d, _ = mt.query(S, distance_upper_bound=a.tol)
cov = np.isfinite(d)
vis = ~hidden
res, by_cls = [], {}
for pi, p in enumerate(parts):
    sel = (OWNER == pi) & vis
    n = int(sel.sum()); c = int((cov & sel).sum())
    res.append({"name": p["name"], "class": p["class"], "visible_samples": n,
                "coverage": round(c / n, 4) if n else None})
    b = by_cls.setdefault(p["class"], [0, 0]); b[0] += n; b[1] += c
tot = int(vis.sum()); totc = int((cov & vis).sum())
summary = {"overall": round(totc / tot, 4), "tolerance_m": a.tol, "sample_spacing_m": a.spacing,
           "map_points": int(len(M)),
           "by_class": {k: round(v[1] / v[0], 4) for k, v in sorted(by_cls.items())},
           "parts": res}
json.dump(summary, open(os.path.join(a.map, "coverage.json"), "w"), indent=1)
print(f"overall coverage {summary['overall']*100:.2f}%  ({totc}/{tot} visible samples)")
for k, v in summary["by_class"].items():
    print(f"  {k:12s} {v*100:6.2f}%")
worst = sorted([r for r in res if r["coverage"] is not None], key=lambda r: r["coverage"])[:8]
print("lowest parts:", ", ".join(f"{r['name']} {r['coverage']*100:.1f}%" for r in worst))

# ---- labelled map + renders ---------------------------------------------------------
st = cKDTree(S)
dd, ii = st.query(M, distance_upper_bound=0.25)
lab = np.where(np.isfinite(dd), np.array([LABELS[parts[o]["class"]] for o in OWNER[np.minimum(ii, len(S) - 1)]]), 0)
lab[(lab == 0) & (np.abs(M[:, 2] - a.water_z) < 0.05)] = LABELS.get("water", 21)
lab[lab == 0] = LABELS.get("terrain", 20)
np.savez_compressed(os.path.join(a.map, "map_labelled.npz"), points=M.astype(np.float32), label=lab.astype(np.uint8))
pal = plt.get_cmap("tab20")
cls_of = {v: k for k, v in LABELS.items()}
col = (np.array([pal(int(l) % 20)[:3] for l in range(64)]) * 255).astype(np.uint8)
with open(os.path.join(a.map, "map.ply"), "w") as o:
    o.write(f"ply\nformat ascii 1.0\nelement vertex {len(M)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar label\nend_header\n")
    for (x, y, z), l in zip(M, lab):
        r, g, b = col[l]; o.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b} {l}\n")

bb = np.array([[min(p["bbox_min"][i] for p in parts) for i in range(3)], [max(p["bbox_max"][i] for p in parts) for i in range(3)]])
inb = np.all((M > bb[0] - 3) & (M < bb[1] + 3), axis=1) & (lab < 20)
B = M[inb]; L = lab[inb]
fig = plt.figure(figsize=(18, 11))
views = [(20, -60, "oblique"), (0, 0, "side (from +x)"), (-25, -120, "from below"), (90, -90, "top")]
for k, (el, az, title) in enumerate(views):
    ax = fig.add_subplot(2, 2, k + 1, projection="3d")
    sub = rng.choice(len(B), min(len(B), 120000), replace=False)
    ax.scatter(*B[sub].T, c=col[L[sub]] / 255, s=0.3)
    ax.view_init(el, az); ax.set_title(title)
    r = (bb[1] - bb[0]).max() / 2; c = (bb[0] + bb[1]) / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r / 3, c[2] + r / 3)
handles = [plt.Line2D([], [], ls="", marker="o", color=col[l] / 255, label=cls_of[l]) for l in sorted(set(L.tolist()))]
fig.legend(handles=handles, loc="lower center", ncol=len(handles))
plt.tight_layout(); plt.savefig(os.path.join(a.map, "map_by_class.png"), dpi=80)

# uncovered visible samples
U = S[vis & ~cov]
fig = plt.figure(figsize=(18, 6))
for k, (el, az, title) in enumerate(views[:3]):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    ax.scatter(*S[vis][::20].T, c="0.8", s=0.2)
    if len(U):
        ax.scatter(*U.T, c="r", s=1.5)
    ax.view_init(el, az); ax.set_title(f"uncovered ({len(U)} samples) - {title}")
    r = (bb[1] - bb[0]).max() / 2; c = (bb[0] + bb[1]) / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r / 3, c[2] + r / 3)
plt.tight_layout(); plt.savefig(os.path.join(a.map, "uncovered.png"), dpi=80)
np.save(os.path.join(a.map, "uncovered_samples.npy"), U.astype(np.float32))
