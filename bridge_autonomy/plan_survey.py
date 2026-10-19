#!/usr/bin/env python3
"""Plan a full inspection/mapping survey for any bridge described by parts.json.

Primitives (all placed from Bridge geometry + config offsets):
  side passes at several heights, a pass over the deck, transverse passes under the deck at
  every mid-span (and extra low levels), sweeps in front of every abutment, a helix around
  every pier, face passes in front of every pier (both sides), and an orbit around every
  column where the clearance allows it.
Every sample is checked against the occupancy clearance field; anything closer than
safety.min_clearance is cut out. Primitives are ordered greedily and joined with A* routes
through free space, smoothed, time-parameterised (speed from clearance) and given a yaw
that looks at the nearest structure. Each sample is flagged if it lies in the GPS-denied zone.

Usage: plan_survey.py <config.yaml> [--plot]      -> <out_dir>/plan.npz (+ plan.png, plan.json)
"""
import argparse, heapq, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_model import Bridge, Occupancy, load_config


def resample(P, step):
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    d = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 1e-6:
        return P[:1]
    t = np.linspace(0, s[-1], max(2, int(s[-1] / step) + 1))
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)


class Planner:
    def __init__(self, cfg):
        self.cfg = cfg
        t = time.time(); self.br = Bridge(cfg); self.occ = Occupancy(self.br)
        print(f"model + occupancy in {time.time()-t:.1f} s; grid {self.occ.shape.tolist()}", flush=True)
        self.minc = cfg["safety"]["min_clearance"]
        self.step = cfg["motion"]["sample_step"]

    # ---- helpers -----------------------------------------------------------------
    def level(self, ref, offset, xy=None):
        br = self.br
        base = {"deck_top": br.deck_top, "soffit": br.soffit, "railing_top": br.railing_top,
                "deck_bottom": br.deck_bottom}.get(ref)
        if ref == "floor":
            base = float(np.max(br.floor_at(xy))) if xy is not None else (br.water_z or 0.0)
        return base + offset

    def valid_pieces(self, P, name):
        """Cut a primitive where it violates clearance; keep pieces longer than 1 m."""
        P = resample(P, self.step)
        ok = self.occ.clearance(P) >= self.minc
        pieces, cur = [], []
        for p, o in zip(P, ok):
            if o:
                cur.append(p)
            elif cur:
                pieces.append(np.array(cur)); cur = []
        if cur:
            pieces.append(np.array(cur))
        pieces = [q for q in pieces if len(q) * self.step > 1.0]
        dropped = 1 - sum(len(q) for q in pieces) / max(len(P), 1)
        return [(name, q) for q in pieces], dropped

    def line_sc(self, s0, c0, s1, c1, z0, z1=None):
        z1 = z0 if z1 is None else z1
        n = max(2, int(math.hypot(s1 - s0, c1 - c0) / self.step))
        s = np.linspace(s0, s1, n); c = np.linspace(c0, c1, n)
        xy = self.br.to_xy(s, c)
        return np.column_stack([xy, np.linspace(z0, z1, n)])

    def circle_sc(self, sc, cc, r, z0, z1, turns, a0=-90.0):
        n = max(8, int(2 * math.pi * r * turns / self.step))
        th = np.radians(a0) + np.linspace(0, 2 * math.pi * turns, n)
        xy = self.br.to_xy(sc + r * np.sin(th), cc + r * np.cos(th))
        return np.column_stack([xy, np.linspace(z0, z1, n)])

    # ---- survey primitives ---------------------------------------------------------
    def primitives(self):
        br, sv = self.br, self.cfg["survey"]
        prims, report = [], {}

        def add(name, P):
            pcs, dropped = self.valid_pieces(P, name)
            prims.extend(pcs); report[name] = round(dropped, 3)

        s0 = min(br.supports) - sv["side_extension"]; s1 = max(br.supports) + sv["side_extension"]
        for side in (-1, 1):
            c = br.c_mid + side * (br.half_width + sv["standoff_side"])
            for lv in sv["side_levels"]:
                xy = br.to_xy(np.linspace(s0, s1, 50), np.full(50, c))
                z = self.level(lv["ref"], lv["offset"], xy)
                add(f"side{'LR'[side > 0]}_{lv['ref']}{lv['offset']:+.1f}", self.line_sc(s0, c, s1, c, z))
        add("top", self.line_sc(s0, br.c_mid, s1, br.c_mid, br.railing_top + sv["top_pass_offset"]))

        cw = br.half_width + sv["standoff_side"]
        for k, sm in enumerate(br.midspans):
            add(f"under_span{k+1}", self.line_sc(sm, br.c_mid - cw, sm, br.c_mid + cw, br.soffit + sv["under_deck_offset"]))
            for h in sv["under_deck_levels_from_floor"]:
                xy = br.to_xy(np.full(20, sm), np.linspace(br.c_mid - cw, br.c_mid + cw, 20))
                add(f"under_span{k+1}_low", self.line_sc(sm, br.c_mid - cw, sm, br.c_mid + cw, self.level("floor", h, xy)))
        for a in br.abutments:
            s = a["face_s"] + a["dir"] * sv["abutment_standoff"]
            xy = br.to_xy(np.full(20, s), np.linspace(br.c_mid - cw, br.c_mid + cw, 20))
            zl = self.level("floor", sv["under_deck_levels_from_floor"][0], xy)
            add(f"{a['name']}_low", self.line_sc(s, br.c_mid - cw, s, br.c_mid + cw, zl))
            add(f"{a['name']}_high", self.line_sc(s, br.c_mid - cw, s, br.c_mid + cw, br.soffit + sv["under_deck_offset"]))
        for k, pr in enumerate(br.piers):
            r = math.hypot(max(pr["s_hi"] - pr["s"], pr["s"] - pr["s_lo"]), (pr["c_hi"] - pr["c_lo"]) / 2) + sv["pier_standoff"]
            xy = br.to_xy(np.full(36, pr["s"]) + r * np.sin(np.linspace(0, 6.3, 36)), pr["c"] + r * np.cos(np.linspace(0, 6.3, 36)))
            zlo = self.level("floor", sv["under_deck_levels_from_floor"][0], xy)
            zhi = min(pr["ztop"], br.soffit) - self.minc
            add(f"pier{k+1}_helix", self.circle_sc(pr["s"], pr["c"], r, zlo, zhi, sv["pier_helix_turns"]))
            for side in (-1, 1):
                s = (pr["s_hi"] if side > 0 else pr["s_lo"]) + side * sv["pier_face_standoff"]
                for z in (zlo, (zlo + zhi) / 2, zhi):
                    add(f"pier{k+1}_face{'-+'[side > 0]}_{z:.1f}", self.line_sc(s, pr["c_lo"] - 2, s, pr["c_hi"] + 2, z))
            for col in pr["cols"]:
                rc = math.hypot(col["hs"], col["hc"]) + sv["column_standoff"]
                add(f"{col['name']}_orbit", self.circle_sc(col["s"], col["c"], rc, zlo, zhi, 1.0))
        return prims, report

    # ---- routing -------------------------------------------------------------------
    def free(self, i):
        return self.occ.clear[tuple(i)] >= self.minc

    def astar(self, a, b):
        occ = self.occ
        ia, ib = tuple(occ.idx(a)[0]), tuple(occ.idx(b)[0])
        if ia == ib:
            return np.array([a, b])
        nb = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
        cost = [math.sqrt(dx * dx + dy * dy + dz * dz) for dx, dy, dz in nb]
        clear = occ.clear; shape = occ.shape; minc = self.minc
        h = lambda i: math.dist(i, ib)
        g = {ia: 0.0}; came = {}; pq = [(h(ia), ia)]; seen = set()
        while pq:
            _, cur = heapq.heappop(pq)
            if cur == ib:
                break
            if cur in seen:
                continue
            seen.add(cur)
            if len(seen) > 3_000_000:
                raise RuntimeError("A* exhausted")
            gc = g[cur]
            for (dx, dy, dz), w in zip(nb, cost):
                n = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                if not (0 <= n[0] < shape[0] and 0 <= n[1] < shape[1] and 0 <= n[2] < shape[2]):
                    continue
                cl = clear[n]
                if cl < minc and n != ib:
                    continue
                ng = gc + w * (1 + 2.0 / max(cl, 0.5))          # prefer roomier space
                if ng < g.get(n, 1e18):
                    g[n] = ng; came[n] = cur; heapq.heappush(pq, (ng + h(n), n))
        if ib not in came and ia != ib:
            raise RuntimeError(f"no route {a} -> {b}")
        path = [ib]
        while path[-1] != ia:
            path.append(came[path[-1]])
        P = np.array([occ.center(i) for i in path[::-1]])
        P[0], P[-1] = a, b
        return self.shortcut(P)

    def los(self, a, b):
        Q = resample(np.array([a, b]), self.occ.res / 2)
        return bool(np.all(self.occ.clearance(Q) >= self.minc))

    def shortcut(self, P):
        out = [P[0]]; i = 0
        while i < len(P) - 1:
            j = len(P) - 1
            while j > i + 1 and not self.los(P[i], P[j]):
                j -= 1
            out.append(P[j]); i = j
        return np.array(out)

    def connect(self, a, b):
        if self.los(a, b):
            return np.array([a, b])
        return self.astar(a, b)

    # ---- full plan -------------------------------------------------------------------
    def plan(self):
        br, mo = self.br, self.cfg["motion"]
        prims, report = self.primitives()
        print(f"{len(prims)} primitive pieces", flush=True)
        start = br.spawn + [0, 0, self.cfg["vehicle"]["takeoff_climb"]]
        if self.occ.clearance(start)[0] < self.minc:
            raise RuntimeError("takeoff point violates clearance - raise vehicle.takeoff_climb")
        path = [resample(np.array([br.spawn + [0, 0, 0.3], start]), self.step)]
        tags = [np.full(len(path[0]), -1)]
        cur = start; todo = list(range(len(prims))); order = []
        while todo:                                          # greedy nearest-end ordering
            best = min([(math.dist(cur, prims[k][1][0]), k, False) for k in todo] +
                       [(math.dist(cur, prims[k][1][-1]), k, True) for k in todo])
            _, k, rev = best; todo.remove(k)
            P = prims[k][1][::-1] if rev else prims[k][1]
            link = resample(self.connect(cur, P[0]), self.step)
            path.append(link); tags.append(np.full(len(link), -1))
            path.append(P); tags.append(np.full(len(P), k)); order.append(prims[k][0])
            cur = P[-1]
        home = br.spawn + [0, 0, self.cfg["vehicle"]["takeoff_climb"]]
        link = resample(self.connect(cur, home), self.step)
        path.append(link); tags.append(np.full(len(link), -1))
        P = np.concatenate(path); T = np.concatenate(tags)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6])
        P, T = P[keep], T[keep]
        clr = self.occ.clearance(P)
        # grid quantisation: route endpoints snap from voxel centres, allow one cell of slack
        bad = (clr < self.minc - self.occ.res) & (np.arange(len(P)) > 5 / self.step)   # takeoff column excepted
        if bad.any():
            raise RuntimeError(f"{bad.sum()} samples violate clearance, worst {clr[bad].min():.2f} m")

        # yaw: look at the nearest structure point, else along the path
        d, i = br.struct_tree.query(P, distance_upper_bound=mo["look_at_radius"])
        tgt = np.where(np.isfinite(d)[:, None], br.struct_pts[np.minimum(i, len(br.struct_pts) - 1)], P + np.gradient(P, axis=0) * 50)
        yaw = np.arctan2(tgt[:, 1] - P[:, 1], tgt[:, 0] - P[:, 0])
        yaw[:int(1.0 / self.step)] = br.spawn_yaw
        yaw = np.unwrap(yaw)
        from scipy.ndimage import uniform_filter1d
        yaw = uniform_filter1d(yaw, int(3.0 / self.step))
        # speed from clearance, turn rate, yaw rate, acceleration
        D = np.linalg.norm(np.diff(P, axis=0), axis=1); S = np.concatenate([[0], np.cumsum(D)])
        f = np.clip((clr - self.minc) / max(mo["slow_clearance"] - self.minc, 1e-3), 0, 1)
        V = mo["v_min"] + f * (mo["v_max"] - mo["v_min"])
        Tg = np.gradient(P, S, axis=0); Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-9
        k = max(1, int(0.5 / self.step)); turn = np.zeros(len(P))
        turn[k:-k] = np.degrees(np.arccos(np.clip(np.sum(Tg[2 * k:] * Tg[:-2 * k], 1), -1, 1)))
        V = np.minimum(V, np.where(turn > 60, 0.3, np.where(turn > 25, 0.6, 99)))
        dyaw = np.abs(np.gradient(yaw, S)) + 1e-6
        V = np.minimum(V, math.radians(mo["yaw_rate_max_deg"]) / dyaw)
        V[0] = V[-1] = 0
        for j in range(1, len(V)):
            V[j] = min(V[j], math.sqrt(V[j - 1] ** 2 + 2 * mo["accel"] * D[j - 1]))
        for j in range(len(V) - 2, -1, -1):
            V[j] = min(V[j], math.sqrt(V[j + 1] ** 2 + 2 * mo["accel"] * D[j]))
        V = np.maximum(V, 0.05)
        t = np.concatenate([[0], np.cumsum(D / ((V[1:] + V[:-1]) / 2))])
        zone = br.in_gps_denied_zone(P)
        info = {"length_m": round(float(S[-1]), 1), "duration_min": round(float(t[-1]) / 60, 1),
                "min_clearance_m": round(float(clr[int(5 / self.step):].min()), 2),
                "gps_denied_fraction": round(float(zone.mean()), 3), "primitives_flown": order,
                "dropped_fraction_per_primitive": report, "bridge": br.summary()}
        return dict(P=P, V=V, T=Tg, yaw=yaw, S=S, t=t, zone=zone, tag=T, clear=clr), info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config); out = cfg["scene"]["out_dir"]; os.makedirs(out, exist_ok=True)
    pl = Planner(cfg)
    plan, info = pl.plan()
    np.savez_compressed(os.path.join(out, "plan.npz"), **plan)
    json.dump(info, open(os.path.join(out, "plan.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ("primitives_flown", "dropped_fraction_per_primitive", "bridge")}))
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        br, P, z = pl.br, plan["P"], plan["zone"]
        sub = br.struct_pts[np.random.default_rng(0).choice(len(br.struct_pts), min(80000, len(br.struct_pts)), replace=False)]
        fig = plt.figure(figsize=(20, 10))
        for k, (el, az) in enumerate([(25, -60), (5, 0), (89, -90)]):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            ax.scatter(*sub.T, s=0.1, c="0.6")
            ax.scatter(*P[~z][::5].T, s=0.3, c="tab:blue"); ax.scatter(*P[z][::5].T, s=0.3, c="tab:red")
            c = (sub.min(0) + sub.max(0)) / 2; r = (sub.max(0) - sub.min(0)).max() / 2 + 10
            ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r / 3, c[2] + r / 3)
            ax.view_init(el, az); ax.set_title("blue: GPS, red: GPS-denied (VIO)")
        plt.tight_layout(); plt.savefig(os.path.join(out, "plan.png"), dpi=70)


if __name__ == "__main__":
    main()
