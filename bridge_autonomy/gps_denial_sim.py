#!/usr/bin/env python3
"""Simulates GPS loss under the bridge: watches the drone's true pose in Gazebo and turns
SITL's GPS off (SIM_GPS1_ENABLE) while it is inside the GPS-denied zone derived from the
bridge geometry (deck footprint + margin, below the deck soffit). Environment side only;
the flight stack never sees this script's truth data.

Usage: gps_denial_sim.py <config.yaml>"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_model import Bridge, load_config
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from pymavlink import mavutil

cfg = load_config(sys.argv[1]); sc = cfg["scene"]
br = Bridge(cfg)
pos = {}


def on_pose(m):
    for p in m.pose:
        if p.name == sc["vehicle_model"]:
            pos["p"] = np.array([p.position.x, p.position.y, p.position.z])


node = Node(); node.subscribe(Pose_V, f"/world/{sc['world_name']}/dynamic_pose/info", on_pose)
mav = mavutil.mavlink_connection(cfg["mavlink"]["gps_sim"], source_system=254)
mav.wait_heartbeat()


def set_gps(on):
    for _ in range(5):
        mav.mav.param_set_send(mav.target_system, mav.target_component, b"SIM_GPS1_ENABLE", 1.0 if on else 0.0,
                               mavutil.mavlink.MAV_PARAM_TYPE_INT8)
        m = mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        if m and m.param_id == "SIM_GPS1_ENABLE" and int(m.param_value) == int(on):
            return True
    return False


state = None
print(f"GPS-denied zone: deck footprint s {np.round(br.s_range,1)} c {np.round(br.c_range,1)} "
      f"(+{cfg['gps_denied']['footprint_margin']} m), below z {br.deck_bottom + cfg['gps_denied']['top_offset']:.2f}", flush=True)
while True:
    if "p" in pos:
        inside = bool(br.in_gps_denied_zone(pos["p"])[0])
        if inside != state:
            ok = set_gps(not inside)
            print(f"{time.strftime('%H:%M:%S')}  drone at {pos['p'].round(1)} -> GPS {'OFF (under bridge)' if inside else 'ON'}"
                  f"{'' if ok else ' (param not acknowledged)'}", flush=True)
            state = inside
    mav.recv_match(blocking=False)
    time.sleep(0.1)
