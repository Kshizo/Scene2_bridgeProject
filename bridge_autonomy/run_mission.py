#!/usr/bin/env python3
"""Launch a full bridge mission from one config file:
Gazebo (GUI) + ArduCopter SITL (params generated from config + drone model) + GPS-denial
simulator + VIO + lidar mapper (ROS 2 / RViz2) + RViz2 + the mission itself.

Usage: run_mission.py <config.yaml> [--replan] [--no-gui] [--no-rviz]
       run_mission.py <config.yaml> --stop
Logs: <out_dir>/logs/*.log"""
import argparse, os, signal, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bridge_model import load_config
from sdf_utils import model_sdf_for_include, link_pose

ap = argparse.ArgumentParser(); ap.add_argument("config")
ap.add_argument("--replan", action="store_true"); ap.add_argument("--no-gui", action="store_true")
ap.add_argument("--no-rviz", action="store_true"); ap.add_argument("--stop", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config); sc = cfg["scene"]; cfgp = os.path.abspath(a.config)
out = sc["out_dir"]; logs = os.path.join(out, "logs"); os.makedirs(logs, exist_ok=True)
sim_root = os.path.dirname(sc["world_sdf"])
env = dict(os.environ, GZ_PARTITION=sc["gz_partition"])
agz = os.path.expanduser(cfg["sitl"].get("ardupilot_gazebo_dir", "~/ardupilot_gazebo"))
env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = ":".join([os.path.join(agz, "build"), env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")])
env["GZ_SIM_RESOURCE_PATH"] = ":".join([os.path.join(sim_root, "models"), os.path.join(agz, "models"),
                                        os.path.join(agz, "worlds"), sim_root, env.get("GZ_SIM_RESOURCE_PATH", "")])
os.environ.update(env)
PROGS = {"arducopter", "rviz2", "mavproxy.py"}
SCRIPTS = {"sim_vehicle.py", "mavproxy.py", "record_lidar.py", "vio.py", "gps_denial_sim.py", "mission.py", "run_mission.py"}


def ours(argv):
    """True if this process is part of the stack (matched on program / script name, not on text)."""
    prog = os.path.basename(argv[0])
    if prog in PROGS:
        return True
    if prog.startswith("python") and len(argv) > 1:
        script = next((x for x in argv[1:] if not x.startswith("-")), "")
        return os.path.basename(script) in SCRIPTS
    return prog == "gz" and argv[1:3] == ["sim", "-v4"] and sc["world_sdf"] in argv or argv[:3] in (["gz", "sim", "server"], ["gz", "sim", "gui"])


def stop_all():
    me = os.getpid()
    for line in subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]:
        pid, args = line.strip().split(None, 1)
        if int(pid) != me and ours(args.split()):
            try:
                os.kill(int(pid), signal.SIGKILL if "arducopter" in args else signal.SIGTERM)
            except ProcessLookupError:
                pass


if a.stop:
    stop_all(); sys.exit()
procs = []


def start(name, cmd, cwd=None):
    f = open(os.path.join(logs, name + ".log"), "w")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    procs.append(p); print(f"started {name} (log {f.name})", flush=True)
    return p


def wait_log(name, text, timeout=180):
    t = time.time()
    while text not in open(os.path.join(logs, name + ".log"), errors="replace").read():
        if time.time() - t > timeout:
            raise SystemExit(f"{name}: '{text}' not seen in {timeout}s")
        time.sleep(1)


stop_all(); time.sleep(2)
py = sys.executable
if a.replan or not os.path.exists(os.path.join(out, "plan.npz")):
    subprocess.run([py, os.path.join(HERE, "plan_survey.py"), cfgp, "--plot"], check=True)

# SITL params: config + camera offset (FRD body frame) from the drone model
st = cfg["sitl"]
_, t_cam, _ = link_pose(model_sdf_for_include(sc["world_sdf"], sc["vehicle_model"]), cfg["vio"]["camera_link"])
params = dict(st["params"]); params.update(VISO_POS_X=round(t_cam[0], 3), VISO_POS_Y=round(-t_cam[1], 3), VISO_POS_Z=round(-t_cam[2], 3))
parm = os.path.join(out, "sitl_generated.parm")
open(parm, "w").write("".join(f"{k} {v}\n" for k, v in params.items()))

for f in ("map_points.npy", "trajectory.npy", "vio_log.npy"):
    if os.path.exists(os.path.join(out, f)):
        os.remove(os.path.join(out, f))
start("gz", ["gz", "sim", "-v4", "-r", sc["world_sdf"]] + (["-s", "--headless-rendering"] if a.no_gui else []))
wait_log("gz", "Serving world names"); time.sleep(3)
ardu = st["ardupilot_dir"]
cmd = ["sim_vehicle.py", "-v", st["vehicle"], "-f", st["frame"], "--model", "JSON", "--no-rebuild", "-w",
       "--mavproxy-args=--non-interactive"]
cmd += [f"--add-param-file={p}" for p in st["param_files"]] + [f"--add-param-file={parm}"]
cmd += [f"--out={o}" for o in cfg["mavlink"]["sitl_outputs"]]
# SITL home = the world's <spherical_coordinates>, so SITL GPS and Gazebo agree
import xml.etree.ElementTree as ET
_sp = ET.parse(sc["world_sdf"]).getroot().find(".//spherical_coordinates")
if _sp is not None:
    _g = lambda k, d=0.0: float(_sp.findtext(k, d))
    cmd.append(f"--custom-location={_g('latitude_deg')},{_g('longitude_deg')},{_g('elevation')},{_g('heading_deg')}")
start("sitl", cmd, cwd=ardu)
mp = cfg["mapping"]
start("gps_denial", [py, "-u", os.path.join(HERE, "gps_denial_sim.py"), cfgp])
start("vio", [py, "-u", os.path.join(HERE, "vio.py"), cfgp])
rec = [py, "-u", os.path.join(os.path.dirname(HERE), "mapping", "record_lidar.py"), sc["world_name"], out, sc["vehicle_model"],
       "--world-sdf", sc["world_sdf"], "--lidar-link", mp["lidar_link"], "--topic", mp["lidar_topic"]]
if not a.no_rviz:
    rec += ["--ros", "--parts", sc["parts_dir"]]
    start("recorder", ["bash", "-c", "source /opt/ros/jazzy/setup.bash && exec " + " ".join(f"'{x}'" for x in rec)])
    start("rviz", ["bash", "-c", f"source /opt/ros/jazzy/setup.bash && exec rviz2 -d {mp['rviz_config']}"])
else:
    start("recorder", rec)
mission = start("mission", [py, "-u", os.path.join(HERE, "mission.py"), cfgp])
try:
    with open(os.path.join(logs, "mission.log"), errors="replace") as f:
        while mission.poll() is None:
            line = f.readline()
            if line:
                print("[mission]", line, end="", flush=True)
            else:
                time.sleep(0.5)
        print(f.read(), end="")
    print(f"mission exited ({mission.returncode}); map in {out}. Ctrl-C closes everything.", flush=True)
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stop_all()
