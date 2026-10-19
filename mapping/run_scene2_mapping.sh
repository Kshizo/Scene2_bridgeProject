#!/usr/bin/env bash
# Scene-2 bridge mapping run: Gazebo GUI + ArduCopter SITL + lidar recorder (ROS 2) + RViz2 +
# the smooth autonomous survey. Logs go to $LOG. Ctrl-C (or ./run_scene2_mapping.sh stop) stops all.
# Output map: <repo>/maps/scene2/ (map_points.npy, trajectory.npy); score it afterwards with
#   python3 eval_coverage.py <repo>/scene2_parts <repo>/maps/scene2
HERE="$(cd "$(dirname "$0")" && pwd)"; W="$(dirname "$HERE")"
LOG=${LOG:-/tmp/scene2_mapping}; OUT=${OUT:-$W/maps/scene2}
export GZ_PARTITION=scene2

stop_all() {
  ps -eo pid,args | awk '(/sim_vehicle.py/ || /bin\/arducopter/ || /mavproxy.py/ || /record_lidar.py/ ||
      /fly_scene2_/ || /rviz2/ || /gz sim -v4 -r .*scene2_ardupilot/) && !/awk/ {print $1}' | xargs -r kill 2>/dev/null
}
if [ "$1" = "stop" ]; then stop_all; exit; fi
trap 'stop_all; exit' INT TERM
stop_all; sleep 2
mkdir -p "$LOG" "$OUT"; rm -f "$OUT"/*.npy
source /opt/ros/jazzy/setup.bash

"$W/run_scene2_sim.sh" > "$LOG/gz.log" 2>&1 &
until grep -q 'Serving world names' "$LOG/gz.log" 2>/dev/null; do sleep 1; done; sleep 3
(cd "$HOME/ardupilot" && sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON \
   --custom-location=47.397971057728974,8.546163739800146,0,0 \
   --add-param-file="$HOME/ardupilot/Tools/autotest/default_params/copter.parm" \
   --add-param-file="$HOME/ardupilot/Tools/autotest/default_params/gazebo-iris.parm" \
   --add-param-file="$HERE/sitl_mapping.parm" \
   --out=udp:127.0.0.1:14560 --no-rebuild -w --mavproxy-args="--non-interactive" > "$LOG/sitl.log" 2>&1 < /dev/null) &
python3 -u "$HERE/record_lidar.py" scene2_world "$OUT" --ros --parts "$W/scene2_parts" > "$LOG/rec.log" 2>&1 &
rviz2 -d "$HERE/bridge_map.rviz" > "$LOG/rviz.log" 2>&1 &
python3 -u "$HERE/fly_scene2_smooth.py" 2>&1 | tee "$LOG/fly.log"
echo "survey finished; map in $OUT. Gazebo/RViz2 stay open - Ctrl-C to close everything."
wait
