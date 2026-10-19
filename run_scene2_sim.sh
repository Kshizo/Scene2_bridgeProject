#!/usr/bin/env bash
# Launch the Scene-2 world in Gazebo Harmonic for ArduCopter SITL.
# Usage: ./run_scene2_sim.sh   (then in a 2nd terminal: ./run_scene2_sim.sh sitl)
DIR="$(cd "$(dirname "$0")" && pwd)"
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
TEX="$HOME/.gz/fuel/fuel.gazebosim.org/openrobotics/models/truss bridge/3/materials/textures"  # Fuel mesh uses bare texture filenames
export GZ_SIM_RESOURCE_PATH=$DIR/models:$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$DIR:$TEX:$GZ_SIM_RESOURCE_PATH

if [ "$1" = "sitl" ]; then
  # Home matches the world's <spherical_coordinates> (lat,lon,alt,heading)
  cd "$HOME/ardupilot" && exec sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON \
    --custom-location=47.397971057728974,8.546163739800146,0,0 \
    --add-param-file="$HOME/ardupilot/Tools/autotest/default_params/copter.parm" \
    --add-param-file="$HOME/ardupilot/Tools/autotest/default_params/gazebo-iris.parm" \
    --map --console "${@:2}"
fi
exec gz sim -v4 -r "$DIR/scene2_ardupilot.sdf" "$@"
