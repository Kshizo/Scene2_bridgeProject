# Bridge component breakdown — shared spec (all scenes)

Goal: every bridge is split into separate per-component entities so a drone can map it
(semantic segmentation / per-part ground truth) without the terrain, water, trees, cars
etc. being fused into the same mesh.

## Label ids (gz-sim-label-system, also in parts.json)
| id | class           | what goes here                                             |
|----|-----------------|------------------------------------------------------------|
| 1  | deck            | deck slab / road surface on the bridge                     |
| 2  | girder          | longitudinal main beams, box girders, stringers            |
| 3  | cross_beam      | diaphragms, transverse floor beams, bracing                |
| 4  | pier_cap        | pier caps / headstocks / crossheads                        |
| 5  | pier_column     | pier columns and pier walls                                |
| 6  | footing         | pile caps, foundations, pier bases                         |
| 7  | abutment        | abutment walls, wingwalls, approach slabs at bridge ends   |
| 8  | railing         | parapets, barriers, railings, kerbs                        |
| 9  | bearing         | bearings / pads between substructure and superstructure    |
| 10 | arch_truss      | arch ribs, truss members, hangers, cables (if present)     |
| 11 | bridge_misc     | lamps, signs, drains, anything else attached to bridge     |
| 20 | terrain         | ground / landscape                                         |
| 21 | water           | river / sea surface                                        |
| 22 | vegetation      | trees, bushes                                              |
| 23 | vehicle         | cars, buses, people                                        |
| 24 | road            | approach roads / pavements off the bridge                  |
| 25 | other           | buildings, everything else                                 |
| 30 | defect          | crack / spalling / rebar decals (Scene-3 `scene3_defects`) |

## Output layout per scene N (in the project root)
- `sceneN_parts/` : one OBJ per component instance (`<class>_<k>.obj`, e.g. `pier_column_03.obj`),
  shared `parts.mtl` + `textures/`, env pieces as `env_<class>.obj`.
- `sceneN_parts/parts.json` : list of {name, class, label, centroid[3], bbox_min[3], bbox_max[3], tris}.
- `sceneN_ardupilot.sdf` : model `sceneN_bridge` with ONE LINK PER COMPONENT (link name = part name),
  each with visual + collision and a Label plugin inside the visual:
  `<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>ID</label></plugin>`
  Environment in separate static models (`sceneN_terrain`, `sceneN_water`, ...), also labelled.
- Instances numbered along the bridge axis (start → end). Tiny fragments (bolts, slivers)
  merged into their nearest parent part. Aim for tens to ~200 parts, not thousands.

