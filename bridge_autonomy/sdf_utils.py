"""Read sensor mounting poses from the drone model SDF instead of hard-coding them."""
import math, os, xml.etree.ElementTree as ET
import numpy as np


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def parse_pose(el):
    if el is None or not el.text:
        return np.eye(3), np.zeros(3)
    v = [float(x) for x in el.text.split()]
    if el.get("degrees") == "true":
        v[3:] = [math.radians(a) for a in v[3:]]
    return rpy_to_R(*v[3:6]), np.array(v[:3])


def resource_paths():
    return [p for p in os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(":") if p]


def model_sdf_for_include(world_sdf, include_name):
    root = ET.parse(world_sdf).getroot()
    for inc in root.iter("include"):
        n = inc.find("name")
        if n is not None and n.text.strip() == include_name:
            uri = inc.find("uri").text.strip()
            name = uri.replace("model://", "")
            here = os.path.dirname(os.path.abspath(world_sdf))
            for base in resource_paths() + [here, os.path.join(here, "models")]:
                p = os.path.join(base, name, "model.sdf")
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(f"{uri} not found on GZ_SIM_RESOURCE_PATH")
    raise KeyError(include_name)


def link_pose(model_sdf, link_name):
    """(R, t) of a top-level link in the model frame, and its first <sensor> element."""
    root = ET.parse(model_sdf).getroot()
    for link in root.iter("link"):
        if link.get("name") == link_name:
            R, t = parse_pose(link.find("pose"))
            return R, t, link.find("sensor")
    raise KeyError(f"link {link_name} not in {model_sdf}")
