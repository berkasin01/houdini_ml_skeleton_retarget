"""
SIMPLE PYTHON SOP - Only applies matches from Overrides
--------------------------------------------------------
This node does NOT run AI - it only reads the Overrides multiparm
and applies the matches to the geometry.

The AI matching is done by the button callback in PythonModule.
This prevents the node from recooking constantly.
"""

import hou
import numpy as np

node = hou.pwd()
geo = node.geometry()

# Get inputs
geo_source = node.inputs()[0].geometry() if node.inputs()[0] else None
geo_target = node.inputs()[1].geometry() if len(node.inputs()) > 1 and node.inputs()[1] else None

if geo_source is None or geo_target is None:
    raise hou.NodeError("Need 2 inputs: Source (input 0), Target (input 1)")

# Get parent HDA
hda = node.parent()

# ============================================================
# READ MATCHES FROM OVERRIDES MULTIPARM
# ============================================================

# Extract skeleton names
source_names = [p.attribValue("name") for p in geo_source.points()]
target_names = [p.attribValue("name") for p in geo_target.points()]

# Build lookups
source_name_to_idx = {name: i for i, name in enumerate(source_names)}
target_name_to_idx = {name: i for i, name in enumerate(target_names)}

# Read overrides
matches = {}

use_overrides = hda.parm("use_overrides")
if use_overrides and use_overrides.evalAsInt():
    num_parm = hda.parm("num_overrides")
    if num_parm:
        num_overrides = num_parm.evalAsInt()

        for i in range(1, num_overrides + 1):
            src_parm = hda.parm(f"override_source{i}")
            tgt_parm = hda.parm(f"override_target{i}")

            if src_parm and tgt_parm:
                src_name = src_parm.evalAsString().strip()
                tgt_name = tgt_parm.evalAsString().strip()

                if src_name and tgt_name and tgt_name != "(none)":
                    s = source_name_to_idx.get(src_name)
                    t = target_name_to_idx.get(tgt_name)

                    if s is not None and t is not None:
                        matches[s] = (t, 1.0)

# ============================================================
# OUTPUT GEOMETRY
# ============================================================

# Add match attributes (if they don't exist)
if not geo.findPointAttrib("matched_name"):
    geo.addAttrib(hou.attribType.Point, "matched_name", "")
if not geo.findPointAttrib("matched_ptnum"):
    geo.addAttrib(hou.attribType.Point, "matched_ptnum", -1)
if not geo.findPointAttrib("match_score"):
    geo.addAttrib(hou.attribType.Point, "match_score", 0.0)

# Set match values on each point
for pt in geo.points():
    s = pt.number()
    if s in matches:
        t, score = matches[s]
        pt.setAttribValue("matched_name", str(target_names[t]))
        pt.setAttribValue("matched_ptnum", int(t))
        pt.setAttribValue("match_score", float(score))
    else:
        pt.setAttribValue("matched_name", "")
        pt.setAttribValue("matched_ptnum", -1)
        pt.setAttribValue("match_score", 0.0)

# Summary
matched_count = len(matches)
total = len(source_names)
if not geo.findGlobalAttrib("match_summary"):
    geo.addAttrib(hou.attribType.Global, "match_summary", "")
geo.setGlobalAttribValue("match_summary", f"Matched {matched_count}/{total} joints")