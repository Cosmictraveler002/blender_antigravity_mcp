"""
Project Refiner: Nexus Mirrorless Camera
Generated from Stage 1 Analyzed Geometry & Material Specification
==================================================================
Target Object Type : camera
Detected Components: ["comp_upper_structure", "comp_main_body", "comp_base_section"]
Detected Materials : ["comp_upper_structure", "comp_main_body", "comp_base_section"]

Subclasses BaseRefinementEngine to implement closed-loop proportional
tuning directly inside Blender based on Stage 3 comparison recommendations.
"""

import os
import sys
import json
from typing import Dict, Any, List

from harness.refiners.base_refiner import BaseRefinementEngine
from harness.blender.client import send_blender_code


class ProjectRefinementEngine(BaseRefinementEngine):
    """
    Custom closed-loop refiner for Nexus Mirrorless Camera.
    Maps comparison discrepancy recommendations to targeted Blender bpy operations.
    """

    def apply_adjustments(self, pass_num: int, recommendations: List[Dict[str, Any]]) -> bool:
        if not recommendations:
            print(f"[Refine Pass {pass_num}] No corrective recommendations provided.")
            return False

        bpy_lines = []
        bpy_lines.append("import bpy")

        # Proportional feedback gain
        kp_geom = 0.35
        kp_color = 0.40

        for rec in recommendations:
            param = rec.get("parameter", "")
            action = rec.get("action", "")
            target = rec.get("target")
            current = rec.get("current")

            # 1. Geometry Scale & Transform Adjustments
            if "scale" in param or "diameter" in param or "size" in param:
                delta = rec.get("target_delta", 0.0)
                if abs(delta) > 1e-4:
                    bpy_lines.append(f"""
# Correction: {action}
for obj in bpy.data.objects:
    if obj.type == "MESH" and "Camera" not in obj.name and "Light" not in obj.name and "Wall" not in obj.name and "Table" not in obj.name:
        obj.scale.x = max(0.1, obj.scale.x + ({delta} * {kp_geom}))
        obj.scale.y = max(0.1, obj.scale.y + ({delta} * {kp_geom}))
        break
""")

            # 2. Material Principled BSDF Color Adjustments
            if "color" in param and isinstance(target, list) and len(target) == 3:
                # Match target material
                mat_pattern = param.replace("_base_color", "").replace("_material_base_color", "")
                bpy_lines.append(f"""
# Material Color Correction for {param}
for mat in bpy.data.materials:
    if "{mat_pattern}" in mat.name.lower() or len(bpy.data.materials) == 1:
        if mat.use_nodes and mat.node_tree:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf and "Base Color" in bsdf.inputs:
                cur = list(bsdf.inputs["Base Color"].default_value[:3])
                new_r = min(1.0, max(0.0, cur[0] + ({target[0]} - cur[0]) * {kp_color}))
                new_g = min(1.0, max(0.0, cur[1] + ({target[1]} - cur[1]) * {kp_color}))
                new_b = min(1.0, max(0.0, cur[2] + ({target[2]} - cur[2]) * {kp_color}))
                bsdf.inputs["Base Color"].default_value = (new_r, new_g, new_b, 1.0)
""")

        if len(bpy_lines) > 1:
            code = "\n".join(bpy_lines)
            send_blender_code(code)
            return True

        return False


def main():
    geom_json = os.environ.get("HARNESS_GEOM_JSON")
    color_json = os.environ.get("HARNESS_COLOR_JSON")
    reports_dir = os.environ.get("HARNESS_REPORTS_DIR", "outputs/reports")
    renders_dir = os.environ.get("HARNESS_RENDER_DIR", "outputs/renders")
    max_iter = int(os.environ.get("HARNESS_MAX_ITERATIONS", "6"))
    target_score = float(os.environ.get("HARNESS_CONVERGENCE_THRESHOLD", "85.0"))

    engine = ProjectRefinementEngine(
        geom_json_path=geom_json,
        color_json_path=color_json,
        reports_dir=reports_dir,
        renders_dir=renders_dir,
        max_iterations=max_iter,
        target_score=target_score
    )
    result = engine.run_loop()
    print(f"[Refine] Finished with converged={result.get('converged')}, score={result.get('final_score')}")


if __name__ == "__main__":
    main()
