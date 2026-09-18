"""
Project Refinement Engine: NEXUS Mirrorless Camera
===================================================
Subclasses BaseRefinementEngine to implement closed-loop proportional tuning
and auto-rebuild escalation based on Stage 3 comparison metrics.
"""

import os
import sys
import json
from typing import Dict, Any, List

from harness.refiners.base_refiner import BaseRefinementEngine
from harness.blender.client import send_blender_code
from harness.utils.color_math import srgb_to_linear


class NexusCameraRefinementEngine(BaseRefinementEngine):
    """Custom closed-loop refiner for NEXUS Mirrorless Camera."""

    def apply_adjustments(self, pass_num: int, recommendations: List[Dict[str, Any]]) -> bool:
        if not recommendations:
            print(f"[Refine Pass {pass_num}] No corrective recommendations provided.")
            return False

        bpy_lines = ["import bpy"]
        kp_color = 0.45
        kp_geom = 0.50
        applied_any = False

        for rec in recommendations:
            param = rec.get("parameter", "")
            target = rec.get("target")

            # 1. Color Micro-Adjustments (CIEDE2000 Proportional Correction)
            if "color" in param and isinstance(target, list) and len(target) == 3:
                mat_pattern = param.replace("_base_color", "").replace("_material_base_color", "").lower()
                target_lin = srgb_to_linear(tuple(target))

                bpy_lines.append(f"""
# Adjust color for material matching '{mat_pattern}'
for mat in bpy.data.materials:
    if "{mat_pattern}" in mat.name.lower() or mat.name.lower() in "{mat_pattern}":
        if mat.use_nodes and mat.node_tree:
            # Check LabelTint node first
            tint_node = mat.node_tree.nodes.get("LabelTint")
            if tint_node and tint_node.inputs and len(tint_node.inputs) > 6:
                cur = list(tint_node.inputs[6].default_value[:3])
                new_r = min(1.0, max(0.0, cur[0] + ({target_lin[0]} - cur[0]) * {kp_color}))
                new_g = min(1.0, max(0.0, cur[1] + ({target_lin[1]} - cur[1]) * {kp_color}))
                new_b = min(1.0, max(0.0, cur[2] + ({target_lin[2]} - cur[2]) * {kp_color}))
                tint_node.inputs[6].default_value = (new_r, new_g, new_b, 1.0)
            else:
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf and "Base Color" in bsdf.inputs:
                    cur = list(bsdf.inputs["Base Color"].default_value[:3])
                    new_r = min(1.0, max(0.0, cur[0] + ({target_lin[0]} - cur[0]) * {kp_color}))
                    new_g = min(1.0, max(0.0, cur[1] + ({target_lin[1]} - cur[1]) * {kp_color}))
                    new_b = min(1.0, max(0.0, cur[2] + ({target_lin[2]} - cur[2]) * {kp_color}))
                    bsdf.inputs["Base Color"].default_value = (new_r, new_g, new_b, 1.0)
""")
                applied_any = True

            # 2. Roughness Micro-Adjustments
            elif "roughness" in param and isinstance(target, (int, float)):
                mat_pattern = param.replace("_roughness", "").lower()
                target_r = float(target)
                bpy_lines.append(f"""
for mat in bpy.data.materials:
    if "{mat_pattern}" in mat.name.lower():
        if mat.use_nodes and mat.node_tree:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf and "Roughness" in bsdf.inputs:
                cur = bsdf.inputs["Roughness"].default_value
                bsdf.inputs["Roughness"].default_value = min(0.98, max(0.02, cur + ({target_r} - cur) * 0.5))
""")
                applied_any = True

            # 3. Geometric Micro-Adjustments (Aspect Ratio / Scale)
            elif "aspect_ratio" in param or "scale" in param:
                scale_delta = float(rec.get("target_delta", 0.0))
                if abs(scale_delta) < 0.05:
                    bpy_lines.append(f"""
# Micro-scale mesh geometry along Z axis
for obj in bpy.data.objects:
    if obj.type == 'MESH' and ('Chassis' in obj.name or 'Grip' in obj.name or 'Lens' in obj.name):
        obj.scale.z *= (1.0 + {scale_delta * kp_geom})
""")
                    applied_any = True

        if applied_any and len(bpy_lines) > 1:
            code = "\n".join(bpy_lines)
            send_blender_code(code)
            return True

        return False


def main():
    geom_json = os.environ.get("HARNESS_GEOM_JSON")
    color_json = os.environ.get("HARNESS_COLOR_JSON")
    reports_dir = os.environ.get("HARNESS_REPORTS_DIR", "outputs/reports")
    renders_dir = os.environ.get("HARNESS_RENDER_DIR", "outputs/renders")
    gen_script = os.environ.get("HARNESS_GENERATE_SCRIPT") or os.path.join(
        os.path.dirname(__file__), "generate_camera.py"
    )
    max_iter = int(os.environ.get("HARNESS_MAX_ITERATIONS", "5"))
    target_score = float(os.environ.get("HARNESS_CONVERGENCE_THRESHOLD", "90.0"))

    engine = NexusCameraRefinementEngine(
        geom_json_path=geom_json,
        color_json_path=color_json,
        reports_dir=reports_dir,
        renders_dir=renders_dir,
        max_iterations=max_iter,
        target_score=target_score,
        generate_script_path=gen_script
    )
    result = engine.run_loop()
    print(f"[Refine] Refinement completed with status: {result.get('status')}")


if __name__ == "__main__":
    main()
