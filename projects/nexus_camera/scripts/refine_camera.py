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

    def apply_adjustments(self, pass_num: int, recommendations: List[Dict[str, Any]]) -> Any:
        if not recommendations:
            print(f"[Refine Pass {pass_num}] No corrective recommendations provided.")
            return False, []

        bpy_lines = ["import bpy"]
        kp_color = 0.45
        kp_geom = 0.50
        applied_any = False
        applied_params = []

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
                applied_params.append(param)

            # 2. Roughness Micro-Adjustments
            elif "roughness" in param and isinstance(target, (int, float)):
                mat_pattern = param.replace("_roughness", "").lower()
                target_r = float(target)
                bpy_lines.append(f"""
for mat in bpy.data.materials:
    if "{mat_pattern}" in mat.name.lower() or any(p in mat.name.lower() for p in ["magnesium", "rubber", "aluminum", "dial"]):
        if mat.use_nodes and mat.node_tree:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf and "Roughness" in bsdf.inputs:
                cur = bsdf.inputs["Roughness"].default_value
                bsdf.inputs["Roughness"].default_value = min(0.98, max(0.02, cur + ({target_r} - cur) * 0.40))
""")
                applied_any = True
                applied_params.append(param)

            # 3. Metallic Micro-Adjustments
            elif "metallic" in param and isinstance(target, (int, float)):
                target_m = float(target)
                param_clean = param.replace("_metallic", "").lower()
                candidate_mats = []
                if "main_body" in param_clean or "body" in param_clean or "chassis" in param_clean:
                    candidate_mats = ["BodyMagnesium", "AnodizedAluminum"]
                elif "upper" in param_clean or "prism" in param_clean or "pentaprism" in param_clean:
                    candidate_mats = ["BodyMagnesium", "AnodizedAluminum", "DialKnurled"]
                elif "base" in param_clean:
                    candidate_mats = ["BodyMagnesium", "AnodizedAluminum"]
                else:
                    candidate_mats = ["BodyMagnesium"]

                bpy_lines.append(f"""
# Metallic adjustment for {param} ({target_m:.2f}) -> {candidate_mats}
for mat in bpy.data.materials:
    if any(c.lower() in mat.name.lower() for c in {candidate_mats}):
        if mat.use_nodes and mat.node_tree:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf and "Metallic" in bsdf.inputs:
                cur = bsdf.inputs["Metallic"].default_value
                bsdf.inputs["Metallic"].default_value = min(1.0, max(0.0, cur + ({target_m} - cur) * 0.45))
""")
                applied_any = True
                applied_params.append(param)

            # 4. Height Ratio Micro-Adjustments (Vertical span tuning)
            elif "height_ratio" in param:
                h_delta = float(rec.get("target_delta", 0.0))
                param_clean = param.replace("_height_ratio", "").lower()
                target_objs = []
                if "upper" in param_clean:
                    target_objs = ["PentaprismViewfinder", "HotShoeBase", "ModeDial", "ExposureCompensationDial"]
                elif "main_body" in param_clean or "body" in param_clean:
                    target_objs = ["CameraChassis", "HandgripBody"]
                elif "base" in param_clean:
                    target_objs = ["BasePlateTrim", "TripodSocketCollar"]
                else:
                    target_objs = ["CameraChassis"]

                scale_factor = max(0.90, min(1.10, 1.0 + h_delta * kp_geom * 2.0))
                bpy_lines.append(f"""
# Height ratio adjustment for {param} (delta={h_delta:+.3f}) -> {target_objs}
for obj in bpy.data.objects:
    if any(t.lower() in obj.name.lower() for t in {target_objs}):
        obj.scale.z = max(0.1, min(4.0, obj.scale.z * {scale_factor}))
""")
                applied_any = True
                applied_params.append(param)

            # 5. Geometric Micro-Adjustments (Aspect Ratio / Scale)
            elif "aspect_ratio" in param or "scale" in param:
                scale_delta = float(rec.get("target_delta", 0.0))
                if abs(scale_delta) < 0.15:
                    scale_mult = 1.0 + scale_delta * kp_geom
                    bpy_lines.append(f"""
# Micro-scale mesh geometry along XY
for obj in bpy.data.objects:
    if obj.type == 'MESH' and any(kw in obj.name for kw in ['Chassis', 'Grip', 'Lens', 'Pentaprism']):
        obj.scale.x *= {scale_mult}
        obj.scale.y *= {scale_mult}
""")
                    applied_any = True
                    applied_params.append(param)

        if applied_any and len(bpy_lines) > 1:
            code = "\n".join(bpy_lines)
            send_blender_code(code)
            return True, applied_params

        return False, []


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
