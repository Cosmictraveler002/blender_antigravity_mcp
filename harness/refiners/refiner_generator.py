"""
Refiner Generator
=================
Generates project-specific closed-loop refinement scripts
(projects/<project>/scripts/refine_<name>.py) directly from the
analyzed geometry and material specifications produced in Stage 1.

Key Contract:
- The base engine in harness/ is 100% object-agnostic.
- There are ZERO hardcoded objects (no "BottleBody", "BambooCap", etc.).
- The refiner script for EVERY project is generated inside the project folder
  upon the report of the image/geometry analyzer.
"""

import os
import sys
import json
import yaml
from typing import Dict, Any, List, Optional

from harness.project_loader import ProjectDefinition


class RefinerGenerator:
    """
    Synthesizes project-specific refiner scripts based on Stage 1 analyzed specifications.
    """

    @classmethod
    def generate_project_refiner(
        cls,
        project: ProjectDefinition,
        output_script_path: Optional[str] = None
    ) -> str:
        """
        Reads analyzed geometry and color specifications from project.specs_dir
        and writes a project-specific refiner to projects/<project>/scripts/refine_<name>.py.
        """
        specs_dir = project.specs_dir
        scripts_dir = os.path.join(project.project_dir, "scripts")
        os.makedirs(scripts_dir, exist_ok=True)

        safe_name = "".join(c if c.isalnum() else "_" for c in project.object_type.lower()).strip("_")
        if not output_script_path:
            output_script_path = os.path.join(scripts_dir, f"refine_{safe_name}.py")

        # 1. Load analyzed schemas
        geom_path = os.path.join(specs_dir, "geometry_design_doc.json")
        color_path = os.path.join(specs_dir, "color_texture_design_doc.json")
        structural_path = os.path.join(specs_dir, "structural_geometry_report.json")
        gemini_path = os.path.join(specs_dir, "gemini_vision_analysis.json")

        geom_doc = {}
        if os.path.exists(geom_path):
            with open(geom_path, "r", encoding="utf-8") as f:
                geom_doc = json.load(f)

        color_doc = {}
        if os.path.exists(color_path):
            with open(color_path, "r", encoding="utf-8") as f:
                color_doc = json.load(f)

        gemini_doc = {}
        if os.path.exists(gemini_path):
            with open(gemini_path, "r", encoding="utf-8") as f:
                gemini_doc = json.load(f)

        structural_doc = {}
        if os.path.exists(structural_path):
            with open(structural_path, "r", encoding="utf-8") as f:
                structural_doc = json.load(f)

        # 2. Extract dynamic components and materials from analyzed report
        detected_components: List[Dict[str, Any]] = []

        # From Gemini vision components
        for c in gemini_doc.get("components", []):
            cid = c.get("component_id", "")
            dname = c.get("display_name", cid)
            cat = c.get("category", "generic")
            detected_components.append({
                "id": cid,
                "display_name": dname,
                "category": cat,
                "target_object_name": dname.replace(" ", "")
            })

        # From Geometry Doc components if empty
        if not detected_components and "components" in geom_doc:
            for k, v in geom_doc["components"].items():
                detected_components.append({
                    "id": k,
                    "display_name": k.replace("_", " ").title(),
                    "category": "mesh",
                    "target_object_name": k.replace("_", " ").title().replace(" ", "")
                })

        # From Structural Geometry primitive segments if still empty
        if not detected_components and "tier2" in structural_doc:
            for s in structural_doc["tier2"].get("primitive_segments", []):
                sid = s["segment_id"]
                ptype = s["primitive_type"]
                detected_components.append({
                    "id": sid,
                    "display_name": f"{ptype} {sid}",
                    "category": ptype.lower(),
                    "target_object_name": sid
                })

        # Dynamic material keys
        detected_materials = list(color_doc.get("component_materials", {}).keys())

        # 3. Generate script code based on BaseRefinementEngine
        script_code = cls._build_script_content(
            project_name=project.name,
            object_type=project.object_type,
            components=detected_components,
            material_keys=detected_materials
        )

        with open(output_script_path, "w", encoding="utf-8") as f:
            f.write(script_code)

        # 4. Update project.yaml manifest if refine_script wasn't set
        rel_script_path = os.path.relpath(output_script_path, project.project_dir).replace("\\", "/")
        cls._register_in_manifest(project, rel_script_path)

        print(f"[RefinerGenerator] Synthesized project refiner: {output_script_path}")
        return output_script_path

    @classmethod
    def _build_script_content(
        cls,
        project_name: str,
        object_type: str,
        components: List[Dict[str, Any]],
        material_keys: List[str]
    ) -> str:
        """Constructs Python source code for the project-specific refiner."""
        comp_summary = json.dumps([c["id"] for c in components])
        mat_summary = json.dumps(material_keys)

        lines = [
            '"""',
            f'Project Refiner: {project_name}',
            f'Generated from Stage 1 Analyzed Geometry & Material Specification',
            '==================================================================',
            f'Target Object Type : {object_type}',
            f'Detected Components: {comp_summary}',
            f'Detected Materials : {mat_summary}',
            '',
            'Subclasses BaseRefinementEngine to implement closed-loop proportional',
            'tuning directly inside Blender based on Stage 3 comparison recommendations.',
            '"""',
            '',
            'import os',
            'import sys',
            'import json',
            'from typing import Dict, Any, List',
            '',
            'from harness.refiners.base_refiner import BaseRefinementEngine',
            'from harness.blender.client import send_blender_code',
            '',
            '',
            'class ProjectRefinementEngine(BaseRefinementEngine):',
            '    """',
            f'    Custom closed-loop refiner for {project_name}.',
            '    Maps comparison discrepancy recommendations to targeted Blender bpy operations.',
            '    """',
            '',
            '    def apply_adjustments(self, pass_num: int, recommendations: List[Dict[str, Any]]) -> bool:',
            '        if not recommendations:',
            '            print(f"[Refine Pass {pass_num}] No corrective recommendations provided.")',
            '            return False',
            '',
            '        bpy_lines = []',
            '        bpy_lines.append("import bpy")',
            '',
            '        # Proportional feedback gain',
            '        kp_geom = 0.35',
            '        kp_color = 0.40',
            '',
            '        for rec in recommendations:',
            '            param = rec.get("parameter", "")',
            '            action = rec.get("action", "")',
            '            target = rec.get("target")',
            '            current = rec.get("current")',
            '',
            '            # 1. Geometry Scale & Transform Adjustments',
            '            if "scale" in param or "diameter" in param or "size" in param:',
            '                delta = rec.get("target_delta", 0.0)',
            '                if abs(delta) > 1e-4:',
            '                    bpy_lines.append(f"""',
            '# Correction: {action}',
            'for obj in bpy.data.objects:',
            '    if obj.type == "MESH" and "Camera" not in obj.name and "Light" not in obj.name and "Wall" not in obj.name and "Table" not in obj.name:',
            '        obj.scale.x = max(0.1, obj.scale.x + ({delta} * {kp_geom}))',
            '        obj.scale.y = max(0.1, obj.scale.y + ({delta} * {kp_geom}))',
            '        break',
            '""")',
            '',
            '            # 2. Material Principled BSDF Color Adjustments',
            '            if "color" in param and isinstance(target, list) and len(target) == 3:',
            '                # Match target material',
            '                mat_pattern = param.replace("_base_color", "").replace("_material_base_color", "")',
            '                bpy_lines.append(f"""',
            '# Material Color Correction for {param}',
            'for mat in bpy.data.materials:',
            '    if "{mat_pattern}" in mat.name.lower() or len(bpy.data.materials) == 1:',
            '        if mat.use_nodes and mat.node_tree:',
            '            bsdf = mat.node_tree.nodes.get("Principled BSDF")',
            '            if bsdf and "Base Color" in bsdf.inputs:',
            '                cur = list(bsdf.inputs["Base Color"].default_value[:3])',
            '                new_r = min(1.0, max(0.0, cur[0] + ({target[0]} - cur[0]) * {kp_color}))',
            '                new_g = min(1.0, max(0.0, cur[1] + ({target[1]} - cur[1]) * {kp_color}))',
            '                new_b = min(1.0, max(0.0, cur[2] + ({target[2]} - cur[2]) * {kp_color}))',
            '                bsdf.inputs["Base Color"].default_value = (new_r, new_g, new_b, 1.0)',
            '""")',
            '',
            '        if len(bpy_lines) > 1:',
            '            code = "\\n".join(bpy_lines)',
            '            send_blender_code(code)',
            '            return True',
            '',
            '        return False',
            '',
            '',
            'def main():',
            '    geom_json = os.environ.get("HARNESS_GEOM_JSON")',
            '    color_json = os.environ.get("HARNESS_COLOR_JSON")',
            '    reports_dir = os.environ.get("HARNESS_REPORTS_DIR", "outputs/reports")',
            '    renders_dir = os.environ.get("HARNESS_RENDER_DIR", "outputs/renders")',
            '    max_iter = int(os.environ.get("HARNESS_MAX_ITERATIONS", "6"))',
            '    target_score = float(os.environ.get("HARNESS_CONVERGENCE_THRESHOLD", "85.0"))',
            '',
            '    engine = ProjectRefinementEngine(',
            '        geom_json_path=geom_json,',
            '        color_json_path=color_json,',
            '        reports_dir=reports_dir,',
            '        renders_dir=renders_dir,',
            '        max_iterations=max_iter,',
            '        target_score=target_score',
            '    )',
            '    result = engine.run_loop()',
            '    print(f"[Refine] Finished with converged={result.get(\'converged\')}, score={result.get(\'final_score\')}")',
            '',
            '',
            'if __name__ == "__main__":',
            '    main()',
            ''
        ]
        return "\n".join(lines)

    @classmethod
    def _register_in_manifest(cls, project: ProjectDefinition, rel_script_path: str):
        """Ensures the generated refiner script is declared in project.yaml."""
        manifest_path = os.path.join(project.project_dir, "project.yaml")
        if not os.path.exists(manifest_path):
            return

        with open(manifest_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if "blender_scripts" not in cfg:
            cfg["blender_scripts"] = {}

        if cfg["blender_scripts"].get("refine") != rel_script_path:
            cfg["blender_scripts"]["refine"] = rel_script_path
            with open(manifest_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            project.refine_script = os.path.join(project.project_dir, rel_script_path)
