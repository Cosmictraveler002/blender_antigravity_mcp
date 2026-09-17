"""
Stage 2: Blender Model Generation
===================================
Executes the project's Blender generation script to construct the 3D model
from the master design specification, then renders an initial image.
"""

import os
import sys
import json
import subprocess
from typing import Any, Dict

from harness.config import HarnessConfig
from harness.project_loader import ProjectDefinition
from harness.blender.client import send_blender_code


def run_generation(project: ProjectDefinition, config: HarnessConfig) -> Dict[str, Any]:
    """
    Execute Stage 2: Build the 3D model in Blender.

    Reads:
        - project.specs_dir/master_3d_design_specification.json
        - project.generate_script

    Produces:
        - 3D model constructed in active Blender scene
        - Initial render saved to project.renders_dir/initial_render.png

    Returns:
        Dict with generation status and render path.
    """
    master_json = os.path.join(project.specs_dir, "master_3d_design_specification.json")
    if not os.path.isfile(master_json):
        raise FileNotFoundError(
            f"Master spec not found: {master_json}. Run 'analyze' stage first."
        )

    print(f"[Generate] Loading master specification: {master_json}")
    with open(master_json, "r", encoding="utf-8") as f:
        master_spec = json.load(f)

    # -------------------------------------------------------------
    # Stage 2.1: Parametric 2D Graphic Surface & Packaging Art (PIPELINE_SOLUTIONS_SPEC.md § 6)
    # -------------------------------------------------------------
    try:
        # Check if project has dedicated high-resolution label generation script
        custom_label_script = None
        for cand in ["generate_label_hires.py", "generate_label.py"]:
            cand_p = os.path.join(project.project_dir, "scripts", cand)
            if os.path.isfile(cand_p):
                custom_label_script = cand_p
                break

        has_custom_label = False
        if custom_label_script:
            print(f"[Generate] Stage 2.1: Executing project high-resolution label script: {custom_label_script}")
            try:
                subprocess.run([sys.executable, custom_label_script], check=True, cwd=project.project_dir)
                has_custom_label = True
            except Exception as e_script:
                print(f"[Generate] Notice: Custom label script execution returned: {e_script}")

        from harness.generators.graphic_synthesizer import GraphicArtSynthesizer
        synth = GraphicArtSynthesizer()
        typo_manifest = master_spec.get("gemini_vision", {}).get("typography_and_labels", [])
        geom_comps = master_spec.get("geometry", {}).get("components", {})

        for cid, cdata in geom_comps.items():
            cat = cdata.get("category", "")
            # Synthesize if component is a substrate, decal layer, or has typography
            if cat in ["substrate", "decal_layer"] or len(typo_manifest) > 0:
                sub_dir = os.path.join(project.textures_dir, cid)
                base_color = cdata.get("pbr_material", {}).get("base_color_hex") or cdata.get("color_hex", "#F0F0F0")
                roughness = float(cdata.get("pbr_material", {}).get("roughness") or cdata.get("estimated_roughness", 0.45))
                manifest = synth.synthesize_packaging(
                    component_id=cid,
                    geometry_spec=cdata,
                    typography_manifest=typo_manifest,
                    output_dir=sub_dir,
                    reference_image_path=project.reference_image,
                    substrate_color_hex=base_color,
                    substrate_roughness=roughness
                )
                print(f"[Generate] Stage 2.1: Baked packaging textures for '{cid}' -> {sub_dir}")

                # Sync to project root textures/ only if project does NOT have a dedicated vector label script
                if not has_custom_label:
                    proj_tex_root = os.path.join(project.project_dir, "textures")
                    if os.path.isdir(proj_tex_root):
                        import shutil
                        for tex_file in ["label_diffuse.png", "label_roughness.png", "label_normal.png"]:
                            src = os.path.join(sub_dir, tex_file)
                            dst = os.path.join(proj_tex_root, tex_file)
                            if os.path.isfile(src):
                                try:
                                    shutil.copyfile(src, dst)
                                except Exception:
                                    pass
    except Exception as e:
        print(f"[Generate] Stage 2.1 Packaging art synthesis note: {e}")

    # If the project has a custom generation script, execute it
    if project.generate_script and os.path.isfile(project.generate_script):
        print(f"[Generate] Executing project script: {project.generate_script}")

        with open(project.generate_script, "r", encoding="utf-8") as f:
            script_content = f.read()

        env = os.environ.copy()
        env["HARNESS_SPEC_DIR"] = project.specs_dir
        env["HARNESS_GEOM_JSON"] = os.path.join(project.specs_dir, "geometry_design_doc.json")
        env["HARNESS_COLOR_JSON"] = os.path.join(project.specs_dir, "color_texture_design_doc.json")
        env["HARNESS_RENDER_DIR"] = project.renders_dir
        env["HARNESS_TEXTURE_DIR"] = project.textures_dir
        env["PYTHONPATH"] = os.getcwd() + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

        # Check if the script is a host-side client runner (uses send_blender_code)
        # or a direct Blender bpy script
        if "send_blender_code" in script_content:
            proc = subprocess.run(
                [sys.executable, project.generate_script],
                env=env,
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            if proc.stdout:
                print(proc.stdout.strip())
            if proc.returncode != 0:
                print(proc.stderr)
                raise RuntimeError(
                    f"Project generation script failed with code {proc.returncode}: {proc.stderr}"
                )
        else:
            # Direct bpy script: send directly to Blender socket
            print(f"[Generate] Sending code to Blender ({config.blender.host}:{config.blender.port})...")
            response = send_blender_code(
                script_content,
                host=config.blender.host,
                port=config.blender.port
            )
            print(f"[Generate] Blender response status: {response.get('status', 'unknown')}")
            if response.get("status") == "error":
                raise RuntimeError(f"Blender generation failed: {response.get('message', 'unknown error')}")
    else:
        print("[Generate] WARNING: No generation script configured in project.yaml")
        print("[Generate] Skipping model construction. Set blender_scripts.generate in project.yaml.")

    render_path = os.path.join(project.renders_dir, "initial_render.png").replace("\\", "/")

    result = {
        "status": "complete",
        "render_path": render_path,
    }
    return result
