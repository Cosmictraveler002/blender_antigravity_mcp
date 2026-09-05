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
