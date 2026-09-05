"""
Stage 4: Iterative Refinement
===============================
Executes closed-loop correction cycles:
  1. Reads recommendations from Stage 3 comparison report
  2. Executes project-specific refinement script (projects/<project>/scripts/refine_*.py)
  3. Tunes Blender objects, transforms, and shaders
  4. Repeats until convergence thresholds are met or max iterations exhausted.
"""

import os
import sys
import json
import subprocess
from typing import Any, Dict

from harness.config import HarnessConfig
from harness.project_loader import ProjectDefinition


def run_refinement(project: ProjectDefinition, config: HarnessConfig) -> Dict[str, Any]:
    """
    Execute Stage 4: Iterative refinement loop.

    Reads:
        - project.specs_dir/geometry_design_doc.json
        - project.specs_dir/color_texture_design_doc.json
        - project.reports_dir/comparison_report.json
        - project.refine_script

    Writes:
        - Progressively improved renders to project.renders_dir/
        - Iteration logs to project.reports_dir/refinement_log.json

    Returns:
        Refinement result with final scores and iteration count.
    """
    convergence = project.convergence or config.convergence

    geom_json = os.path.join(project.specs_dir, "geometry_design_doc.json")
    color_json = os.path.join(project.specs_dir, "color_texture_design_doc.json")
    comparison_json = os.path.join(project.reports_dir, "comparison_report.json")
    log_path = os.path.join(project.reports_dir, "refinement_log.json")

    print("=" * 76)
    print(" CLOSED-LOOP REFINEMENT STAGE (Stage 4)")
    print(f" Project:            {project.name}")
    print(f" Max iterations:     {convergence.max_iterations}")
    print(f" Target fidelity:    >= {convergence.geometry_score_threshold * 100:.1f}%")
    print(f" CIEDE2000 target:   < {convergence.ciede2000_threshold}")
    print("=" * 76)

    # Check if project refinement script is configured or synthesize from analyzed schema
    refine_script = project.refine_script
    if not refine_script or not os.path.isfile(refine_script):
        from harness.refiners.refiner_generator import RefinerGenerator
        print(f"[Refine] Synthesizing project refinement script from Stage 1 analyzed schema...")
        refine_script = RefinerGenerator.generate_project_refiner(project)
        project.refine_script = refine_script

    if refine_script and os.path.isfile(refine_script):
        print(f"[Refine] Executing project refinement script: {refine_script}")

        env = os.environ.copy()
        env["HARNESS_SPEC_DIR"] = project.specs_dir
        env["HARNESS_GEOM_JSON"] = geom_json
        env["HARNESS_COLOR_JSON"] = color_json
        env["HARNESS_REPORTS_DIR"] = project.reports_dir
        env["HARNESS_RENDER_DIR"] = project.renders_dir
        env["HARNESS_TEXTURE_DIR"] = project.textures_dir
        env["HARNESS_COMPARISON_JSON"] = comparison_json
        env["HARNESS_MAX_ITERATIONS"] = str(convergence.max_iterations)
        env["HARNESS_CONVERGENCE_THRESHOLD"] = str(convergence.geometry_score_threshold * 100)
        env["PYTHONPATH"] = os.getcwd() + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

        proc = subprocess.run(
            [sys.executable, refine_script],
            env=env,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        if proc.stdout:
            print(proc.stdout.strip())
        if proc.returncode != 0:
            print(proc.stderr)
            raise RuntimeError(f"Refinement script failed with code {proc.returncode}: {proc.stderr}")

        # Read generated log if available
        if os.path.isfile(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            return results
    else:
        print(f"[Refine] NOTICE: No refinement script found for project '{project.name}'.")
        print("  Pipeline Requirement: Following Stage 1 analysis and Stage 3 comparison recommendations,")
        print(f"  the AI Agent/LLM generates 'projects/{os.path.basename(project.project_dir)}/scripts/refine_<name>.py'.")
        print("  See template at: harness/refiners/template_refiner.py")

    # Fallback status structure
    fallback_results = {
        "converged": False,
        "final_score": 0.0,
        "total_passes": 0,
        "history": [],
        "message": "Refinement script executed or awaiting generation"
    }

    if not os.path.isfile(log_path):
        os.makedirs(project.reports_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(fallback_results, f, indent=2)

    return fallback_results
