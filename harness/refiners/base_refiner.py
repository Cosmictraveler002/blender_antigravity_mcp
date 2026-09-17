"""
Base Generic Refinement Engine
===============================
Provides an abstract, object-agnostic feedback controller for closed-loop
refinement in Blender.

Responsibilities:
  - Tracks iteration passes, convergence criteria, and score progression
  - Triggers renders and invokes RenderGeometryComparator
  - Delegates object-specific parameter mutations to project scripts
  - Records structured refinement logs (refinement_log.json)
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional


class BaseRefinementEngine:
    """
    Abstract closed-loop controller. Subclasses or project scripts implement
    apply_adjustments(pass_num, recommendations) to mutate specific 3D objects.
    Features automatic procedural re-generation when structural recommendations are encountered.
    """

    def __init__(
        self,
        geom_json_path: str,
        color_json_path: str,
        reports_dir: str,
        renders_dir: str,
        max_iterations: int = 6,
        target_score: float = 85.0,
        generate_script_path: Optional[str] = None,
    ):
        self.geom_json_path = os.path.abspath(geom_json_path)
        self.color_json_path = os.path.abspath(color_json_path)
        self.reports_dir = os.path.abspath(reports_dir)
        self.renders_dir = os.path.abspath(renders_dir)
        self.max_iterations = max_iterations
        self.target_score = target_score

        gen_p = generate_script_path or os.environ.get("HARNESS_GENERATE_SCRIPT")
        self.generate_script_path = os.path.abspath(gen_p) if (gen_p and os.path.exists(gen_p)) else None
        self.auto_rebuild_enabled = True
        self.rebuild_count = 0
        self.max_rebuilds = 2

        with open(self.geom_json_path, "r", encoding="utf-8") as f:
            self.target_geom = json.load(f)

        self.target_color = {}
        if os.path.exists(self.color_json_path):
            with open(self.color_json_path, "r", encoding="utf-8") as f:
                self.target_color = json.load(f)

        self.iteration_history: List[Dict[str, Any]] = []

    def check_structural_rebuild_needed(self, recommendations: List[Dict[str, Any]]) -> bool:
        """Determine if recommendations mandate a clean 3D procedural rebuild."""
        if not self.auto_rebuild_enabled or self.rebuild_count >= self.max_rebuilds:
            return False
        if not self.generate_script_path or not os.path.isfile(self.generate_script_path):
            return False

        for rec in recommendations:
            if rec.get("requires_rebuild") or rec.get("is_structural"):
                return True
            param = rec.get("parameter", "")
            if "scale" in param:
                delta = abs(float(rec.get("target_delta", 0.0)))
                if delta > 0.035:
                    return True
            if "height_ratio" in param:
                delta = abs(float(rec.get("target_delta", 0.0)))
                if delta > 0.015:
                    return True
        return False

    def trigger_rebuild(self, recommendations: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Triggers a clean procedural re-generation in Blender using generate_script.
        """
        if not self.generate_script_path or not os.path.isfile(self.generate_script_path):
            print(f"[Refine Auto-Rebuild] Generation script not configured or not found: {self.generate_script_path}")
            return False

        if self.rebuild_count >= self.max_rebuilds:
            print(f"[Refine Auto-Rebuild] Maximum rebuild quota reached ({self.max_rebuilds}). Proceeding with micro-adjustments.")
            return False

        self.rebuild_count += 1
        print("\n" + ("=" * 76))
        print(f" [Refine Auto-Rebuild] INITIATING PROCEDURAL REBUILD #{self.rebuild_count}/{self.max_rebuilds}")
        print(f" Script: {self.generate_script_path}")
        print("=" * 76)

        with open(self.generate_script_path, "r", encoding="utf-8") as f:
            script_code = f.read()

        from harness.blender.client import send_blender_code
        try:
            if "send_blender_code" in script_code:
                import subprocess
                env = os.environ.copy()
                env["HARNESS_GEOM_JSON"] = self.geom_json_path
                env["HARNESS_COLOR_JSON"] = self.color_json_path
                env["HARNESS_RENDER_DIR"] = self.renders_dir
                env["PYTHONPATH"] = os.getcwd() + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
                proc = subprocess.run(
                    [sys.executable, self.generate_script_path],
                    env=env,
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd()
                )
                if proc.returncode != 0:
                    print(f"[Refine Auto-Rebuild] Rebuild script error: {proc.stderr}")
                    return False
            else:
                # Direct bpy script: dispatch cleanly over active Blender socket
                res = send_blender_code(script_code)
                if res.get("status") == "error":
                    print(f"[Refine Auto-Rebuild] Blender socket error: {res.get('message')}")
                    return False

            print(f"[Refine Auto-Rebuild] Rebuild #{self.rebuild_count} successfully applied in Blender.")
            return True
        except Exception as e:
            print(f"[Refine Auto-Rebuild] Rebuild encountered exception: {e}")
            return False

    def apply_adjustments(self, pass_num: int, recommendations: List[Dict[str, Any]]) -> bool:
        """
        Object-specific adjustment implementation.
        Must be implemented by project-specific refiners (e.g. scripts/refine_<name>.py).
        """
        raise NotImplementedError(
            "apply_adjustments() must be implemented by the project's refinement script."
        )

    def run_loop(self) -> Dict[str, Any]:
        """
        Execute the iterative refinement loop until convergence or max iterations.
        """
        print("=" * 76)
        print(" CLOSED-LOOP REFINEMENT FEEDBACK ENGINE")
        print(f" Target Score: >= {self.target_score:.1f}% | Max Iterations: {self.max_iterations}")
        print(f" Auto-Rebuild: {'ENABLED' if self.generate_script_path else 'DISABLED'}")
        print("=" * 76)

        converged = False
        final_score = 0.0
        recommendations = self.get_latest_recommendations()

        for pass_num in range(1, self.max_iterations + 1):
            print(f"\n[Refine Pass {pass_num}/{self.max_iterations}] Evaluating corrective actions ({len(recommendations)} recs)...")

            # Check for structural rebuild escalation
            if self.check_structural_rebuild_needed(recommendations):
                print(f"[Refine Pass {pass_num}] Structural recommendations detected requiring clean topology reconstruction.")
                rebuilt = self.trigger_rebuild(recommendations)
                if rebuilt:
                    # Capture fresh baseline render and comparison post-rebuild
                    render_path = self.trigger_render(f"{pass_num}_rebuild")
                    comp_result = self.evaluate_render(render_path)
                    current_score = comp_result.get("overall_score", 0.0)
                    recommendations = comp_result.get("recommendations", [])
                    print(f"[Refine Pass {pass_num} Post-Rebuild] Fresh Baseline Score: {current_score:.1f}%")
                    self.iteration_history.append({
                        "pass": f"{pass_num}_rebuild",
                        "score": current_score,
                        "render_path": render_path,
                        "rebuild_cycle": self.rebuild_count,
                        "recommendations_count": len(recommendations),
                        "timestamp": time.time()
                    })
                    conv_status = comp_result.get("convergence_status", {})
                    if (conv_status.get("converged", False) or current_score >= self.target_score):
                        converged = conv_status.get("converged", False)
                        final_score = current_score
                        break

            # Apply project-specific mutations
            adjusted = self.apply_adjustments(pass_num, recommendations)
            if not adjusted:
                print(f"[Refine Pass {pass_num}] No further adjustments applied.")

            # Trigger render & comparison
            render_path = self.trigger_render(pass_num)
            comp_result = self.evaluate_render(render_path)

            current_score = comp_result.get("overall_score", 0.0)
            final_score = current_score
            recommendations = comp_result.get("recommendations", [])

            self.iteration_history.append({
                "pass": pass_num,
                "score": current_score,
                "render_path": render_path,
                "recommendations_count": len(recommendations),
                "timestamp": time.time()
            })

            conv_status = comp_result.get("convergence_status", {})
            is_converged = conv_status.get("converged", False) if conv_status else (current_score >= self.target_score)

            print(f"[Refine Pass {pass_num}] Fidelity Score: {current_score:.1f}% (Target: {self.target_score:.1f}%) | Component Gates: {'PASSED' if is_converged else 'PENDING'}")

            if is_converged and current_score >= self.target_score:
                print(f"\n[Refine] SUCCESS: All component tolerance gates and target score reached on pass {pass_num}!")
                converged = True
                break
            elif current_score >= self.target_score and not is_converged:
                print(f"[Refine Pass {pass_num}] Target score reached ({current_score:.1f}%), but component tolerances pending. Continuing refinement...")

        # Update canonical front render with latest refined pass
        if self.iteration_history:
            latest_pass_render = self.iteration_history[-1].get("render_path")
            if latest_pass_render and os.path.exists(latest_pass_render):
                import shutil
                front_dest = os.path.join(self.renders_dir, "can_front_render.png")
                try:
                    shutil.copyfile(latest_pass_render, front_dest)
                    print(f"[Refine] Updated canonical front render -> {front_dest}")
                except Exception as e:
                    print(f"[Refine] Could not update canonical front render: {e}")

        log_data = {
            "converged": converged,
            "final_score": final_score,
            "total_passes": len(self.iteration_history),
            "rebuilds_executed": self.rebuild_count,
            "history": self.iteration_history
        }

        log_path = os.path.join(self.reports_dir, "refinement_log.json")
        os.makedirs(self.reports_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        return log_data

    def get_latest_recommendations(self) -> List[Dict[str, Any]]:
        """Load recommendations from previous comparison report if available."""
        comp_json = os.path.join(self.reports_dir, "comparison_report.json")
        if os.path.exists(comp_json):
            try:
                with open(comp_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                recs = data.get("comparison", {}).get("correction_recommendations")
                if recs is None:
                    recs = data.get("correction_recommendations", data.get("recommendations", []))
                # Filter out dummy/empty recommendations while keeping structural ones
                return [r for r in recs if r.get("target") is not None or r.get("target_delta") is not None or r.get("target_rgb") is not None or r.get("requires_rebuild") or r.get("is_structural")]
            except Exception:
                pass
        return []

    def trigger_render(self, pass_num: int) -> str:
        """Trigger render in Blender and return path."""
        from harness.blender.client import send_blender_code
        render_path = os.path.join(self.renders_dir, f"refine_pass_{pass_num}.png").replace("\\", "/")
        bpy_code = f"""
import bpy
cam = bpy.data.objects.get('Cam_Front') or bpy.data.objects.get('Camera')
if cam:
    bpy.context.scene.camera = cam
bpy.context.scene.render.resolution_x = 720
bpy.context.scene.render.resolution_y = 1280
bpy.context.scene.render.filepath = '{render_path}'
bpy.ops.render.render(write_still=True)
"""
        send_blender_code(bpy_code)
        return render_path

    def evaluate_render(self, render_path: str) -> Dict[str, Any]:
        """Run comparator on newly rendered pass."""
        from harness.comparators.render_geometry_comparator import RenderGeometryComparator
        comparator = RenderGeometryComparator(
            render_image_path=render_path,
            target_geom_json=self.geom_json_path,
            target_color_json=self.color_json_path
        )
        rend_data = comparator.analyze_rendered_scene()
        comp_res = comparator.compare_against_target(rend_data)
        return {
            "overall_score": comp_res.get("overall_fidelity", {}).get("total_score_pct", 0.0),
            "recommendations": comp_res.get("correction_recommendations", []),
            "convergence_status": comp_res.get("convergence_status", {}),
            "metrics": comp_res.get("metrics", [])
        }
