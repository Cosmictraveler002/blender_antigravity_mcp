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
    """

    def __init__(
        self,
        geom_json_path: str,
        color_json_path: str,
        reports_dir: str,
        renders_dir: str,
        max_iterations: int = 6,
        target_score: float = 85.0,
    ):
        self.geom_json_path = os.path.abspath(geom_json_path)
        self.color_json_path = os.path.abspath(color_json_path)
        self.reports_dir = os.path.abspath(reports_dir)
        self.renders_dir = os.path.abspath(renders_dir)
        self.max_iterations = max_iterations
        self.target_score = target_score

        with open(self.geom_json_path, "r", encoding="utf-8") as f:
            self.target_geom = json.load(f)

        self.target_color = {}
        if os.path.exists(self.color_json_path):
            with open(self.color_json_path, "r", encoding="utf-8") as f:
                self.target_color = json.load(f)

        self.iteration_history: List[Dict[str, Any]] = []

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
        print("=" * 76)

        converged = False
        final_score = 0.0

        for pass_num in range(1, self.max_iterations + 1):
            print(f"\n[Refine Pass {pass_num}/{self.max_iterations}] Executing parameter adjustments...")
            recommendations = self.get_latest_recommendations()

            # Apply project-specific mutations
            adjusted = self.apply_adjustments(pass_num, recommendations)
            if not adjusted:
                print(f"[Refine Pass {pass_num}] No further adjustments applied.")

            # Trigger render & comparison
            render_path = self.trigger_render(pass_num)
            comp_result = self.evaluate_render(render_path)

            current_score = comp_result.get("overall_score", 0.0)
            final_score = current_score

            self.iteration_history.append({
                "pass": pass_num,
                "score": current_score,
                "render_path": render_path,
                "recommendations_count": len(comp_result.get("recommendations", [])),
                "timestamp": time.time()
            })

            print(f"[Refine Pass {pass_num}] Fidelity Score: {current_score:.1f}% (Target: {self.target_score:.1f}%)")

            if current_score >= self.target_score:
                print(f"\n[Refine] SUCCESS: Target score reached on pass {pass_num}!")
                converged = True
                break

        log_data = {
            "converged": converged,
            "final_score": final_score,
            "total_passes": len(self.iteration_history),
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
                return data.get("recommendations", [])
            except Exception:
                pass
        return []

    def trigger_render(self, pass_num: int) -> str:
        """Trigger render in Blender and return path."""
        from harness.blender.client import send_blender_code
        render_path = os.path.join(self.renders_dir, f"refine_pass_{pass_num}.png").replace("\\", "/")
        bpy_code = f"""
import bpy
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
            "recommendations": comp_res.get("correction_recommendations", [])
        }
