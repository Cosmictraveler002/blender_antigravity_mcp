"""
Stage 3: Render vs Reference Comparison
=========================================
Runs the multi-modal comparator on the rendered image against the
design specifications extracted in Stage 1.
"""

import os
import json
from typing import Any, Dict, Optional

from harness.config import HarnessConfig
from harness.project_loader import ProjectDefinition


def run_comparison(project: ProjectDefinition, config: HarnessConfig) -> Dict[str, Any]:
    """
    Execute Stage 3: Compare render against reference specifications.

    Reads:
        - project.renders_dir (latest scene render)
        - project.specs_dir/geometry_design_doc.json
        - project.specs_dir/color_texture_design_doc.json
        - project.specs_dir/geometry_analysis_annotated.png

    Writes to project.reports_dir:
        - comparison_report.json
        - render_geometry_annotated.png
        - comparison_side_by_side.png

    Returns:
        Structured comparison report dict with scores and recommendations.
    """
    from harness.comparators.render_geometry_comparator import RenderGeometryComparator

    # Find the latest render
    render_path = _find_latest_render(project.renders_dir)
    if render_path is None:
        raise FileNotFoundError(
            f"No scene renders found in {project.renders_dir}. Run 'generate' stage first."
        )

    geom_json = os.path.join(project.specs_dir, "geometry_design_doc.json")
    color_json = os.path.join(project.specs_dir, "color_texture_design_doc.json")

    if not os.path.isfile(geom_json):
        raise FileNotFoundError(f"Geometry spec not found: {geom_json}. Run 'analyze' stage first.")

    print(f"[Compare] Render:    {render_path}")
    print(f"[Compare] Geom Spec: {geom_json}")
    print(f"[Compare] Color Spec:{color_json}")

    os.makedirs(project.reports_dir, exist_ok=True)

    comparator = RenderGeometryComparator(
        render_image_path=render_path,
        target_geom_json=geom_json,
        target_color_json=color_json
    )

    print("[Compare] Analyzing rendered 3D scene...")
    rend_data = comparator.analyze_rendered_scene()

    print("[Compare] Executing multi-modal comparison against reference specifications...")
    comp_results = comparator.compare_against_target(rend_data)

    # Annotated render
    annotated_output = os.path.join(project.reports_dir, "render_geometry_annotated.png")
    annotated_path = comparator.generate_annotated_render(rend_data, output_path=annotated_output)
    print(f"[Compare] Annotated render saved: {annotated_path}")

    # Side-by-side diagnostic comparison
    ref_annotated = os.path.join(project.specs_dir, "geometry_analysis_annotated.png")
    side_by_side_output = os.path.join(project.reports_dir, "comparison_side_by_side.png")
    sbs_path = comparator.generate_side_by_side_comparison(
        ref_annotated_path=ref_annotated,
        rend_annotated_path=annotated_output,
        rend_data=rend_data,
        comp_res=comp_results,
        output_path=side_by_side_output
    )
    if sbs_path:
        print(f"[Compare] Visual comparison saved: {sbs_path}")

    total_score = comp_results.get("overall_fidelity", {}).get("total_score_pct", 0.0)
    recommendations = comp_results.get("correction_recommendations", [])

    # Component-Level Fidelity Comparison (using Gemini Vision)
    print("[Compare] Executing multi-angle component fidelity comparison...")
    from harness.comparators.component_fidelity_comparator import ComponentFidelityComparator
    component_comparator = ComponentFidelityComparator()
    
    component_reports = []
    
    # Dynamically find any closeup component renders
    closeup_renders = [
        f for f in os.listdir(project.renders_dir)
        if f.startswith("render_closeup_") and f.endswith(".png")
    ]
    for cref in closeup_renders:
        comp_name = cref.replace("render_closeup_", "").replace(".png", "").replace("_", " ").title()
        cref_path = os.path.join(project.renders_dir, cref)
        try:
            creport = component_comparator.compare(project.reference_image, cref_path, comp_name)
            component_reports.append(creport)
        except Exception as e:
            print(f"[Compare] Component comparison failed for {comp_name}: {e}")
            
    # Add to recommendations
    for creport in component_reports:
        comp_name = creport.get('component_name')
        for rec in creport.get("recommended_actions", []):
            recommendations.append({
                "parameter": f"component_fidelity_{comp_name}",
                "action": rec,
                "target": None,
                "current": None
            })

    report = {
        "rendered_geometry": rend_data,
        "comparison": comp_results,
        "annotated_render_path": annotated_path,
        "side_by_side_path": sbs_path,
        "overall_score": total_score,
        "convergence_status": comp_results.get("convergence_status", {}),
        "correction_recommendations": comp_results.get("correction_recommendations", []),
        "recommendations": recommendations,
        "component_fidelity": component_reports,
    }

    # Write comparison report
    report_json = os.path.join(project.reports_dir, "comparison_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[Compare] Report saved: {report_json}")
    print(f"[Compare] Overall Fidelity Score: {total_score:.1f}%")
    print(f"[Compare] Recommendations: {len(recommendations)} items")

    return report


def _find_latest_render(renders_dir: str) -> Optional[str]:
    """Find the most recently modified scene render PNG in the renders directory."""
    if not os.path.isdir(renders_dir):
        return None

    # Exclude annotated / comparison composite images
    excluded = ["annotated", "side_by_side", "comparison", "swatches"]
    png_files = [
        os.path.join(renders_dir, f)
        for f in os.listdir(renders_dir)
        if f.lower().endswith(".png") and not any(ex in f.lower() for ex in excluded)
    ]

    if not png_files:
        return None

    # Prioritize front or initial render if present
    front_candidates = [
        p for p in png_files
        if "front" in os.path.basename(p).lower() or "initial" in os.path.basename(p).lower()
    ]
    if front_candidates:
        return max(front_candidates, key=os.path.getmtime)

    return max(png_files, key=os.path.getmtime)
