"""
Stage 5: Final Report Generation
==================================
Generates a comprehensive human-readable report summarizing:
  - Analysis results
  - Comparison metrics
  - Refinement iteration history
  - Pass/fail verdicts per quality dimension
  - Side-by-side visual comparisons
"""

import os
import json
import datetime
from typing import Any, Dict, Optional, List

from harness.config import HarnessConfig
from harness.project_loader import ProjectDefinition


def run_report(project: ProjectDefinition, config: HarnessConfig) -> Dict[str, Any]:
    """
    Execute Stage 5: Generate final quality report.

    Reads:
        - project.specs_dir/master_3d_design_specification.json
        - project.reports_dir/comparison_report.json
        - project.reports_dir/refinement_log.json

    Writes:
        - project.reports_dir/final_report.md
        - project.reports_dir/final_report.json

    Returns:
        Report summary dict.
    """
    convergence = project.convergence or config.convergence
    os.makedirs(project.reports_dir, exist_ok=True)

    # Load available data
    master_spec = _load_json(os.path.join(project.specs_dir, "master_3d_design_specification.json"))
    comparison = _load_json(os.path.join(project.reports_dir, "comparison_report.json"))
    refinement = _load_json(os.path.join(project.reports_dir, "refinement_log.json"))
    viewport_analysis = _load_json(os.path.join(project.reports_dir, "viewport_analysis_report.json"))

    # Build report
    report = {
        "project_name": project.name,
        "project_version": project.version,
        "object_type": project.object_type,
        "timestamp": datetime.datetime.now().isoformat(),
        "stages_completed": [],
        "verdicts": {},
        "summary": {},
    }

    if master_spec:
        report["stages_completed"].append("analyze")
    if viewport_analysis:
        report["stages_completed"].append("capture")
        health = viewport_analysis.get("overall_health_score", 0.0)
        report["verdicts"]["viewport_capture"] = "PASS" if health >= 75.0 else "WARN"
        report["summary"]["viewport_health_score"] = health
        report["summary"]["viewports_analyzed"] = viewport_analysis.get("analyzed_viewports", 0)
    if comparison:
        report["stages_completed"].append("compare")
        overall = comparison.get("overall_score", 0.0)
        report["verdicts"]["comparison"] = "PASS" if overall >= 80.0 else "FAIL"
        report["summary"]["comparison_score"] = overall
    if refinement:
        report["stages_completed"].append("refine")
        report["verdicts"]["refinement"] = "PASS" if refinement.get("converged", False) else "FAIL"
        report["summary"]["geometry_score"] = refinement.get("final_geometry_score", 0.0)
        report["summary"]["color_score"] = refinement.get("final_color_score", 0.0)
        report["summary"]["total_iterations"] = (
            refinement.get("geometry_iterations", 0) + refinement.get("color_iterations", 0)
        )

    # Overall verdict
    all_pass = all(v in ("PASS", "WARN") for v in report["verdicts"].values())
    report["verdicts"]["overall"] = "PASS" if (all_pass and report["verdicts"]) else "FAIL"

    # Write JSON report
    json_path = os.path.join(project.reports_dir, "final_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write Markdown report
    md_path = os.path.join(project.reports_dir, "final_report.md")
    _write_markdown_report(md_path, report, project, comparison, refinement, viewport_analysis)

    print(f"[Report] JSON: {json_path}")
    print(f"[Report] Markdown: {md_path}")
    print(f"[Report] Overall Verdict: {report['verdicts']['overall']}")

    return report


def _load_json(path: str) -> Dict:
    """Safely load a JSON file, returning empty dict if not found."""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_markdown_report(path: str, report: Dict, project: ProjectDefinition,
                           comparison: Dict, refinement: Dict,
                           viewport_analysis: Optional[Dict] = None) -> None:
    """Generate a formatted Markdown report."""
    verdict_emoji = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN"}
    overall = report["verdicts"].get("overall", "N/A")

    lines = [
        f"# Final Quality Report: {project.name}",
        "",
        f"**Generated**: {report['timestamp']}",
        f"**Version**: {project.version}",
        f"**Object Type**: {project.object_type}",
        f"**Overall Verdict**: **{overall}**",
        "",
        "---",
        "",
        "## Stages Completed",
        "",
    ]

    for stage in report["stages_completed"]:
        lines.append(f"- [x] {stage.capitalize()}")

    expected = ["analyze", "generate", "capture", "compare", "refine", "report"]
    for stage in expected:
        if stage not in report["stages_completed"]:
            lines.append(f"- [ ] {stage.capitalize()}")

    lines.extend(["", "---", "", "## Quality Metrics", ""])

    # Multi-Viewport Section
    if viewport_analysis:
        lines.append("### Multi-Viewport 360° Quality Analysis")
        lines.append("")
        health = viewport_analysis.get("overall_health_score", 0.0)
        analyzed_count = viewport_analysis.get("analyzed_viewports", 0)
        lines.append(f"- **360° Health Score**: `{health:.1f}%`")
        lines.append(f"- **Viewports Analyzed**: `{analyzed_count}` angles")

        cross = viewport_analysis.get("cross_viewport_metrics", {})
        if "lateral_symmetry_iou" in cross:
            lines.append(f"- **Lateral Symmetry (Left vs Right IoU)**: `{cross['lateral_symmetry_iou']:.1f}%`")
        if "anterior_posterior_iou" in cross:
            lines.append(f"- **Anterior-Posterior Symmetry (Front vs Back IoU)**: `{cross['anterior_posterior_iou']:.1f}%`")

        # Check for contact sheet
        contact_sheet = os.path.join(project.reports_dir, "viewport_contact_sheet.png")
        if os.path.isfile(contact_sheet):
            lines.extend([
                "",
                "#### 360° Visual Contact Sheet",
                "",
                f"![Multi-Viewport Contact Sheet](viewport_contact_sheet.png)",
                ""
            ])

        # Per-viewport summary table
        per_view = viewport_analysis.get("per_viewport_metrics", {})
        if per_view:
            lines.extend([
                "",
                "| Viewport | Coverage | Aspect Ratio (H/W) | Clipped | Status |",
                "|:---|:---|:---|:---|:---|"
            ])
            for name, m in per_view.items():
                if m.get("status") == "success":
                    cov = f"{m.get('coverage_percentage', 0.0):.1f}%"
                    ar = f"{m.get('aspect_ratio', 0.0):.2f}"
                    clipped = "Yes (Warning)" if m.get("is_clipped") else "No"
                    status = "OK" if not m.get("is_clipped") else "Clipping"
                    lines.append(f"| `{name}` | {cov} | {ar} | {clipped} | {status} |")
                else:
                    lines.append(f"| `{name}` | - | - | - | {m.get('status', 'N/A')} |")
            lines.append("")

        # Defect warnings
        warnings = viewport_analysis.get("defect_warnings", [])
        if warnings:
            lines.extend(["#### Viewport Diagnostics & Warnings", ""])
            for w in warnings:
                lines.append(f"- **[{w.get('severity', 'INFO')}] `{w.get('viewport')}`**: {w.get('issue')} — *{w.get('recommendation')}*")
            lines.append("")

        lines.extend(["---", ""])

    if comparison:
        lines.append("### Comparison Scores")
        lines.append("")
        lines.append("| Metric | Value | Threshold | Verdict |")
        lines.append("|--------|-------|-----------|---------|")

        overall_score = comparison.get("overall_score", 0.0)
        lines.append(f"| Overall Score | {overall_score:.1f}% | >= 80.0% | "
                      f"{'PASS' if overall_score >= 80.0 else 'FAIL'} |")

        if "geometry_score" in comparison:
            gs = comparison["geometry_score"]
            lines.append(f"| Geometry | {gs:.1f}% | >= {project.convergence.geometry_score_threshold * 100 if project.convergence else 85.0:.0f}% | "
                          f"{'PASS' if gs >= 85.0 else 'FAIL'} |")

        if "color_score" in comparison:
            cs = comparison["color_score"]
            lines.append(f"| Color (dE2000) | {cs:.2f} | < {project.convergence.ciede2000_threshold if project.convergence else 3.0:.1f} | "
                          f"{'PASS' if cs < 3.0 else 'FAIL'} |")

        lines.append("")

    if refinement:
        lines.append("### Refinement Results")
        lines.append("")
        lines.append("| Phase | Iterations | Final Score | Converged |")
        lines.append("|-------|-----------|-------------|-----------|")
        lines.append(f"| Geometry | {refinement.get('geometry_iterations', 0)} | "
                      f"{refinement.get('final_geometry_score', 0.0):.2f} | "
                      f"{'Yes' if refinement.get('converged', False) else 'No'} |")
        lines.append(f"| Color | {refinement.get('color_iterations', 0)} | "
                      f"{refinement.get('final_color_score', 0.0):.2f} | "
                      f"{'Yes' if refinement.get('converged', False) else 'No'} |")
        lines.append("")

    if comparison and "recommendations" in comparison:
        lines.extend(["---", "", "## Recommendations", ""])
        for i, rec in enumerate(comparison["recommendations"], 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    lines.extend(["---", "", f"*Report generated by Blender MCP Reconstruction Harness v4.0.0*"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
