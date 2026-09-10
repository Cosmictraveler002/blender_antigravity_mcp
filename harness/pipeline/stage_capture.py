"""
Stage: Multi-Viewport Capture & Analysis
========================================
Orchestrates rendering the active 3D model from 14 calibrated viewpoints,
analyzing 360-degree coverage and silhouettes, and generating an annotated
contact sheet montage.
"""

import os
import json
from typing import Dict, Any

from harness.config import HarnessConfig, DEFAULT_CONFIG
from harness.project_loader import ProjectDefinition
from harness.capture.multi_viewport_renderer import MultiViewportRenderer
from harness.capture.contact_sheet_generator import ContactSheetGenerator
from harness.analyzers.viewport_analyzer import ViewportAnalyzer


def run_capture(project: ProjectDefinition, config: HarnessConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """
    Execute Viewport Capture Stage:
      1. Renders 14 viewports via Blender socket.
      2. Analyzes coverage, silhouettes, and cross-angle symmetry.
      3. Compiles a 4x4 visual contact sheet montage.

    Returns:
        Structured stage report dictionary.
    """
    print(f"\n[Stage:Capture] Starting multi-viewport capture for '{project.name}'...")
    os.makedirs(project.viewports_dir, exist_ok=True)
    os.makedirs(project.reports_dir, exist_ok=True)

    # 1. Render all 14 viewports
    renderer = MultiViewportRenderer(config=config)
    manifest = renderer.capture_all(project)

    # 2. Run multi-angle silhouette and coverage analysis
    print("[Stage:Capture] Running 360° silhouette and coverage analysis...")
    analyzer = ViewportAnalyzer()
    report_json_path = os.path.join(project.reports_dir, "viewport_analysis_report.json")
    analysis_results = analyzer.analyze_all(manifest, output_report_path=report_json_path)

    # 3. Assemble 4x4 Contact Sheet Montage
    print("[Stage:Capture] Generating 4x4 visual contact sheet montage...")
    contact_generator = ContactSheetGenerator()
    contact_sheet_path = os.path.join(project.reports_dir, "viewport_contact_sheet.png")
    contact_generator.generate(
        viewport_manifest=manifest,
        output_path=contact_sheet_path,
        reference_image_path=project.reference_image,
        project_name=project.name
    )

    captured = manifest.get("captured_count", 0)
    total = manifest.get("total_viewports", 14)
    health = analysis_results.get("overall_health_score", 0.0)
    warnings = analysis_results.get("defect_warnings", [])

    print(f"\n[Stage:Capture] Complete:")
    print(f"  - Viewports Captured: {captured}/{total}")
    print(f"  - 360° Health Score:  {health:.1f}%")
    print(f"  - Defect Warnings:    {len(warnings)}")
    print(f"  - Contact Sheet:      {contact_sheet_path}")
    print(f"  - Analysis Report:    {report_json_path}")

    return {
        "status": "success" if captured == total else "partial",
        "captured_count": captured,
        "total_viewports": total,
        "manifest": manifest,
        "analysis": analysis_results,
        "contact_sheet_path": contact_sheet_path,
        "report_path": report_json_path,
        "health_score": health
    }
