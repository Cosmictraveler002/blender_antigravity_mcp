import os
import json
import pytest
import tempfile
from typing import List, Dict, Any

from harness.refiners.base_refiner import BaseRefinementEngine
from harness.comparators.render_geometry_comparator import RenderGeometryComparator
from harness.analyzers.gemini_analyzer import GeminiVisionAnalyzer
from harness.pipeline.stage_report import _write_markdown_report, run_report
from harness.project_loader import ProjectDefinition
from harness.config import HarnessConfig


class DummyStagnatingRefiner(BaseRefinementEngine):
    def __init__(self, geom_path, color_path, reports_dir, renders_dir, max_iterations=6, target_score=90.0):
        super().__init__(
            geom_json_path=geom_path,
            color_json_path=color_path,
            reports_dir=reports_dir,
            renders_dir=renders_dir,
            max_iterations=max_iterations,
            target_score=target_score,
            generate_script_path=None
        )
        self.apply_calls = 0

    def trigger_render(self, pass_num: int) -> str:
        render_path = os.path.join(self.renders_dir, f"dummy_pass_{pass_num}.png")
        with open(render_path, "w") as f:
            f.write("mock_render")
        return render_path

    def evaluate_render(self, render_path: str) -> Dict[str, Any]:
        return {
            "overall_score": 45.18,
            "geometry_score": 50.0,
            "color_score": 40.0,
            "texture_score": 45.0,
            "recommendations": [
                {
                    "recommendation_id": "comp_main_body_metallic:0.0",
                    "parameter": "comp_main_body_metallic",
                    "current": 0.95,
                    "target": 0.0,
                    "target_delta": -0.95,
                    "priority": "HIGH"
                }
            ],
            "convergence_status": {"converged": False},
            "metrics": []
        }

    def apply_adjustments(self, pass_num: int, recommendations: List[Dict[str, Any]]):
        self.apply_calls += 1
        return True, ["comp_main_body_metallic"]


def test_stagnation_detection_and_early_exit(tmp_path):
    geom_file = tmp_path / "geom.json"
    color_file = tmp_path / "color.json"
    reports_dir = tmp_path / "reports"
    renders_dir = tmp_path / "renders"
    reports_dir.mkdir()
    renders_dir.mkdir()

    geom_file.write_text(json.dumps({
        "overall_dimensions": {"aspect_ratio_height_to_width": 1.0},
        "components": {}
    }))
    color_file.write_text(json.dumps({}))

    refiner = DummyStagnatingRefiner(
        geom_path=str(geom_file),
        color_path=str(color_file),
        reports_dir=str(reports_dir),
        renders_dir=str(renders_dir),
        max_iterations=6,
        target_score=90.0
    )

    result = refiner.run_loop()

    # Must abort early due to stagnation (3 passes instead of 6)
    assert result["status"] == "stagnated"
    assert result["converged"] is False
    assert result["total_passes"] == 3
    assert len(result["history"]) == 3
    assert result["final_score"] == 45.18
    assert result["final_geometry_score"] == 50.0
    assert result["final_color_score"] == 40.0

    # Verify refinement_log.json was written with matching schema
    log_file = reports_dir / "refinement_log.json"
    assert log_file.exists()
    with open(log_file, "r") as f:
        saved_log = json.load(f)
    assert saved_log["status"] == "stagnated"
    assert saved_log["total_passes"] == 3
    assert saved_log["final_score"] == 45.18


def test_unresolvable_recommendations_escalation(tmp_path):
    geom_file = tmp_path / "geom.json"
    color_file = tmp_path / "color.json"
    reports_dir = tmp_path / "reports"
    renders_dir = tmp_path / "renders"
    reports_dir.mkdir()
    renders_dir.mkdir()

    geom_file.write_text(json.dumps({
        "overall_dimensions": {"aspect_ratio_height_to_width": 1.0},
        "components": {}
    }))
    color_file.write_text(json.dumps({}))

    refiner = DummyStagnatingRefiner(
        geom_path=str(geom_file),
        color_path=str(color_file),
        reports_dir=str(reports_dir),
        renders_dir=str(renders_dir),
        max_iterations=4,
        target_score=90.0
    )

    result = refiner.run_loop()
    # On pass 3, rec_seen_counts reaches 3 -> marked unresolvable
    history = result["history"]
    # Recommendations count for active recommendations should drop
    assert history[2]["recommendations_count"] == 0
    assert result["unresolvable_recommendations_count"] == 1


def test_stage_report_reads_refinement_log(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    md_file = reports_dir / "final_report.md"

    refinement_data = {
        "status": "stagnated",
        "converged": False,
        "final_score": 45.18,
        "final_geometry_score": 48.0,
        "final_color_score": 42.0,
        "final_texture_score": 45.0,
        "total_passes": 3,
        "rebuilds_executed": 0,
        "history": []
    }

    project = ProjectDefinition(
        name="nexus_camera",
        project_dir=str(tmp_path),
        reference_image=str(tmp_path / "ref.jpg"),
        version="1.0.0",
        object_type="camera",
        specs_dir=str(tmp_path / "specs"),
        renders_dir=str(tmp_path / "renders"),
        textures_dir=str(tmp_path / "textures"),
        reports_dir=str(reports_dir)
    )

    report_meta = {
        "project_name": project.name,
        "project_version": project.version,
        "object_type": project.object_type,
        "timestamp": "2026-09-18T12:00:00",
        "stages_completed": ["refine", "report"],
        "verdicts": {"refinement": "FAIL", "overall": "FAIL"},
        "summary": {}
    }

    _write_markdown_report(
        path=str(md_file),
        report=report_meta,
        project=project,
        comparison={},
        refinement=refinement_data
    )

    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    # Verify report contains Unified Closed-Loop table with 3 passes and 45.18% score
    assert "Unified Closed-Loop" in content
    assert "45.18%" in content
    assert "Passes / Iterations" in content
    assert "Score Breakdown" in content


def test_gemini_fallback_confidence_and_category_safety():
    analyzer = GeminiVisionAnalyzer(api_key=None)
    # Test on a small synthetic image
    import cv2
    import numpy as np
    dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 80
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_img_path = f.name
    try:
        cv2.imwrite(temp_img_path, dummy_img)
        doc = analyzer._generate_fallback_analysis(temp_img_path)
        assert doc["decomposition_mode"] == "classical_cv_fallback"
        assert doc["decomposition_confidence"] == "low"
        for comp in doc["components"]:
            assert comp["metallic_confidence"] == "low"
            assert comp["roughness_confidence"] == "low"
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)


def test_comparator_metallic_confidence_gating():
    # Setup dummy comparison with low confidence component
    import numpy as np
    import cv2

    dummy_rend = np.ones((100, 100, 3), dtype=np.uint8) * 100
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        rend_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        geom_path = f.name

    try:
        cv2.imwrite(rend_path, dummy_rend)
        geom_spec = {
            "overall_dimensions": {
                "aspect_ratio_height_to_width": 1.0,
                "bbox_pixels": [10, 10, 80, 80]
            },
            "components": {
                "comp_main_body": {
                    "display_name": "Main Body",
                    "height_ratio_to_total": 0.5,
                    "estimated_metallic": 0.0,
                    "metallic_confidence": "low"
                }
            }
        }
        with open(geom_path, "w") as f:
            json.dump(geom_spec, f)

        comp = RenderGeometryComparator(
            render_image_path=rend_path,
            target_geom_json=geom_path
        )

        # Rendered mock data where rendered metallic is 0.45 (diff = 0.45)
        mock_rendered = {
            "aspect_ratio": 1.0,
            "components": {
                "comp_main_body": {
                    "height_ratio_to_total": 0.5,
                    "estimated_metallic": 0.45,
                    "estimated_roughness": 0.5
                }
            }
        }

        res = comp.compare_against_target(mock_rendered)
        # Because metallic_confidence is low, threshold is widened to 0.60.
        # diff of 0.45 should NOT trigger recommendation!
        rec_params = [r["parameter"] for r in res["correction_recommendations"]]
        assert "comp_main_body_metallic" not in rec_params

        # Now test with diff > 0.60
        mock_rendered["components"]["comp_main_body"]["estimated_metallic"] = 0.85
        res2 = comp.compare_against_target(mock_rendered)
        rec_params2 = [r["parameter"] for r in res2["correction_recommendations"]]
        assert "comp_main_body_metallic" in rec_params2
        # Verify recommendation has recommendation_id
        for r in res2["correction_recommendations"]:
            assert "recommendation_id" in r
    finally:
        if os.path.exists(rend_path):
            os.remove(rend_path)
        if os.path.exists(geom_path):
            os.remove(geom_path)
