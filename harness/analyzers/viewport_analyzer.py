"""
Viewport Analyzer
=================
Processes multi-viewport rendered images to analyze 360-degree coverage,
silhouette integrity, cross-viewport symmetry, and geometry defects.
"""

import os
import sys
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image


class ViewportAnalyzer:
    """Analyzes multi-viewport renders for coverage, silhouettes, symmetry, and defects."""

    def __init__(self, background_tolerance: int = 15):
        self.bg_tolerance = background_tolerance

    def analyze_all(
        self,
        viewport_manifest: Dict[str, Any],
        output_report_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze all viewports described in manifest.

        Args:
            viewport_manifest: Dict containing list of viewport entries.
            output_report_path: Optional path to save JSON report.

        Returns:
            Structured analysis report dictionary.
        """
        viewports = viewport_manifest.get("viewports", [])
        per_view_results = {}
        masks = {}

        for vp in viewports:
            name = vp.get("name")
            img_path = vp.get("path") or vp.get("file_path")
            if not img_path or not os.path.isfile(img_path):
                per_view_results[name] = {"status": "missing"}
                continue

            try:
                metrics, mask = self._analyze_single_image(img_path)
                metrics["azimuth"] = vp.get("azimuth")
                metrics["elevation"] = vp.get("elevation")
                metrics["ortho"] = vp.get("ortho")
                per_view_results[name] = metrics
                masks[name] = mask
            except Exception as e:
                per_view_results[name] = {"status": "error", "message": str(e)}

        # Cross-viewport comparisons
        cross_view_metrics = self._compute_cross_viewport_metrics(per_view_results, masks)

        # Defect detection & overall scoring
        defect_warnings = self._detect_defects(per_view_results)
        overall_score = self._compute_overall_health(per_view_results, cross_view_metrics, defect_warnings)

        report = {
            "total_viewports": len(viewports),
            "analyzed_viewports": len([v for v in per_view_results.values() if v.get("status") != "missing"]),
            "overall_health_score": round(overall_score, 1),
            "defect_warnings": defect_warnings,
            "cross_viewport_metrics": cross_view_metrics,
            "per_viewport_metrics": per_view_results
        }

        if output_report_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
            with open(output_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[ViewportAnalyzer] Report saved to: {output_report_path}")

        return report

    def _analyze_single_image(self, image_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
        """Extract silhouette and compute spatial metrics for one image."""
        img = Image.open(image_path)
        w, h = img.size
        arr = np.array(img)

        # Extract binary foreground mask
        has_transparency = arr.shape[2] == 4 and np.any(arr[:, :, 3] < 250)
        if has_transparency:
            # RGBA with genuine transparent pixels: Alpha > 15
            alpha = arr[:, :, 3]
            mask = alpha > 15
        else:
            # Solid background (RGB or opaque RGBA): Check difference from border background
            corners = np.array([
                arr[0, 0, :3], arr[0, -1, :3],
                arr[-1, 0, :3], arr[-1, -1, :3],
                arr[0, w // 2, :3], arr[-1, w // 2, :3]
            ], dtype=np.float32)
            bg_color = np.median(corners, axis=0)
            diff = np.linalg.norm(arr[:, :, :3].astype(np.float32) - bg_color, axis=2)
            mask = diff > self.bg_tolerance

        total_pixels = w * h
        fg_pixels = int(np.sum(mask))
        coverage_pct = round((fg_pixels / total_pixels) * 100.0, 2)

        metrics = {
            "status": "success",
            "image_size": [w, h],
            "foreground_pixels": fg_pixels,
            "coverage_percentage": coverage_pct,
            "is_clipped": False,
            "aspect_ratio": 1.0,
            "bbox": [0, 0, 0, 0]
        }

        if fg_pixels > 0:
            y_indices, x_indices = np.where(mask)
            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
            bbox_w = x_max - x_min + 1
            bbox_h = y_max - y_min + 1

            # Check clipping against edges (1px border)
            is_clipped = bool(x_min <= 1 or y_min <= 1 or x_max >= w - 2 or y_max >= h - 2)
            aspect_ratio = round(bbox_h / max(1, bbox_w), 3)

            metrics.update({
                "bbox": [x_min, y_min, bbox_w, bbox_h],
                "bounding_width_px": bbox_w,
                "bounding_height_px": bbox_h,
                "aspect_ratio": aspect_ratio,
                "is_clipped": is_clipped,
                "framing_fill_factor": round((fg_pixels / max(1, bbox_w * bbox_h)) * 100.0, 1)
            })

        return metrics, mask

    def _compute_cross_viewport_metrics(
        self,
        per_view: Dict[str, Any],
        masks: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Compute IoU symmetry and dimensional ratios across corresponding angles."""
        cross = {}

        # 1. Left vs Right Symmetry (flip right view horizontally)
        if "left" in masks and "right" in masks:
            m_left = masks["left"]
            m_right = masks["right"]
            if m_left.shape == m_right.shape and np.sum(m_left) > 0 and np.sum(m_right) > 0:
                m_right_flipped = np.fliplr(m_right)
                intersection = np.logical_and(m_left, m_right_flipped)
                union = np.logical_or(m_left, m_right_flipped)
                iou = float(np.sum(intersection) / max(1, np.sum(union)))
                cross["lateral_symmetry_iou"] = round(iou * 100.0, 2)

        # 2. Front vs Back Symmetry
        if "front" in masks and "back" in masks:
            m_front = masks["front"]
            m_back = masks["back"]
            if m_front.shape == m_back.shape and np.sum(m_front) > 0 and np.sum(m_back) > 0:
                m_back_flipped = np.fliplr(m_back)
                intersection = np.logical_and(m_front, m_back_flipped)
                union = np.logical_or(m_front, m_back_flipped)
                iou = float(np.sum(intersection) / max(1, np.sum(union)))
                cross["anterior_posterior_iou"] = round(iou * 100.0, 2)

        # 3. Top vs Bottom Area Ratio
        top_res = per_view.get("top", {})
        bot_res = per_view.get("bottom", {})
        if top_res.get("status") == "success" and bot_res.get("status") == "success":
            top_px = top_res.get("foreground_pixels", 0)
            bot_px = bot_res.get("foreground_pixels", 0)
            if bot_px > 0:
                cross["top_to_bottom_area_ratio"] = round(top_px / bot_px, 3)

        return cross

    def _detect_defects(self, per_view: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential geometric issues from multi-angle metrics."""
        warnings = []

        for name, m in per_view.items():
            if m.get("status") != "success":
                continue

            # Clipped view detection
            if m.get("is_clipped", False):
                warnings.append({
                    "severity": "MEDIUM",
                    "viewport": name,
                    "issue": "Model clipped at frame boundary",
                    "recommendation": "Increase camera distance or ortho scale padding"
                })

            # Low coverage detection
            cov = m.get("coverage_percentage", 0.0)
            if cov < 2.0:
                warnings.append({
                    "severity": "HIGH",
                    "viewport": name,
                    "issue": f"Extremely low object visibility ({cov}%)",
                    "recommendation": "Verify camera focus, target center, or object occlusion"
                })

        return warnings

    def _compute_overall_health(
        self,
        per_view: Dict[str, Any],
        cross: Dict[str, Any],
        defects: List[Dict[str, Any]]
    ) -> float:
        """Aggregate health score between 0.0 and 100.0."""
        valid_views = [m for m in per_view.values() if m.get("status") == "success"]
        if not valid_views:
            return 0.0

        # Base score from successfully captured viewports
        coverage_scores = []
        for m in valid_views:
            cov = m.get("coverage_percentage", 0.0)
            # Optimal coverage is between 8% and 50%
            if 8.0 <= cov <= 60.0:
                score = 100.0
            elif cov < 8.0:
                score = max(20.0, (cov / 8.0) * 100.0)
            else:
                score = max(50.0, 100.0 - (cov - 60.0))
            coverage_scores.append(score)

        avg_coverage_health = float(np.mean(coverage_scores))

        # Penalty for defects
        penalty = 0.0
        for d in defects:
            if d.get("severity") == "HIGH":
                penalty += 10.0
            elif d.get("severity") == "MEDIUM":
                penalty += 4.0

        health = max(10.0, min(100.0, avg_coverage_health - penalty))
        return health
