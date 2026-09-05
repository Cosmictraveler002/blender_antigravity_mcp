"""
Generalized Image Geometry Analyzer (v4.0.0)
==============================================
Object-agnostic 2D computer vision analyzer for 3D reconstruction.
Extracts:
  - Alpha-aware or background-subtracted foreground segmentation
  - Precise bounding boxes, aspect ratios, center of mass, and fullness
  - 100-level radial/cross-sectional profile mesh from base to apex
  - 6th-degree silhouette polynomial contours & edge curvature
  - Multi-tier or semantic component decomposition
  - Measurement annotation overlays

Completely resolution-independent with zero hardcoded object dimensions or labels.
"""

import os
import sys
import json
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy.interpolate import UnivariateSpline


class GeometryAnalyzer:
    """
    Object-agnostic 2D geometry extractor for automated 3D modeling.
    Works with any object category, aspect ratio, and image dimensions.
    """

    def __init__(self, image_path: str):
        self.image_path = os.path.abspath(image_path)
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        # Load with alpha channel if present
        self.img_raw = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if self.img_raw is None:
            raise ValueError(f"Failed to read image at {self.image_path}")

        if len(self.img_raw.shape) == 3 and self.img_raw.shape[2] == 4:
            self.has_alpha = True
            self.alpha = self.img_raw[:, :, 3]
            self.img_bgr = self.img_raw[:, :, :3]
        else:
            self.has_alpha = False
            self.alpha = None
            self.img_bgr = self.img_raw if len(self.img_raw.shape) == 3 else cv2.cvtColor(self.img_raw, cv2.COLOR_GRAY2BGR)

        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.img_rgb.shape[:2]

    def segment_foreground(self) -> np.ndarray:
        """
        Extracts clean foreground binary mask:
        1. Uses alpha channel if valid alpha transparency is detected.
        2. Falls back to corner-sampled background subtraction & Otsu thresholding.
        3. Applies morphological filtering to ensure solid object silhouette.
        """
        # 1. Alpha channel check
        if self.has_alpha and self.alpha is not None:
            if np.any(self.alpha < 240) and np.any(self.alpha > 20):
                _, mask = cv2.threshold(self.alpha, 127, 255, cv2.THRESH_BINARY)
                # Fill small holes
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                return mask

        # 2. Studio / Uniform Background Subtraction
        corners = [
            self.img_rgb[:15, :15],
            self.img_rgb[:15, -15:],
            self.img_rgb[-15:, :15],
            self.img_rgb[-15:, -15:]
        ]
        corner_means = [np.mean(c.reshape(-1, 3), axis=0) for c in corners]
        corner_stds = [np.std(c.reshape(-1, 3), axis=0) for c in corners]

        avg_bg_color = np.mean(corner_means, axis=0)
        max_corner_std = np.max(corner_stds)

        if max_corner_std < 35.0:
            # Uniform background detected
            diff = np.linalg.norm(self.img_rgb.astype(np.float32) - avg_bg_color, axis=2)
            norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, mask = cv2.threshold(norm_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Non-uniform background: GrabCut initialization
            mask_gc = np.zeros((self.h, self.w), dtype=np.uint8)
            rect = (int(self.w * 0.05), int(self.h * 0.05), int(self.w * 0.90), int(self.h * 0.90))
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(self.img_bgr, mask_gc, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
                mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)
            except Exception:
                gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Keep significant contours (area > 0.5% of image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        min_area = (self.w * self.h) * 0.005
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

        return clean_mask if np.any(clean_mask > 0) else mask

    def extract_bounds_and_center(self, fg_mask: np.ndarray) -> Tuple[int, int, int, int, float, float]:
        """Calculates bounding box [x, y, w, h] and centroid (cx, cy)."""
        y_indices, x_indices = np.where(fg_mask > 0)
        if len(y_indices) == 0:
            return 0, 0, self.w, self.h, self.w / 2.0, self.h / 2.0

        x_min = int(x_indices.min())
        x_max = int(x_indices.max())
        y_min = int(y_indices.min())
        y_max = int(y_indices.max())

        w_box = max(1, x_max - x_min)
        h_box = max(1, y_max - y_min)
        cx = float(np.mean(x_indices))
        cy = float(np.mean(y_indices))

        return x_min, y_min, w_box, h_box, cx, cy

    def extract_silhouette_profile(self, fg_mask: np.ndarray, y_top: int, y_bot: int, cx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples left and right edge coordinates along every pixel row."""
        y_coords = np.arange(y_top, y_bot + 1)
        left_edges = []
        right_edges = []

        for y in y_coords:
            row = fg_mask[y, :]
            xs = np.where(row > 0)[0]
            if len(xs) > 0:
                left_edges.append(xs[0])
                right_edges.append(xs[-1])
            else:
                left_edges.append(cx)
                right_edges.append(cx)

        return y_coords, np.array(left_edges, dtype=np.float32), np.array(right_edges, dtype=np.float32)

    def fit_silhouette_polynomials(self, y_coords: np.ndarray, left_edges: np.ndarray, right_edges: np.ndarray, degree: int = 6) -> Dict[str, Any]:
        """Fits polynomials to silhouette edges for procedural contour synthesis."""
        if len(y_coords) < degree + 1:
            degree = max(1, len(y_coords) - 1)

        y_norm = (y_coords - y_coords[0]) / float(max(1, y_coords[-1] - y_coords[0]))
        left_coeffs = np.polyfit(y_norm, left_edges, degree).tolist()
        right_coeffs = np.polyfit(y_norm, right_edges, degree).tolist()

        # Curvature metrics
        dx_left = np.gradient(left_edges)
        ddx_left = np.gradient(dx_left)
        curv_left = np.abs(ddx_left) / (1.0 + dx_left**2)**1.5

        dx_right = np.gradient(right_edges)
        ddx_right = np.gradient(dx_right)
        curv_right = np.abs(ddx_right) / (1.0 + dx_right**2)**1.5

        return {
            "polynomial_degree": degree,
            "y_normalized_domain": [0.0, 1.0],
            "left_edge_coeffs": [round(float(c), 4) for c in left_coeffs],
            "right_edge_coeffs": [round(float(c), 4) for c in right_coeffs],
            "mean_left_curvature": round(float(np.mean(curv_left)), 5),
            "mean_right_curvature": round(float(np.mean(curv_right)), 5),
            "max_left_curvature": round(float(np.max(curv_left)), 5),
            "max_right_curvature": round(float(np.max(curv_right)), 5)
        }

    def generate_radial_profile_mesh(self, y_coords: np.ndarray, left_edges: np.ndarray, right_edges: np.ndarray, max_diameter_px: float, num_levels: int = 100) -> List[Dict[str, Any]]:
        """
        Generates dense cross-sectional profiles across the vertical span.
        rel_z: 0.000 (base/ground) to 1.000 (top apex).
        """
        profile_mesh = []
        total_rows = len(y_coords)
        if total_rows == 0:
            return profile_mesh

        max_radius = max(1.0, max_diameter_px / 2.0)

        for i in range(num_levels):
            rel_z = round(i / float(num_levels - 1), 3)
            # Row index inverted: rel_z 0.0 is at y_bot (bottom row), rel_z 1.0 is at y_top (top row)
            row_idx = int(np.clip((1.0 - rel_z) * (total_rows - 1), 0, total_rows - 1))

            y_pixel = int(y_coords[row_idx])
            lx = float(left_edges[row_idx])
            rx = float(right_edges[row_idx])
            width_px = max(0.0, rx - lx)
            radius = width_px / 2.0
            cx_row = (lx + rx) / 2.0
            radius_ratio = round(float(radius / max_radius), 4)

            profile_mesh.append({
                "level_index": i,
                "rel_z": rel_z,
                "pixel_y": y_pixel,
                "left_x": round(lx, 1),
                "right_x": round(rx, 1),
                "center_x": round(cx_row, 1),
                "width_px": round(width_px, 1),
                "radius_px": round(radius, 1),
                "radius_ratio": radius_ratio,
                "radius_ratio_to_max": radius_ratio
            })

        return profile_mesh

    def decompose_components(self, fg_mask: np.ndarray, bbox: Tuple[int, int, int, int], cx: float, gemini_spec_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Object-agnostic component decomposition:
        If gemini_vision_analysis.json is available, matches semantic components to spatial bounding boxes.
        Otherwise, establishes structural vertical tiers (base, mid, upper, top).
        """
        bx, by, bw, bh = bbox
        components = {}

        # Check for Gemini components
        gemini_comps = []
        if gemini_spec_path and os.path.exists(gemini_spec_path):
            try:
                with open(gemini_spec_path, "r", encoding="utf-8") as f:
                    gdata = json.load(f)
                    gemini_comps = gdata.get("components", [])
            except Exception:
                gemini_comps = []

        if gemini_comps:
            num_c = len(gemini_comps)
            for idx, c in enumerate(gemini_comps):
                cid = c.get("component_id", f"comp_{idx}")
                # Allocate proportional vertical strata based on sequence or category
                rel_top = idx / float(num_c)
                rel_bot = (idx + 1) / float(num_c)
                py1 = int(by + rel_top * bh)
                py2 = int(by + rel_bot * bh)
                ph = max(10, py2 - py1)

                components[cid] = {
                    "display_name": c.get("display_name", cid),
                    "category": c.get("category", "generic"),
                    "pixel_bbox": [bx, py1, bw, ph],
                    "pixel_y_top": py1,
                    "pixel_y_bottom": py2,
                    "height_px": ph,
                    "height_ratio_to_total": round(float(ph / max(1, bh)), 3),
                    "center_x_px": round(cx, 1),
                    "center_y_px": round(py1 + ph / 2.0, 1),
                    "color_hex": c.get("color_hex", "#808080"),
                    "estimated_roughness": c.get("estimated_roughness", 0.5),
                    "pbr_material_keywords": c.get("pbr_material_keywords", [])
                }
        else:
            # Default structural tiers
            tiers = [
                ("base_structure", 0.00, 0.30, "Base foundation or lower body"),
                ("mid_body", 0.30, 0.70, "Main central structure"),
                ("upper_structure", 0.70, 0.90, "Upper transition or shoulder"),
                ("top_apex", 0.90, 1.00, "Apex, cap, or crown element")
            ]
            for tid, z_start, z_end, desc in tiers:
                py1 = int(by + (1.0 - z_end) * bh)
                py2 = int(by + (1.0 - z_start) * bh)
                ph = max(10, py2 - py1)

                components[tid] = {
                    "display_name": tid.replace("_", " ").title(),
                    "category": "structural_tier",
                    "pixel_bbox": [bx, py1, bw, ph],
                    "pixel_y_top": py1,
                    "pixel_y_bottom": py2,
                    "height_px": ph,
                    "height_ratio_to_total": round(float(ph / max(1, bh)), 3),
                    "center_x_px": round(cx, 1),
                    "center_y_px": round(py1 + ph / 2.0, 1),
                    "description": desc
                }

        return components

    def generate_annotated_image(self, doc: Dict[str, Any], output_path: str):
        """Draws visual measurements, bounding boxes, centerline, and profile samples."""
        vis = self.img_bgr.copy()
        dims = doc["overall_dimensions"]
        bx, by, bw, bh = dims["bbox_pixels"]
        cx = int(dims["center_x_px"])

        # 1. Bounding box & Centerline
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.line(vis, (cx, max(0, by - 15)), (cx, min(self.h - 1, by + bh + 15)), (255, 255, 0), 1, cv2.LINE_AA)

        # 2. Radial profile mesh sample points
        for m in doc.get("radial_profile_mesh", [])[::5]:
            py = int(m["pixel_y"])
            lx = int(m["left_x"])
            rx = int(m["right_x"])
            cv2.circle(vis, (lx, py), 2, (0, 255, 255), -1)
            cv2.circle(vis, (rx, py), 2, (0, 255, 255), -1)

        # 3. Component strata lines
        colors = [(255, 0, 0), (0, 200, 255), (255, 0, 255), (0, 255, 0), (255, 120, 0)]
        for idx, (cid, comp) in enumerate(doc.get("components", {}).items()):
            color = colors[idx % len(colors)]
            y_top = comp.get("pixel_y_top", by)
            y_bot = comp.get("pixel_y_bottom", by + bh)
            cv2.line(vis, (max(0, bx - 20), y_top), (min(self.w - 1, bx + bw + 20), y_top), color, 1, cv2.LINE_AA)
            cv2.putText(vis, f"{comp.get('display_name', cid)}", (min(self.w - 180, bx + bw + 5), (y_top + y_bot) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        # Text banner
        cv2.putText(vis, f"Dim: {bw}x{bh}px | Aspect: {dims['aspect_ratio_height_to_width']:.2f}",
                    (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv2.LINE_AA)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, vis)

    def analyze(self, output_json_path: str = "geometry_design_doc.json", output_vis_path: str = "geometry_analysis_annotated.png", gemini_spec_path: Optional[str] = None) -> Dict[str, Any]:
        """Executes full geometric analysis pipeline."""
        print(f"[GeometryAnalyzer] Extracting high-fidelity dimensions for {self.image_path}...")

        # 1. Segment foreground
        fg_mask = self.segment_foreground()

        # 2. Extract bounds
        bx, by, bw, bh, cx, cy = self.extract_bounds_and_center(fg_mask)
        total_pixels = int(np.sum(fg_mask > 0))
        fullness = round(total_pixels / float(max(1, bw * bh)), 3)
        aspect_ratio = round(float(bh / max(1, bw)), 3)

        # 3. Silhouette profiles
        y_coords, left_edges, right_edges = self.extract_silhouette_profile(fg_mask, by, by + bh, cx)

        # 4. Polynomial contours
        poly_data = self.fit_silhouette_polynomials(y_coords, left_edges, right_edges)

        # 5. 100-level radial profile mesh
        profile_mesh = self.generate_radial_profile_mesh(y_coords, left_edges, right_edges, float(bw), num_levels=100)

        # If gemini spec path wasn't provided, try resolving from common locations
        if not gemini_spec_path:
            cand = os.path.join(os.path.dirname(output_json_path), "gemini_vision_analysis.json")
            if os.path.exists(cand):
                gemini_spec_path = cand

        # 6. Decompose components
        components = self.decompose_components(fg_mask, (bx, by, bw, bh), cx, gemini_spec_path)

        doc = {
            "image_metadata": {
                "source_path": self.image_path,
                "image_width_px": self.w,
                "image_height_px": self.h,
                "has_alpha_channel": self.has_alpha
            },
            "overall_dimensions": {
                "bbox_pixels": [bx, by, bw, bh],
                "body_diameter_px": bw,
                "body_radius_px": round(bw / 2.0, 1),
                "height_px": bh,
                "aspect_ratio_height_to_width": aspect_ratio,
                "center_x_px": round(cx, 1),
                "center_y_px": round(cy, 1),
                "foreground_area_pixels": total_pixels,
                "silhouette_fullness_ratio": fullness
            },
            "silhouette_contours": poly_data,
            "radial_profile_mesh": profile_mesh,
            "components": components
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        print(f"[GeometryAnalyzer] Saved geometry design doc -> {output_json_path}")

        self.generate_annotated_image(doc, output_vis_path)
        print(f"[GeometryAnalyzer] Saved annotated measurement image -> {output_vis_path}")

        return doc


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "reference.png"
    analyzer = GeometryAnalyzer(target)
    doc = analyzer.analyze()
    print(f"Overall Dimensions: {doc['overall_dimensions']['bbox_pixels']}")
    print(f"Components: {list(doc['components'].keys())}")
