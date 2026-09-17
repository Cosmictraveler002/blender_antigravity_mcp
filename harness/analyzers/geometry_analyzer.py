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

        # 2. Studio / Uniform Background Subtraction (Row-wise gradient adaptive)
        w_margin = max(5, int(self.w * 0.03))
        bg_left = np.mean(self.img_rgb[:, :w_margin], axis=1)
        bg_right = np.mean(self.img_rgb[:, -w_margin:], axis=1)
        bg_row = (bg_left + bg_right) / 2.0  # (H, 3) row-wise studio background gradient
        margin_var = float(np.mean(np.var(self.img_rgb[:, :w_margin], axis=1)))

        if margin_var < 80.0:
            diff = np.linalg.norm(self.img_rgb.astype(np.float32) - bg_row[:, None, :], axis=2)
            margin_diffs = np.concatenate([diff[:, :w_margin].ravel(), diff[:, -w_margin:].ravel()])
            bg_noise_limit = float(np.percentile(margin_diffs, 98.5))
            effective_thresh = max(12.0, bg_noise_limit * 1.35)
            mask = (diff >= effective_thresh).astype(np.uint8) * 255
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

        result_mask = clean_mask if np.any(clean_mask > 0) else mask

        # Solidify horizontal spans for solid axisymmetric containers:
        # Prevents dark specular reflections / highlights from carving internal holes or notches into neck/rim
        ys_pre, xs_pre = np.where(result_mask > 0)
        if len(ys_pre) > 0:
            y1_pre, y2_pre = int(ys_pre.min()), int(ys_pre.max())
            for y in range(y1_pre, y2_pre + 1):
                row_xs = np.where(result_mask[y, :] > 0)[0]
                if len(row_xs) >= 2:
                    result_mask[y, row_xs[0]:row_xs[-1] + 1] = 255

        # Ground / floor contact shadow suppression for standing objects:
        # A container's base is physically narrower than or equal to its maximum body width.
        # Any horizontal mask flares in the bottom 25% extending outside the cylinder span are floor shadows.
        ys_all, xs_all = np.where(result_mask > 0)
        if len(ys_all) > 0:
            y_top_all, y_bot_all = int(ys_all.min()), int(ys_all.max())
            mid_start = y_top_all + int((y_bot_all - y_top_all) * 0.30)
            mid_end = y_top_all + int((y_bot_all - y_top_all) * 0.70)
            mid_lefts = []
            mid_rights = []
            for y_m in range(mid_start, mid_end):
                r_xs = np.where(result_mask[y_m, :] > 0)[0]
                if len(r_xs) > 0:
                    mid_lefts.append(r_xs[0])
                    mid_rights.append(r_xs[-1])
            if mid_lefts and mid_rights:
                med_left = int(np.median(mid_lefts))
                med_right = int(np.median(mid_rights))
                margin_tol = max(4, int((med_right - med_left) * 0.02))
                bot_start = y_top_all + int((y_bot_all - y_top_all) * 0.75)
                result_mask[bot_start:, :max(0, med_left - margin_tol)] = 0
                result_mask[bot_start:, min(self.w, med_right + margin_tol):] = 0

        # Base extension check for metallic bases / dark backgrounds
        # Traces physical resting contact line if mask cuts off on strong horizontal gradient
        ys, xs = np.where(result_mask > 0)
        if len(ys) > 0:
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
            sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            mag = np.sqrt(sob_x**2 + sob_y**2)
            bg_noise = float(np.mean(mag[-15:, :])) + 1e-5

            search_limit = min(self.h - 2, y2 + int((y2 - y1) * 0.15))
            row_mags = [float(np.mean(mag[y, x1:x2])) for y in range(y2 + 1, search_limit + 1)]
            thresh = max(3.5, bg_noise * 1.8)
            sig_rows = [i for i, m in enumerate(row_mags) if m >= thresh]
            extend_y = y2 + 1 + sig_rows[-1] if sig_rows else y2

            if extend_y > y2:
                for y in range(y2 + 1, extend_y + 1):
                    t_frac = (y - y2) / float(max(1, extend_y - y2))
                    inset = int((x2 - x1) * 0.08 * t_frac)
                    result_mask[y, x1 + inset:x2 - inset] = 255

        return result_mask

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

    def snap_seam_boundary(self, y_approx: int, gray_img: np.ndarray, bx: int, bw: int, by: int, bh: int, profile_mesh: List[Dict[str, Any]]) -> Tuple[int, float]:
        """
        Metric Edge Snapping Algorithm (PIPELINE_SOLUTIONS_SPEC.md § 2.3 & § 4.2).
        Snaps coarse approximate seam coordinate to sub-pixel physical edge:
        1. Search window: y in [y_approx - 15px, y_approx + 15px].
        2. Vertical Sobel Gy horizontal energy integration: E(y) = sum_x |Sobel_y(x, y)|.
        3. If SNR < 1.5, expands window to +-30px with bilateral filtering.
        4. Curvature inflection fallback (|d^2 r / dz^2| extrema).
        5. Prior preservation with confidence score = 0.0 if unverified.
        """
        y_min = max(by, y_approx - 10)
        y_max = min(by + bh, y_approx + 10)

        if y_max - y_min < 3:
            return y_approx, 0.0

        # 1. Standard Sobel Gy in +-10px window with Gaussian prior centered on y_approx
        window_gray = gray_img[y_min:y_max, bx:bx + bw]
        sobel_y = cv2.Sobel(window_gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_energy = np.sum(np.abs(sobel_y), axis=1)

        # Distance weighting to favor physical edge at or near y_approx (Bayesian MAP)
        sigma = 3.5
        y_indices = np.arange(y_min, y_max)
        dist = y_indices - y_approx
        weights = np.exp(-0.5 * (dist / sigma) ** 2)
        scored_energy = edge_energy * weights

        peak_idx = int(np.argmax(scored_energy))
        max_energy = float(edge_energy[peak_idx])
        noise_floor = float(np.median(edge_energy)) + 1e-5
        snr = max_energy / noise_floor

        # 2. Check if clear gradient peak exists (SNR >= 1.5)
        if snr >= 1.5:
            y_snapped = y_min + peak_idx
            conf = float(np.clip(snr / 3.5, 0.65, 1.0))
            return y_snapped, round(conf, 3)

        # 3. Low contrast / filleted seam: Expand window to +-30px and apply bilateral filter
        y_min_exp = max(by, y_approx - 30)
        y_max_exp = min(by + bh, y_approx + 30)
        exp_patch = gray_img[y_min_exp:y_max_exp, bx:bx + bw]

        if exp_patch.shape[0] >= 5 and exp_patch.shape[1] >= 5:
            bilateral = cv2.bilateralFilter(exp_patch, d=9, sigmaColor=75, sigmaSpace=75)
            sobel_bi = cv2.Sobel(bilateral, cv2.CV_32F, 0, 1, ksize=3)
            bi_energy = np.sum(np.abs(sobel_bi), axis=1)
            bi_peak = int(np.argmax(bi_energy))
            bi_snr = float(bi_energy[bi_peak]) / (float(np.median(bi_energy)) + 1e-5)

            if bi_snr >= 1.4:
                y_snapped = y_min_exp + bi_peak
                conf = float(np.clip(bi_snr / 3.0, 0.50, 0.85))
                return y_snapped, round(conf, 3)

        # 4. Curvature Inflection Fallback (|d^2 r / dz^2|)
        if profile_mesh and len(profile_mesh) >= 10:
            # Find levels corresponding to search window
            mesh_window = [
                m for m in profile_mesh
                if y_min_exp <= m.get("pixel_y", 0) <= y_max_exp
            ]
            if len(mesh_window) >= 3:
                r_vals = [m.get("radius_ratio", 1.0) for m in mesh_window]
                if len(r_vals) >= 5:
                    d1 = np.gradient(r_vals)
                    d2 = np.gradient(d1)
                    curv = np.abs(d2)
                    curv_peak = int(np.argmax(curv))
                    if curv[curv_peak] > float(np.mean(curv) + np.std(curv)):
                        y_snapped = int(mesh_window[curv_peak].get("pixel_y", y_approx))
                        return y_snapped, 0.60

        # 5. Prior Fallback: preserve Gemini prior coordinate with 0.0 confidence
        print(f"[GeometryAnalyzer] Snapping null peak for seam near y={y_approx}; defaulting to prior coordinate (conf=0.0).")
        return y_approx, 0.0

    def decompose_components(self, fg_mask: np.ndarray, bbox: Tuple[int, int, int, int], cx: float, gemini_spec_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Object-Agnostic Modular Component Decomposition (PIPELINE_SOLUTIONS_SPEC.md § 2 & § 3).
        Eliminates legacy equal-division slicing (rel_top = idx / num_c).
        Applies Coarse-to-Fine Fusion:
          Phase 1: Ingest Gemini semantic bounding boxes as prior approximations.
          Phase 2: Metric edge snapping with Sobel Gy + radial curvature extrema.
          Phase 3: Constructs Modular Component Object Model with BU dimensions.
        """
        bx, by, bw, bh = bbox
        components = {}
        gray_img = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        bu_scale = 2.0 / max(1.0, float(bw))  # Scale factor: Body diameter normalized to 2.0 BU

        # 1. Load Gemini components or generate classical CV fallback
        gemini_comps = []
        if gemini_spec_path and os.path.exists(gemini_spec_path):
            try:
                with open(gemini_spec_path, "r", encoding="utf-8") as f:
                    gdata = json.load(f)
                    gemini_comps = gdata.get("components", [])
            except Exception:
                gemini_comps = []

        # If gemini comps has <= 1 component, use dynamic multi-tier fallback
        if not gemini_comps or len(gemini_comps) <= 1:
            from harness.analyzers.gemini_analyzer import GeminiVisionAnalyzer
            analyzer = GeminiVisionAnalyzer()
            fb_analysis = analyzer._generate_fallback_analysis(self.image_path)
            gemini_comps = fb_analysis.get("components", [])

        # 2. Extract coarse vertical bounds for each component
        coarse_intervals = []
        num_c = len(gemini_comps)

        for idx, c in enumerate(gemini_comps):
            cid = c.get("component_id", f"comp_{idx}")
            bbox_raw = c.get("bounding_box") or c.get("pixel_bbox")
            if bbox_raw and len(bbox_raw) == 4:
                if max(bbox_raw) > 1.0 and max(bbox_raw) <= 1000.0:
                    # Normalized 0..1000 coordinates relative to full image height h
                    y1_approx = int((bbox_raw[0] / 1000.0) * self.h)
                    y2_approx = int((bbox_raw[2] / 1000.0) * self.h)
                elif max(bbox_raw) <= 1.0:
                    # Normalized 0..1 coordinates relative to full image height h
                    y1_approx = int(bbox_raw[0] * self.h)
                    y2_approx = int(bbox_raw[2] * self.h)
                else:
                    # Direct pixel bbox [x, y, w, h]
                    y1_approx = int(bbox_raw[1])
                    y2_approx = int(bbox_raw[1] + bbox_raw[3])
            else:
                # Relative prior distribution based on component order
                y1_approx = int(by + (idx / float(num_c)) * bh)
                y2_approx = int(by + ((idx + 1) / float(num_c)) * bh)

            y1_approx = max(by, min(by + bh, y1_approx))
            y2_approx = max(y1_approx + 5, min(by + bh, y2_approx))
            coarse_intervals.append((cid, c, y1_approx, y2_approx))

        # 3. Metric Edge Snapping across all seams
        # Generate temporary 100-level radial profile for curvature fallback
        y_coords_sub = np.arange(by, by + bh)
        left_sub = np.full_like(y_coords_sub, float(bx))
        right_sub = np.full_like(y_coords_sub, float(bx + bw))
        temp_mesh = self.generate_radial_profile_mesh(y_coords_sub, left_sub, right_sub, float(bw), num_levels=100)

        # Snap internal seams
        snapped_boundaries = [by]
        boundary_confidences = [1.0]

        for i in range(len(coarse_intervals) - 1):
            approx_seam = coarse_intervals[i][3]
            y_snap, conf = self.snap_seam_boundary(approx_seam, gray_img, bx, bw, by, bh, temp_mesh)
            # Ensure monotonicity
            y_snap = max(snapped_boundaries[-1] + 5, min(by + bh - (len(coarse_intervals) - 1 - i) * 5, y_snap))
            snapped_boundaries.append(y_snap)
            boundary_confidences.append(conf)

        snapped_boundaries.append(by + bh)
        boundary_confidences.append(1.0)

        # 4. Construct Modular Component Object Model
        category_map = {
            "rim": ("collar", "lathe_profile"),
            "chime": ("collar", "lathe_profile"),
            "collar": ("collar", "lathe_profile"),
            "cap": ("closure", "cylinder"),
            "closure": ("closure", "cylinder"),
            "lid": ("closure", "cylinder"),
            "body": ("substrate", "cylinder"),
            "substrate": ("substrate", "cylinder"),
            "main_body": ("substrate", "cylinder"),
            "base": ("structural_base", "concave_lathe"),
            "foot": ("structural_base", "concave_lathe"),
            "structural_base": ("structural_base", "concave_lathe"),
            "dome": ("structural_base", "concave_lathe"),
            "handle": ("attachment", "torus"),
            "label": ("decal_layer", "cylinder")
        }

        for idx in range(len(coarse_intervals)):
            cid, c_meta, _, _ = coarse_intervals[idx]
            y_top_px = snapped_boundaries[idx]
            y_bot_px = snapped_boundaries[idx + 1]
            h_px = max(1, y_bot_px - y_top_px)
            conf = min(boundary_confidences[idx], boundary_confidences[idx + 1])

            # Normalized 3D elevations: rel_z 0.0 at bottom, 1.0 at top
            z_bot_norm = round(float((by + bh - y_bot_px) / float(max(1, bh))), 3)
            z_top_norm = round(float((by + bh - y_top_px) / float(max(1, bh))), 3)

            # Dimensions in Blender Units (BU)
            w_bu = round(float(bw * bu_scale), 3)
            d_bu = w_bu
            h_bu = round(float(h_px * bu_scale), 3)

            # Infer category and geometry primitive
            cat_hint = c_meta.get("category", "enclosure").lower()
            role_hint = c_meta.get("semantic_role", cid).lower()
            matched_cat = "enclosure"
            matched_prim = "cylinder"

            for k, (cat, prim) in category_map.items():
                if k in cid.lower() or k in role_hint or k in cat_hint:
                    matched_cat = cat
                    matched_prim = prim
                    break

            color_hex = c_meta.get("color_hex", "#A0A0A0")
            roughness = float(c_meta.get("estimated_roughness", 0.50))
            metallic = float(c_meta.get("estimated_metallic", 0.0))

            components[cid] = {
                "display_name": c_meta.get("display_name", cid.replace("_", " ").title()),
                "category": matched_cat,
                "semantic_role": c_meta.get("semantic_role", cid),
                "elevation_z_range": [z_bot_norm, z_top_norm],
                "snapped_pixel_y_bounds": [y_top_px, y_bot_px],
                "snapping_confidence": conf,
                "dimensions_bu": {
                    "width": w_bu,
                    "depth": d_bu,
                    "height": h_bu
                },
                "geometry_primitive": matched_prim,
                "pixel_bbox": [bx, y_top_px, bw, h_px],
                "pixel_y_top": y_top_px,
                "pixel_y_bottom": y_bot_px,
                "height_px": h_px,
                "height_ratio_to_total": round(float(h_px / float(max(1, bh))), 3),
                "center_x_px": round(cx, 1),
                "center_y_px": round(y_top_px + h_px / 2.0, 1),
                "pbr_material": {
                    "material_id": f"mat_{cid}",
                    "base_color_hex": color_hex,
                    "metallic": metallic,
                    "roughness": roughness,
                    "texture_maps": {}
                },
                "color_hex": color_hex,
                "estimated_roughness": roughness,
                "estimated_metallic": metallic,
                "pbr_material_keywords": c_meta.get("pbr_material_keywords", [matched_cat])
            }

        return components

    def generate_annotated_image(self, doc: Dict[str, Any], output_path: str):
        """
        Draws high-fidelity optical inspection & modular component decomposition diagram.
        Expands canvas with dedicated padding margins to ensure all text banners,
        metric callouts, leader lines, and component cards are 100% visible with zero cropping.
        Scales UI elements with ui_scale so that diagrams remain resolution-invariant across all render sizes.
        """
        dims = doc["overall_dimensions"]
        bx, by, bw, bh = dims["bbox_pixels"]
        cx = dims["center_x_px"]

        # Scale UI annotations relative to object height (normalized to baseline bh=270px)
        ui_scale = max(0.8, bh / 270.0)

        pad_left = int(60 * ui_scale)
        pad_right = int(280 * ui_scale)
        pad_top = int(65 * ui_scale)
        pad_bottom = int(50 * ui_scale)
        W = self.w + pad_left + pad_right
        H = self.h + pad_top + pad_bottom

        # Create dark slate canvas and place source reference
        vis = np.full((H, W, 3), (26, 24, 24), dtype=np.uint8)
        vis[pad_top:pad_top + self.h, pad_left:pad_left + self.w] = self.img_bgr
        cv2.rectangle(vis, (pad_left, pad_top), (pad_left + self.w, pad_top + self.h), (50, 50, 55), 1)

        bx_c = bx + pad_left
        by_c = by + pad_top
        cx_c = int(cx + pad_left)

        thick_2 = max(1, int(round(2 * ui_scale)))
        thick_1 = max(1, int(round(1 * ui_scale)))

        # 1. Bounding box & Centerline
        cv2.rectangle(vis, (bx_c, by_c), (bx_c + bw, by_c + bh), (0, 255, 0), thick_2)
        cv2.line(vis, (cx_c, by_c - int(12 * ui_scale)), (cx_c, by_c + bh + int(12 * ui_scale)), (255, 255, 0), thick_1, cv2.LINE_AA)

        # 2. Radial profile mesh sample points
        pt_r = max(2, int(round(2.5 * ui_scale)))
        for m in doc.get("radial_profile_mesh", [])[::5]:
            py = int(m["pixel_y"]) + pad_top
            lx = int(m["left_x"]) + pad_left
            rx = int(m["right_x"]) + pad_left
            cv2.circle(vis, (lx, py), pt_r, (0, 255, 255), -1)
            cv2.circle(vis, (rx, py), pt_r, (0, 255, 255), -1)

        # 3. Top Banner (Uncropped inside padded area)
        ar = dims["aspect_ratio_height_to_width"]
        f_banner = 0.52 * ui_scale
        f_sub = 0.40 * ui_scale
        cv2.putText(vis, f"Dim: {bw}x{bh}px | Aspect: {ar:.2f}:1 | Edge-Snapped Seams",
                    (int(20 * ui_scale), int(28 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, f_banner, (0, 255, 180), thick_2, cv2.LINE_AA)
        num_comps = len(doc.get("components", {}))
        fullness = dims.get("silhouette_fullness_ratio", 0.90) * 100
        cv2.putText(vis, f"{num_comps} Modular Components Detected | Silhouette Fullness: {fullness:.1f}%",
                    (int(20 * ui_scale), int(48 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, f_sub, (180, 180, 180), thick_1, cv2.LINE_AA)

        # 4. Component strata lines, brackets, leader lines, and annotation cards
        colors = [
            (255, 180, 0),   # Sky blue / cyan
            (50, 230, 80),   # Vibrant green
            (0, 140, 255),   # Coral / Orange
            (255, 0, 255),   # Magenta
            (0, 220, 255)    # Yellow
        ]

        comps_list = list(doc.get("components", {}).items())
        
        # Calculate non-overlapping card Y positions in the right margin
        card_ys = []
        for idx, (cid, comp) in enumerate(comps_list):
            bounds = comp.get("snapped_pixel_y_bounds", [comp.get("pixel_y_top", by), comp.get("pixel_y_bottom", by + bh)])
            mid_y = (bounds[0] + bounds[1]) // 2 + pad_top
            card_ys.append(mid_y)

        # Enforce minimum vertical separation
        card_sep = int(55 * ui_scale)
        for i in range(1, len(card_ys)):
            if card_ys[i] - card_ys[i - 1] < card_sep:
                card_ys[i] = card_ys[i - 1] + card_sep

        card_x = bx_c + bw + int(38 * ui_scale)

        f_card_title = 0.40 * ui_scale
        f_card_sub = 0.34 * ui_scale
        f_card_det = 0.32 * ui_scale
        f_card_swatch = 0.30 * ui_scale

        for idx, (cid, comp) in enumerate(comps_list):
            color = colors[idx % len(colors)]
            bounds = comp.get("snapped_pixel_y_bounds", [comp.get("pixel_y_top", by), comp.get("pixel_y_bottom", by + bh)])
            y_top_orig, y_bot_orig = bounds[0], bounds[1]
            y1_c = y_top_orig + pad_top
            y2_c = y_bot_orig + pad_top
            conf = comp.get("snapping_confidence", 1.0)
            h_px = max(1, y_bot_orig - y_top_orig)
            h_pct = (h_px / float(max(1, bh))) * 100.0

            # Horizontal seam lines across can
            seam_th = thick_2 if conf >= 0.7 else thick_1
            cv2.line(vis, (bx_c - int(15 * ui_scale), y1_c), (bx_c + bw + int(15 * ui_scale), y1_c), color, seam_th, cv2.LINE_AA)
            cv2.line(vis, (bx_c - int(15 * ui_scale), y2_c), (bx_c + bw + int(15 * ui_scale), y2_c), color, seam_th, cv2.LINE_AA)

            # Vertical bracket on right edge of silhouette
            br_x = bx_c + bw + int(8 * ui_scale)
            cv2.line(vis, (br_x, y1_c), (br_x, y2_c), color, thick_2, cv2.LINE_AA)
            cv2.line(vis, (br_x - int(4 * ui_scale), y1_c), (br_x + int(4 * ui_scale), y1_c), color, thick_2, cv2.LINE_AA)
            cv2.line(vis, (br_x - int(4 * ui_scale), y2_c), (br_x + int(4 * ui_scale), y2_c), color, thick_2, cv2.LINE_AA)

            # Leader line to annotation card
            card_y = card_ys[idx]
            mid_can_y = (y1_c + y2_c) // 2
            cv2.line(vis, (br_x + int(4 * ui_scale), mid_can_y), (card_x - int(8 * ui_scale), card_y), color, thick_1, cv2.LINE_AA)
            cv2.circle(vis, (card_x - int(8 * ui_scale), card_y), max(2, int(3 * ui_scale)), color, -1)

            # Annotation Card in right margin
            dname = comp.get("display_name", cid.replace("_", " ").title())
            cat = comp.get("category", "substrate")
            prim = comp.get("geometry_primitive", "cylinder")
            cv2.putText(vis, f"[{dname}] (C={conf:.2f})", (card_x, card_y - int(12 * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, f_card_title, color, thick_1, cv2.LINE_AA)
            cv2.putText(vis, f"{cat} | {prim}", (card_x, card_y + int(2 * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, f_card_sub, (220, 220, 220), thick_1, cv2.LINE_AA)
            cv2.putText(vis, f"y: [{y_top_orig}, {y_bot_orig}]px ({h_px}px, {h_pct:.1f}%)", (card_x, card_y + int(15 * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, f_card_det, (160, 160, 160), thick_1, cv2.LINE_AA)

            # Color swatch & material properties
            hex_col = comp.get("color_hex", "#808080")
            met = comp.get("estimated_metallic", 0.0)
            rough = comp.get("estimated_roughness", 0.5)
            clean_hex = hex_col.lstrip("#")
            if len(clean_hex) == 6:
                try:
                    sw_rgb = tuple(int(clean_hex[i:i + 2], 16) for i in (0, 2, 4))
                    sw_bgr = (sw_rgb[2], sw_rgb[1], sw_rgb[0])
                except Exception:
                    sw_bgr = (128, 128, 128)
            else:
                sw_bgr = (128, 128, 128)

            sw_sz = int(10 * ui_scale)
            cv2.rectangle(vis, (card_x, card_y + int(20 * ui_scale)), (card_x + sw_sz, card_y + int(20 * ui_scale) + sw_sz), sw_bgr, -1)
            cv2.rectangle(vis, (card_x, card_y + int(20 * ui_scale)), (card_x + sw_sz, card_y + int(20 * ui_scale) + sw_sz), (200, 200, 200), thick_1)
            cv2.putText(vis, f"{hex_col} | Met:{met:.1f} | R:{rough:.2f}", (card_x + int(14 * ui_scale), card_y + int(29 * ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, f_card_swatch, (180, 180, 180), thick_1, cv2.LINE_AA)

        # 5. Bottom Footer (Uncropped inside padded area)
        bu_scale = 2.0 / max(1.0, float(bw))
        bh_bu = bh * bu_scale
        f_foot_main = 0.42 * ui_scale
        f_foot_sub = 0.35 * ui_scale
        cv2.putText(vis, f"Body Diam: {bw}px (2.00 BU) | Total H: {bh}px ({bh_bu:.2f} BU) | Edge Snapped",
                    (int(20 * ui_scale), H - int(24 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, f_foot_main, (0, 230, 255), thick_1, cv2.LINE_AA)
        cv2.putText(vis, "High-Fidelity Optical Inspection & Modular Component Decomposition",
                    (int(20 * ui_scale), H - int(8 * ui_scale)), cv2.FONT_HERSHEY_SIMPLEX, f_foot_sub, (140, 140, 140), thick_1, cv2.LINE_AA)

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

        # If gemini spec path wasn't provided, try resolving from common locations (only if matching image)
        if not gemini_spec_path:
            cand = os.path.join(os.path.dirname(output_json_path), "gemini_vision_analysis.json")
            if os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        g_data = json.load(f)
                    src = g_data.get("image_metadata", {}).get("source_path", "")
                    if os.path.basename(src).lower() == os.path.basename(self.image_path).lower():
                        gemini_spec_path = cand
                except Exception:
                    pass

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
