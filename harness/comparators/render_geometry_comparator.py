import os
import sys
import json
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, List, Optional
import scipy.ndimage as ndi

from harness.utils.color_math import (
    srgb_to_xyz, xyz_to_lab, rgb_to_lab,
    compute_ciede2000, compute_procrustes_distance,
)

class RenderGeometryComparator:
    """
    Advanced Multi-Modal Comparator for 3D Render Verification:
    - Analyzes rendered image geometry, colors, textures, and text.
    - Compares 100-level radial profile mesh (MAE and worst-deviation level).
    - Measures perceptual color differences (CIEDE2000 ΔE) per component.
    - Calculates 1D height IoU and Procrustes shape distance.
    - Generates actionable correction recommendations for Blender parameters.
    - Produces enhanced multi-panel visual comparison with error status indicators.
    """

    def __init__(self, render_image_path: str = "bottle_render_analyzed.png",
                 target_geom_json: str = "geometry_design_doc.json",
                 target_color_json: str = "color_texture_design_doc.json"):
        self.render_path = os.path.abspath(render_image_path)
        self.geom_json_path = os.path.abspath(target_geom_json)
        self.color_json_path = os.path.abspath(target_color_json)

        if not os.path.exists(self.render_path):
            raise FileNotFoundError(f"Render image not found: {self.render_path}")
        if not os.path.exists(self.geom_json_path):
            raise FileNotFoundError(f"Target geometry doc not found: {self.geom_json_path}")

        self.img_bgr = cv2.imread(self.render_path)
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.h, self.w, _ = self.img_rgb.shape

        with open(self.geom_json_path, "r", encoding="utf-8") as f:
            self.target_geom = json.load(f)

        self.target_color = {}
        if os.path.exists(self.color_json_path):
            with open(self.color_json_path, "r", encoding="utf-8") as f:
                self.target_color = json.load(f)

    def analyze_rendered_scene(self) -> Dict[str, Any]:
        """
        Extracts dimensional landmarks, radial profile, and modular components
        directly from the rendered 3D image using the SAME GeometryAnalyzer model
        as the reference specification pipeline.
        """
        from harness.analyzers.geometry_analyzer import GeometryAnalyzer

        # 1. Run the exact same GeometryAnalyzer model on the rendered image
        ga = GeometryAnalyzer(self.render_path)
        specs_dir = os.path.dirname(os.path.abspath(self.geom_json_path))
        rend_json_path = os.path.join(specs_dir, "render_geometry_doc.json")
        reports_dir = os.path.join(os.path.dirname(specs_dir), "reports")
        rend_vis_path = os.path.join(reports_dir, "render_geometry_annotated.png")

        # Force analysis without reusing reference vision spec
        rend_doc = ga.analyze(output_json_path=rend_json_path, output_vis_path=rend_vis_path, gemini_spec_path="NON_EXISTENT")

        dims = rend_doc["overall_dimensions"]
        bx, by, bw, bh = dims["bbox_pixels"]
        cx = dims["center_x_px"]
        ar = dims["aspect_ratio_height_to_width"]
        rend_comps = rend_doc.get("components", {})

        # Extract RGB for each component
        for cid, comp in rend_comps.items():
            hex_c = comp.get("color_hex", "#808080").lstrip("#")
            if len(hex_c) == 6:
                comp["rgb"] = [int(hex_c[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
            else:
                comp["rgb"] = [0.5, 0.5, 0.5]

        result_dict = {
            "bbox": [bx, by, bw, bh],
            "total_height_px": int(bh),
            "body_diameter_px": int(bw),
            "aspect_ratio": ar,
            "center_x_px": round(float(cx), 1),
            "radial_profile_mesh": [m.get("radius_ratio", 0.0) for m in rend_doc.get("radial_profile_mesh", [])],
            "is_bottle": ("bamboo_cap" in self.target_geom.get("components", {})),
            "components": rend_comps,
            "doc": rend_doc
        }

        # Backwards compatibility fields for legacy comparator checks
        if "comp_upper_structure" in rend_comps:
            us = rend_comps["comp_upper_structure"]
            result_dict["shoulder_dome"] = {
                "pixel_y_top": us["pixel_y_top"],
                "pixel_y_bottom": us["pixel_y_bottom"],
                "height_px": us["height_px"],
                "height_ratio": us["height_ratio_to_total"]
            }
        if "comp_main_body" in rend_comps:
            mb = rend_comps["comp_main_body"]
            result_dict["main_cylinder"] = {
                "pixel_y_top": mb["pixel_y_top"],
                "pixel_y_bottom": mb["pixel_y_bottom"],
                "height_px": mb["height_px"],
                "height_ratio": mb["height_ratio_to_total"],
                "rgb": mb["rgb"]
            }
        if "comp_base_section" in rend_comps:
            bs = rend_comps["comp_base_section"]
            result_dict["base_section"] = {
                "pixel_y_seam": bs["pixel_y_top"],
                "pixel_y_bottom": bs["pixel_y_bottom"],
                "height_px": bs["height_px"],
                "height_ratio": bs["height_ratio_to_total"]
            }

        return result_dict

    def compare_against_target(self, rendered: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multi-modal comparison evaluating geometry, color, texture, and text.
        Generates concrete correction recommendations for Blender.
        """
        t_dims = self.target_geom["overall_dimensions"]
        t_comp = self.target_geom["components"]
        t_mats = self.target_color.get("component_materials", {})

        metrics_comparison = []
        recommendations = []

        # 1. Dimensional Metrics
        # Aspect Ratio
        target_ar = t_dims["aspect_ratio_height_to_width"]
        rend_ar = rendered["aspect_ratio"]
        ar_err = abs(rend_ar - target_ar) / target_ar
        ar_fid = max(0.0, 100.0 * (1.0 - ar_err))
        metrics_comparison.append({
            "metric": "Aspect Ratio (H:W)", "category": "GEOMETRY",
            "target": f"{target_ar:.2f} : 1", "rendered": f"{rend_ar:.2f} : 1",
            "error_pct": round(ar_err * 100.0, 1), "fidelity_pct": round(ar_fid, 1)
        })
        if ar_err > 0.025:
            # If rend_ar > target_ar (narrow/tall), target_delta > 0 to widen body
            # If rend_ar < target_ar (wide/short), target_delta < 0 to narrow body
            target_delta = round((rend_ar / target_ar - 1.0) * 0.4, 3)
            is_structural = bool(ar_err > 0.030)
            recommendations.append({
                "parameter": "body_scale_xy",
                "current": 1.00,
                "target_delta": target_delta,
                "priority": "HIGH" if is_structural else "MEDIUM",
                "is_structural": is_structural,
                "requires_rebuild": is_structural,
                "action": f"Adjust body scale XY by {target_delta:+.3f} to align aspect ratio to {target_ar:.2f}:1"
            })

        geom_fids = [ar_fid]

        # Modular Component Geometric Checks
        rend_comps = rendered.get("components", {})
        for cid, comp in t_comp.items():
            disp_name = comp.get("display_name", cid.replace("_", " ").title())
            if cid in rend_comps:
                target_h_ratio = comp.get("height_ratio_to_total", 0.0)
                rend_h_ratio = rend_comps[cid].get("height_ratio_to_total", 0.0)
                h_err = abs(rend_h_ratio - target_h_ratio) / max(1e-4, target_h_ratio)
                h_fid = max(0.0, 100.0 * (1.0 - h_err))
                metrics_comparison.append({
                    "metric": f"{disp_name} Height % of Total", "category": "GEOMETRY",
                    "target": f"{target_h_ratio * 100.0:.1f}%", "rendered": f"{rend_h_ratio * 100.0:.1f}%",
                    "error_pct": round(h_err * 100.0, 1), "fidelity_pct": round(h_fid, 1)
                })
                geom_fids.append(h_fid)

                # Dynamic tolerance: widen by 2.5x for unverified semantic priors (§ 4.2 Step 3)
                snap_conf = float(comp.get("snapping_confidence", 1.0))
                base_h_err_threshold = 0.05
                base_structural_threshold = 0.080
                base_abs_diff_threshold = 0.015

                if snap_conf == 0.0:
                    h_err_threshold = base_h_err_threshold * 2.5        # 0.05 -> 0.125
                    structural_threshold = base_structural_threshold * 2.5  # 0.080 -> 0.200
                    abs_diff_threshold = base_abs_diff_threshold * 2.5  # 0.015 -> 0.0375
                    print(f"[Tolerance] Joint '{cid}' has snapping_confidence=0.0; "
                          f"widened tolerance by 2.5x (h_err: {h_err_threshold}, structural: {structural_threshold})")
                else:
                    h_err_threshold = base_h_err_threshold
                    structural_threshold = base_structural_threshold
                    abs_diff_threshold = base_abs_diff_threshold

                if h_err > h_err_threshold:
                    delta_pct = round((rend_h_ratio - target_h_ratio) * 100.0, 1)
                    is_structural = bool(h_err > structural_threshold or abs(target_h_ratio - rend_h_ratio) > abs_diff_threshold)
                    recommendations.append({
                        "parameter": f"{cid}_height_ratio",
                        "current": round(rend_h_ratio, 3),
                        "target": round(target_h_ratio, 3),
                        "target_delta": round(target_h_ratio - rend_h_ratio, 3),
                        "priority": "HIGH" if (h_err > (0.15 * (2.5 if snap_conf == 0.0 else 1.0)) or is_structural) else "MEDIUM",
                        "is_structural": is_structural,
                        "requires_rebuild": is_structural,
                        "action": f"Adjust {disp_name} vertical span by {delta_pct:+.1f}% to match reference proportion ({target_h_ratio*100.0:.1f}%)"
                    })

        # 2. Radial Profile Mesh MAE & Procrustes Distance
        target_mesh = [m.get("radius_ratio", m.get("radius_ratio_to_max", 0.0)) for m in self.target_geom.get("radial_profile_mesh", [])]
        rendered_mesh = rendered.get("radial_profile_mesh", [])
        if len(target_mesh) == len(rendered_mesh) and len(target_mesh) > 0:
            diffs = np.abs(np.array(target_mesh) - np.array(rendered_mesh))
            profile_mae = float(np.mean(diffs))
            worst_level_idx = int(np.argmax(diffs))
            worst_level_z = round(worst_level_idx / 100.0, 2)
            profile_fid = max(0.0, 100.0 * (1.0 - profile_mae * 2.5))
            procrustes_d = compute_procrustes_distance(np.array(target_mesh), np.array(rendered_mesh))
        else:
            profile_mae = 0.05
            worst_level_z = 0.5
            profile_fid = 92.0
            procrustes_d = 0.02

        metrics_comparison.append({
            "metric": "Radial Profile Mesh MAE", "category": "SHAPE_CONTOUR",
            "target": "0.000", "rendered": f"{profile_mae:.4f} (worst @ rel_z={worst_level_z})",
            "error_pct": round(profile_mae * 100.0, 1), "fidelity_pct": round(profile_fid, 1)
        })
        geom_fids.append(profile_fid)

        if profile_mae > 0.040:
            recommendations.append({
                "parameter": "radial_profile_contour",
                "current": round(profile_mae, 4),
                "target": 0.025,
                "priority": "HIGH",
                "is_structural": True,
                "requires_rebuild": True,
                "action": f"Radial profile contour deviation ({profile_mae:.4f} MAE) exceeds threshold; clean procedural rebuild recommended"
            })

        # 3. Dynamic Perceptual Color Differences (CIEDE2000 ΔE)
        color_evals = {}
        col_fids = []

        def parse_rgb(c_spec):
            if isinstance(c_spec, list) and len(c_spec) >= 3:
                return tuple(c_spec[:3])
            if isinstance(c_spec, str) and c_spec.startswith("#"):
                h = c_spec.lstrip("#")
                if len(h) == 6:
                    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            return None

        for cid, comp in t_comp.items():
            ref_c = comp.get("color_hex") or comp.get("rgb")
            ref_rgb = parse_rgb(ref_c)
            disp_name = comp.get("display_name", cid.replace("_", " ").title())
            if cid in rend_comps and ref_rgb:
                rend_c = rend_comps[cid].get("color_hex") or rend_comps[cid].get("rgb")
                rend_rgb = parse_rgb(rend_c)
                if rend_rgb:
                    delta_e = compute_ciede2000(rgb_to_lab(ref_rgb), rgb_to_lab(rend_rgb))
                    col_fid = max(0.0, 100.0 - delta_e * 3.5)
                    metrics_comparison.append({
                        "metric": f"{disp_name} Color dE00", "category": "COLOR",
                        "target": f"{comp.get('color_hex', str(ref_rgb))}",
                        "rendered": f"{rend_comps[cid].get('color_hex', str(rend_rgb))} (dE={delta_e:.1f})",
                        "error_pct": round(delta_e, 1), "fidelity_pct": round(col_fid, 1)
                    })
                    color_evals[cid] = {"delta_e": round(delta_e, 2), "fidelity_pct": round(col_fid, 1)}
                    col_fids.append(col_fid)

                    if delta_e > 3.5:
                        recommendations.append({
                            "parameter": f"{cid}_base_color",
                            "current": rend_comps[cid].get("color_hex", str(rend_rgb)),
                            "target": comp.get("color_hex", str(ref_rgb)),
                            "target_rgb": list(ref_rgb) if ref_rgb else None,
                            "current_rgb": list(rend_rgb) if rend_rgb else None,
                            "priority": "HIGH" if delta_e > 8.0 else "MEDIUM",
                            "action": f"Adjust {disp_name} material color towards target ({comp.get('color_hex', str(ref_rgb))})"
                        })

        # Also evaluate materials in t_mats if any
        if t_mats:
            for mat_key, mat_spec in t_mats.items():
                bsdf = mat_spec.get("principled_bsdf", {})
                ref_rgb = bsdf.get("base_color_rgb")
                if not ref_rgb:
                    continue

                matched_rend_rgb = None
                if mat_key in rend_comps and "rgb" in rend_comps[mat_key]:
                    matched_rend_rgb = rend_comps[mat_key]["rgb"]
                elif "body" in mat_key and "comp_main_body" in rend_comps:
                    matched_rend_rgb = rend_comps["comp_main_body"].get("rgb")

                if matched_rend_rgb is not None:
                    delta_e = compute_ciede2000(rgb_to_lab(tuple(ref_rgb)), rgb_to_lab(tuple(matched_rend_rgb)))
                    col_fid = max(0.0, 100.0 - delta_e * 3.5)
                    disp_name = mat_key.replace("_", " ").title()

                    metrics_comparison.append({
                        "metric": f"{disp_name} Material dE00", "category": "COLOR",
                        "target": f"RGB {ref_rgb}", "rendered": f"RGB {matched_rend_rgb} (dE={delta_e:.1f})",
                        "error_pct": round(delta_e, 1), "fidelity_pct": round(col_fid, 1)
                    })
                    color_evals[f"{mat_key}_material"] = {"delta_e": round(delta_e, 2), "fidelity_pct": round(col_fid, 1)}
                    col_fids.append(col_fid)

        if not col_fids:
            col_fids = [85.0]

        # Real Texture and Material Surface Finish Evaluation (Roughness & Metallic fidelity)
        tex_fids = []
        for cid, comp in t_comp.items():
            if cid in rend_comps:
                r_target = float(comp.get("estimated_roughness", 0.50))
                r_rend = float(rend_comps[cid].get("estimated_roughness", 0.50))
                m_target = float(comp.get("estimated_metallic", 0.0))
                m_rend = float(rend_comps[cid].get("estimated_metallic", 0.0))
                r_fid = max(0.0, 100.0 * (1.0 - abs(r_target - r_rend)))
                m_fid = max(0.0, 100.0 * (1.0 - abs(m_target - m_rend)))
                tex_fids.append(0.50 * r_fid + 0.50 * m_fid)

                disp_name = comp.get("display_name", cid.replace("_", " ").title())
                if abs(m_target - m_rend) > 0.20:
                    recommendations.append({
                        "parameter": f"{cid}_metallic",
                        "current": m_rend,
                        "target": m_target,
                        "target_delta": round(m_target - m_rend, 2),
                        "priority": "HIGH",
                        "action": f"Adjust {disp_name} metallic property towards {m_target:.2f} (currently {m_rend:.2f})"
                    })
                if abs(r_target - r_rend) > 0.20:
                    recommendations.append({
                        "parameter": f"{cid}_roughness",
                        "current": r_rend,
                        "target": r_target,
                        "target_delta": round(r_target - r_rend, 2),
                        "priority": "MEDIUM",
                        "action": f"Adjust {disp_name} roughness property towards {r_target:.2f} (currently {r_rend:.2f})"
                    })
        tex_score = float(np.mean(tex_fids)) if tex_fids else 85.0

        # Weighted Category Aggregations
        geom_score = float(np.mean(geom_fids)) if geom_fids else 85.0
        color_score = float(np.mean(col_fids))

        # Overall Multi-Modal Fidelity: 50% Geometry & Contour, 35% Perceptual Color, 15% Texture
        total_fidelity = round(0.50 * geom_score + 0.35 * color_score + 0.15 * tex_score, 2)

        # Multi-Gate Component-Level Convergence Validation
        components_proportioned = True
        comp_height_errors = {}
        for cid, comp in t_comp.items():
            if cid in rend_comps:
                target_h_ratio = comp.get("height_ratio_to_total", 0.0)
                rend_h_ratio = rend_comps[cid].get("height_ratio_to_total", 0.0)
                diff = abs(rend_h_ratio - target_h_ratio)
                comp_height_errors[cid] = round(diff, 4)
                if diff > 0.008:
                    components_proportioned = False

        component_color_evals = {k: v for k, v in color_evals.items() if not k.endswith("_material")}
        colors_calibrated = all(eval_info.get("delta_e", 0.0) <= 6.5 for eval_info in component_color_evals.values())
        ar_ok = bool(ar_err <= 0.02)
        contour_ok = bool(profile_mae <= 0.040)
        converged = bool(ar_ok and components_proportioned and colors_calibrated and contour_ok)

        structural_rebuild_needed = any(r.get("requires_rebuild", False) for r in recommendations)

        convergence_status = {
            "converged": converged,
            "aspect_ratio_ok": ar_ok,
            "components_proportioned": components_proportioned,
            "colors_calibrated": colors_calibrated,
            "contour_profile_ok": contour_ok,
            "structural_rebuild_recommended": structural_rebuild_needed,
            "component_height_ratio_errors": comp_height_errors
        }

        return {
            "overall_fidelity": {
                "total_score_pct": total_fidelity,
                "geometry_score_pct": round(geom_score, 1),
                "color_score_pct": round(color_score, 1),
                "texture_score_pct": round(tex_score, 1)
            },
            "convergence_status": convergence_status,
            "metrics": metrics_comparison,
            "radial_profile_analysis": {
                "mae": round(profile_mae, 4),
                "worst_deviation_rel_z": worst_level_z,
                "procrustes_distance": round(procrustes_d, 4)
            },
            "color_evaluations": color_evals,
            "correction_recommendations": recommendations
        }

    def generate_annotated_render(self, rendered: Dict[str, Any], output_path: str = "render_geometry_annotated.png") -> str:
        """
        Returns annotated render generated by the exact same GeometryAnalyzer.
        Preserves uncropped canvas, padded margins, leader lines, and component cards.
        """
        if "doc" in rendered:
            from harness.analyzers.geometry_analyzer import GeometryAnalyzer
            ga = GeometryAnalyzer(self.render_path)
            ga.generate_annotated_image(rendered["doc"], output_path)
            return output_path

        if os.path.exists(output_path):
            return output_path

        return output_path

    def generate_side_by_side_comparison(self, ref_annotated_path: str = "geometry_analysis_annotated.png",
                                         rend_annotated_path: str = "render_geometry_annotated.png",
                                         rend_data: Optional[Dict[str, Any]] = None,
                                         comp_res: Optional[Dict[str, Any]] = None,
                                         output_path: str = "geometry_comparison_side_by_side.png") -> Optional[str]:
        """
        Creates an advanced diagnostic comparison collage:
        [Header with Fidelity Scores | Reference Photo | Status Indicators | Reconstructed 3D Render]
        Normalizes both panels to 1:1 object display height with aligned baselines, eliminating
        apparent scale/margin discrepancies and aligning corresponding component seams horizontally.
        """
        if not os.path.exists(ref_annotated_path) or not os.path.exists(rend_annotated_path):
            return None

        img_ref = cv2.imread(ref_annotated_path)
        img_rend = cv2.imread(rend_annotated_path)
        if img_ref is None or img_rend is None:
            return None

        ref_dims = self.target_geom.get("overall_dimensions", {})
        ref_bx, ref_by, ref_bw, ref_bh = ref_dims.get("bbox_pixels", [0, 0, img_ref.shape[1], img_ref.shape[0]])

        rend_dims = (rend_data.get("doc", {}).get("overall_dimensions") or rend_data) if rend_data else {}
        rend_bbox = rend_dims.get("bbox_pixels", rend_data.get("bbox", [0, 0, img_rend.shape[1], img_rend.shape[0]]) if rend_data else [0, 0, img_rend.shape[1], img_rend.shape[0]])
        rend_bx, rend_by, rend_bw, rend_bh = rend_bbox

        ref_bh = max(1, ref_bh)
        rend_bh = max(1, rend_bh)

        # Scale UI annotations relative to object height (baseline bh=270px)
        ui_scale_ref = max(0.8, ref_bh / 270.0)
        ui_scale_rend = max(0.8, rend_bh / 270.0)
        pad_top_ref = int(65 * ui_scale_ref)
        pad_top_rend = int(65 * ui_scale_rend)

        # Normalized target object display height
        target_obj_h = 720.0
        scale_ref = target_obj_h / float(ref_bh)
        scale_rend = target_obj_h / float(rend_bh)

        # Scale full annotated images
        ref_scaled = cv2.resize(img_ref, (int(round(img_ref.shape[1] * scale_ref)), int(round(img_ref.shape[0] * scale_ref))), interpolation=cv2.INTER_LANCZOS4)
        rend_scaled = cv2.resize(img_rend, (int(round(img_rend.shape[1] * scale_rend)), int(round(img_rend.shape[0] * scale_rend))), interpolation=cv2.INTER_LANCZOS4)

        # Object tops in scaled images
        y_top_ref_scaled = int(round((ref_by + pad_top_ref) * scale_ref))
        y_top_rend_scaled = int(round((rend_by + pad_top_rend) * scale_rend))

        margin_top = 45
        margin_bottom = 45
        panel_h = int(target_obj_h + margin_top + margin_bottom)

        def extract_aligned_panel(scaled_img, y_obj_top, width_out, height_out, m_top):
            panel = np.full((height_out, width_out, 3), (24, 24, 26), dtype=np.uint8)
            src_y1 = y_obj_top - m_top
            src_y2 = src_y1 + height_out
            dst_y1 = max(0, -src_y1)
            dst_y2 = height_out - max(0, src_y2 - scaled_img.shape[0])
            real_src_y1 = max(0, src_y1)
            real_src_y2 = min(scaled_img.shape[0], src_y2)
            real_w = min(width_out, scaled_img.shape[1])
            panel[dst_y1:dst_y2, 0:real_w] = scaled_img[real_src_y1:real_src_y2, 0:real_w]
            return panel

        w_ref_panel = ref_scaled.shape[1]
        w_rend_panel = rend_scaled.shape[1]

        panel_ref = extract_aligned_panel(ref_scaled, y_top_ref_scaled, w_ref_panel, panel_h, margin_top)
        panel_rend = extract_aligned_panel(rend_scaled, y_top_rend_scaled, w_rend_panel, panel_h, margin_top)

        header_h = 100
        divider_w = 90
        total_w = w_ref_panel + divider_w + w_rend_panel
        canvas = np.full((panel_h + header_h, total_w, 3), (22, 20, 22), dtype=np.uint8)

        # Header with scores
        of = comp_res.get("overall_fidelity", {}) if comp_res else {}
        total_score = of.get("total_score_pct", 95.0)
        geom_score = of.get("geometry_score_pct", 96.0)
        color_score = of.get("color_score_pct", 90.0)
        tex_score = of.get("texture_score_pct", 90.0)

        cv2.putText(canvas, f"MULTI-MODAL GEOMETRY & COMPONENT FIDELITY: {total_score:.1f}%",
                    (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 255, 180), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Geometry: {geom_score:.1f}%  |  Color (CIEDE2000): {color_score:.1f}%  |  Texture: {tex_score:.1f}%  |  1:1 Normalized Scale",
                    (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # Subtitles
        ref_ar = ref_dims.get("aspect_ratio_height_to_width", 0.0)
        rend_ar = rend_dims.get("aspect_ratio_height_to_width", rend_data.get("aspect_ratio", 0.0) if rend_data else 0.0)
        cv2.putText(canvas, f"REFERENCE SPECIFICATION ({ref_bw}x{ref_bh}px, AR: {ref_ar:.2f}:1)",
                    (30, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"3D RENDER RECONSTRUCTION ({rend_bw}x{rend_bh}px, AR: {rend_ar:.2f}:1)",
                    (w_ref_panel + divider_w + 30, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

        # Place panels
        canvas[header_h:header_h + panel_h, 0:w_ref_panel] = panel_ref
        canvas[header_h:header_h + panel_h, w_ref_panel + divider_w:total_w] = panel_rend

        # Divider vertical centerline
        cv2.line(canvas, (w_ref_panel + divider_w // 2, header_h), (w_ref_panel + divider_w // 2, canvas.shape[0] - 10), (45, 45, 50), 1)

        # Connect landmarks across divider
        y_apex = header_h + margin_top
        y_base = header_h + int(margin_top + target_obj_h)

        landmarks = [
            ("Top Apex", y_apex, y_apex, (255, 0, 255)),
        ]

        ref_comps = self.target_geom.get("components", {})
        rend_comps = rend_data.get("components", {}) if rend_data else {}

        seam_colors = [(0, 255, 255), (0, 140, 255), (255, 180, 0), (255, 0, 255)]
        seam_idx = 0
        ref_comp_keys = list(ref_comps.keys())
        for cid in ref_comp_keys:
            if cid in rend_comps and cid != ref_comp_keys[-1]:
                c_ref = ref_comps[cid]
                c_rend = rend_comps[cid]
                ref_seam_raw = c_ref.get("pixel_y_bottom", ref_by + ref_bh)
                rend_seam_raw = c_rend.get("pixel_y_bottom", rend_by + rend_bh)
                ref_seam_rel = (ref_seam_raw - ref_by) / float(ref_bh)
                rend_seam_rel = (rend_seam_raw - rend_by) / float(rend_bh)
                y_ref_seam = header_h + int(margin_top + ref_seam_rel * target_obj_h)
                y_rend_seam = header_h + int(margin_top + rend_seam_rel * target_obj_h)

                disp_name = c_ref.get("display_name", cid).replace("Comp ", "")
                col = seam_colors[seam_idx % len(seam_colors)]
                seam_idx += 1
                landmarks.append((f"{disp_name} Seam", y_ref_seam, y_rend_seam, col))

        landmarks.append(("Table Base", y_base, y_base, (0, 255, 0)))

        for name, y_left, y_right, color in landmarks:
            cv2.line(canvas, (w_ref_panel - 25, y_left), (w_ref_panel, y_left), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (w_ref_panel + divider_w, y_right), (w_ref_panel + divider_w + 25, y_right), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (w_ref_panel, y_left), (w_ref_panel + divider_w, y_right), color, 2, cv2.LINE_AA)
            cv2.circle(canvas, (w_ref_panel, y_left), 4, color, -1)
            cv2.circle(canvas, (w_ref_panel + divider_w, y_right), 4, color, -1)

            delta = abs(y_left - y_right)
            dot_col = (0, 255, 0) if delta <= 8 else ((0, 165, 255) if delta <= 20 else (0, 0, 255))
            mid_x = w_ref_panel + divider_w // 2
            mid_y = (y_left + y_right) // 2
            cv2.circle(canvas, (mid_x, mid_y), 4, dot_col, -1)

            lbl = f"{name} (dY:{delta}px)"
            cv2.putText(canvas, lbl, (w_ref_panel + 6, mid_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (220, 220, 220), 1, cv2.LINE_AA)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, canvas)
        return output_path

    def run_comparison(self) -> Dict[str, Any]:
        print(f"[RenderGeometryComparator] Analyzing rendered 3D scene in {self.render_path}...")
        rend_data = self.analyze_rendered_scene()

        print("[RenderGeometryComparator] Executing multi-modal comparison against reference specifications...")
        comp_results = self.compare_against_target(rend_data)

        annotated_path = self.generate_annotated_render(rend_data)
        print(f"[RenderGeometryComparator] Saved annotated render -> {annotated_path}")

        sbs_path = self.generate_side_by_side_comparison(
            rend_data=rend_data, comp_res=comp_results
        )
        if sbs_path:
            print(f"[RenderGeometryComparator] Saved side-by-side comparison -> {sbs_path}")

        return {
            "rendered_geometry": rend_data,
            "comparison": comp_results,
            "annotated_render_path": annotated_path,
            "side_by_side_path": sbs_path
        }

if __name__ == "__main__":
    rend_file = sys.argv[1] if len(sys.argv) > 1 else "bottle_render_analyzed.png"
    geom_file = sys.argv[2] if len(sys.argv) > 2 else "geometry_design_doc.json"
    comparator = RenderGeometryComparator(rend_file, geom_file)
    res = comparator.run_comparison()

    print("\n==========================================================================")
    print(f" MULTI-MODAL COMPARISON RESULTS (Total Fidelity: {res['comparison']['overall_fidelity']['total_score_pct']}%)")
    print("==========================================================================")
    print(f"{'Metric':<28} | {'Target':<14} | {'Rendered':<24} | {'Fidelity %'}")
    print("-" * 80)
    for m in res["comparison"]["metrics"]:
        print(f"{m['metric']:<28} | {m['target']:<14} | {m['rendered']:<24} | {m['fidelity_pct']}%")
    print("==========================================================================")
    print("\nActionable Correction Recommendations:")
    for r in res["comparison"]["correction_recommendations"]:
        print(f"[{r['priority']}] {r['parameter']}: {r['action']}")
