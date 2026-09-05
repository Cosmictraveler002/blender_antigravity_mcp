"""
Structural Geometry Analyzer (v4.0.0)
=====================================
Analyzes 3D meshes (existing objects in the Blender scene or reference meshes)
using an object-agnostic 7-step structural profiling pipeline:

1. Z-slice profiling (bpy/bmesh)  -> cross-section curve per slice
2. Symmetry detection             -> rotational / mirror axis + confidence
3. Primitive segmentation         -> cylindrical / tapered / flat segments
4. Material zone segmentation     -> segments grouped by material slot
5. Category classification        -> "bottle" / "box" / "generic"
6. Schema matching (non-generic)  -> expected vs. detected segments scored
7. Report assembly                -> { tier1, tier2, category_matched, warnings }
"""

import os
import sys
import json
import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np


class StructuralGeometryAnalyzer:
    """
    Object-agnostic 7-step structural geometry analyzer for 3D meshes in Blender.
    Can be run via socket client from host or directly inside Blender Python environment.
    """

    def __init__(self, num_slices: int = 100):
        self.num_slices = num_slices

    def analyze_scene_object(
        self,
        object_name: Optional[str] = None,
        category_hint: Optional[str] = None,
        output_json_path: Optional[str] = None,
        output_md_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the 7-step structural geometry analysis on a named object
        (or active object/all mesh objects) in the Blender scene.
        """
        blender_script = f"""
import bpy
import bmesh
import json
import math

def get_mesh_data(target_name):
    obj = None
    if target_name:
        obj = bpy.data.objects.get(target_name)
    if not obj:
        mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH' and 'Camera' not in o.name and 'Light' not in o.name and 'Wall' not in o.name and 'Table' not in o.name]
        obj = mesh_objs[0] if mesh_objs else bpy.context.active_object

    if not obj or obj.type != 'MESH':
        return {{"error": f"No mesh object found for target '{{target_name}}'"}}

    mw = obj.matrix_world
    verts_world = [mw @ v.co for v in obj.data.vertices]
    if not verts_world:
        return {{"error": "Object has no vertices"}}

    z_coords = [v.z for v in verts_world]
    x_coords = [v.x for v in verts_world]
    y_coords = [v.y for v in verts_world]

    z_min, z_max = min(z_coords), max(z_coords)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    total_h = max(1e-5, z_max - z_min)

    slots = [s.name for s in obj.material_slots]
    poly_materials = [p.material_index for p in obj.data.polygons]
    poly_z_centers = [(mw @ p.center).z for p in obj.data.polygons]

    edges = [(mw @ obj.data.vertices[e.vertices[0]].co, mw @ obj.data.vertices[e.vertices[1]].co) for e in obj.data.edges]
    
    num_slices = {self.num_slices}
    slices_data = []

    for i in range(num_slices):
        rel_z = i / max(1, num_slices - 1)
        cur_z = z_min + rel_z * total_h

        pts = []
        for v1, v2 in edges:
            if (v1.z - cur_z) * (v2.z - cur_z) <= 0.0 and abs(v1.z - v2.z) > 1e-6:
                t = (cur_z - v1.z) / (v2.z - v1.z)
                ix = v1.x + t * (v2.x - v1.x)
                iy = v1.y + t * (v2.y - v1.y)
                pts.append((ix, iy))

        if len(pts) < 3:
            nearby = [(v.x, v.y) for v in verts_world if abs(v.z - cur_z) <= (total_h / num_slices) * 1.5]
            pts = nearby if len(nearby) >= 3 else [(0.0, 0.0)]

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        radii = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
        r_mean = sum(radii) / len(radii) if radii else 0.0
        r_min = min(radii) if radii else 0.0
        r_max = max(radii) if radii else 0.0
        r_var = sum((r - r_mean) ** 2 for r in radii) / len(radii) if radii else 0.0

        angles = [math.atan2(p[1] - cy, p[0] - cx) for p in pts]
        sorted_indices = sorted(range(len(pts)), key=lambda k: angles[k])
        sorted_pts = [pts[k] for k in sorted_indices]

        area = 0.0
        perim = 0.0
        n_pts = len(sorted_pts)
        for j in range(n_pts):
            p_a = sorted_pts[j]
            p_b = sorted_pts[(j + 1) % n_pts]
            area += (p_a[0] * p_b[1] - p_b[0] * p_a[1])
            perim += math.hypot(p_b[0] - p_a[0], p_b[1] - p_a[1])
        area = abs(area) * 0.5

        circularity = (4.0 * math.pi * area) / (perim ** 2) if perim > 1e-4 else 0.0
        circularity = min(1.0, max(0.0, circularity))

        slices_data.append({{
            "slice_index": i,
            "rel_z": round(rel_z, 4),
            "world_z": round(cur_z, 4),
            "centroid": [round(cx, 4), round(cy, 4)],
            "radius_mean": round(r_mean, 4),
            "radius_min": round(r_min, 4),
            "radius_max": round(r_max, 4),
            "radius_var": round(r_var, 6),
            "area": round(area, 4),
            "perimeter": round(perim, 4),
            "circularity": round(circularity, 4),
            "point_count": len(pts)
        }})

    slice_mats = []
    for i in range(num_slices):
        cur_z = z_min + (i / max(1, num_slices - 1)) * total_h
        dz_thresh = (total_h / num_slices) * 1.5
        m_indices = [poly_materials[j] for j, pz in enumerate(poly_z_centers) if abs(pz - cur_z) <= dz_thresh]
        dom_idx = max(set(m_indices), key=m_indices.count) if m_indices else 0
        mat_name = slots[dom_idx] if dom_idx < len(slots) else "DefaultMaterial"
        slice_mats.append({{"slot_index": dom_idx, "material_name": mat_name}})

    out = {{
        "object_name": obj.name,
        "bbox": {{
            "x_min": round(x_min, 4), "x_max": round(x_max, 4), "width": round(x_max - x_min, 4),
            "y_min": round(y_min, 4), "y_max": round(y_max, 4), "depth": round(y_max - y_min, 4),
            "z_min": round(z_min, 4), "z_max": round(z_max, 4), "height": round(total_h, 4)
        }},
        "material_slots": slots,
        "slices": slices_data,
        "slice_materials": slice_mats
    }}
    return out

raw_data = get_mesh_data({repr(object_name)})
print("__STRUCTURAL_GEOM_START__" + json.dumps(raw_data) + "__STRUCTURAL_GEOM_END__")
"""
        from harness.blender.client import send_blender_code
        res = send_blender_code(blender_script)
        raw_stdout = res.get("result", {}).get("result", "")

        if "__STRUCTURAL_GEOM_START__" not in raw_stdout:
            raise RuntimeError(f"Failed to profile mesh in Blender: {raw_stdout or res.get('message', 'No output')}")

        json_str = raw_stdout.split("__STRUCTURAL_GEOM_START__")[1].split("__STRUCTURAL_GEOM_END__")[0]
        mesh_profile = json.loads(json_str)

        if "error" in mesh_profile:
            raise ValueError(mesh_profile["error"])

        report = self.process_profile_data(mesh_profile, category_hint=category_hint)

        if output_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        if output_md_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_md_path)), exist_ok=True)
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(self.render_markdown_report(report))

        return report

    def process_profile_data(self, profile: Dict[str, Any], category_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes Steps 2 through 7 on the profiled mesh slices.
        """
        slices = profile["slices"]
        bbox = profile["bbox"]
        slice_mats = profile.get("slice_materials", [])

        warnings: List[str] = []

        # --- STEP 2: SYMMETRY DETECTION ---
        symmetry_data = self._detect_symmetry(slices, bbox)

        # --- STEP 3: PRIMITIVE SEGMENTATION ---
        primitive_segments = self._segment_primitives(slices, bbox, warnings)

        # --- STEP 4: MATERIAL ZONE SEGMENTATION ---
        material_zones = self._segment_material_zones(primitive_segments, slice_mats, profile.get("material_slots", []))

        # --- STEP 5: CATEGORY CLASSIFICATION ---
        category = self._classify_category(bbox, symmetry_data, primitive_segments, category_hint)

        # --- STEP 6: SCHEMA MATCHING ---
        schema_match = self._match_schema(category, primitive_segments, bbox)

        # --- STEP 7: REPORT ASSEMBLY ---
        tier1 = {
            "object_name": profile["object_name"],
            "bounding_box": bbox,
            "aspect_ratio_height_to_width": round(bbox["height"] / max(1e-4, bbox["width"]), 3),
            "aspect_ratio_height_to_depth": round(bbox["height"] / max(1e-4, bbox["depth"]), 3),
            "symmetry": symmetry_data,
            "dominant_category": category,
            "primitive_segment_count": len(primitive_segments),
            "material_slot_count": len(profile.get("material_slots", []))
        }

        tier2 = {
            "slice_count": len(slices),
            "z_slices": slices,
            "primitive_segments": primitive_segments,
            "material_zones": material_zones
        }

        report = {
            "tier1": tier1,
            "tier2": tier2,
            "category_matched": schema_match,
            "warnings": warnings
        }

        return report

    def _detect_symmetry(self, slices: List[Dict[str, Any]], bbox: Dict[str, Any]) -> Dict[str, Any]:
        """Detect rotational and mirror symmetry across the Z-slices."""
        circularities = [s["circularity"] for s in slices if s["radius_mean"] > 1e-3]
        mean_circ = float(np.mean(circularities)) if circularities else 0.0

        r_vars = [s["radius_var"] / max(1e-4, s["radius_mean"] ** 2) for s in slices if s["radius_mean"] > 1e-3]
        mean_norm_var = float(np.mean(r_vars)) if r_vars else 1.0

        rot_conf = max(0.0, min(1.0, (mean_circ * 0.70) + (max(0.0, 1.0 - mean_norm_var * 5.0) * 0.30)))
        is_rotational = rot_conf >= 0.85

        centroids_x = [abs(s["centroid"][0]) for s in slices]
        centroids_y = [abs(s["centroid"][1]) for s in slices]
        width_half = max(1e-3, bbox["width"] / 2.0)
        depth_half = max(1e-3, bbox["depth"] / 2.0)

        cx_err = float(np.mean(centroids_x)) / width_half
        cy_err = float(np.mean(centroids_y)) / depth_half

        mirror_xz_conf = max(0.0, min(1.0, 1.0 - cy_err * 2.0))
        mirror_yz_conf = max(0.0, min(1.0, 1.0 - cx_err * 2.0))

        return {
            "rotational_z": {
                "is_symmetric": is_rotational,
                "confidence": round(rot_conf, 3),
                "mean_circularity": round(mean_circ, 3),
                "symmetry_order": "continuous_rotational" if is_rotational else "none"
            },
            "mirror_xz": {
                "is_symmetric": mirror_xz_conf >= 0.80,
                "confidence": round(mirror_xz_conf, 3)
            },
            "mirror_yz": {
                "is_symmetric": mirror_yz_conf >= 0.80,
                "confidence": round(mirror_yz_conf, 3)
            }
        }

    def _segment_primitives(self, slices: List[Dict[str, Any]], bbox: Dict[str, Any], warnings: List[str]) -> List[Dict[str, Any]]:
        """
        Segment the Z-profile into CYLINDRICAL, TAPERED, CURVED_DOME, or FLAT_DISC intervals
        using first and second derivatives of the radial profile r(z).
        """
        if len(slices) < 5:
            warnings.append("Insufficient slice count for detailed primitive segmentation.")
            return []

        radii = np.array([s["radius_mean"] for s in slices])
        total_h = bbox["height"]

        dz = max(1e-4, total_h / len(slices))
        dr_dz = np.gradient(radii, dz)
        d2r_dz2 = np.gradient(dr_dz, dz)

        cyl_thresh = 0.12     # |dr/dz| < 0.12 => constant radius
        taper_thresh = 0.18   # |d2r/dz2| < 0.18 => linear slope

        slice_classes = []
        for i in range(len(slices)):
            slope = abs(dr_dz[i])
            curv = abs(d2r_dz2[i])

            if slope < cyl_thresh:
                slice_classes.append("CYLINDRICAL")
            elif curv < taper_thresh:
                slice_classes.append("TAPERED")
            else:
                slice_classes.append("CURVED_DOME")

        raw_segments: List[Dict[str, Any]] = []
        cur_class = slice_classes[0]
        start_idx = 0

        for i in range(1, len(slice_classes)):
            if slice_classes[i] != cur_class:
                raw_segments.append({
                    "type": cur_class,
                    "start_idx": start_idx,
                    "end_idx": i - 1
                })
                cur_class = slice_classes[i]
                start_idx = i

        raw_segments.append({
            "type": cur_class,
            "start_idx": start_idx,
            "end_idx": len(slice_classes) - 1
        })

        merged: List[Dict[str, Any]] = []
        for seg in raw_segments:
            length = seg["end_idx"] - seg["start_idx"] + 1
            if length < 3 and merged:
                merged[-1]["end_idx"] = seg["end_idx"]
            else:
                merged.append(seg)

        final_segments: List[Dict[str, Any]] = []
        for idx, seg in enumerate(merged):
            s_i = seg["start_idx"]
            e_i = seg["end_idx"]
            seg_z_min = slices[s_i]["world_z"]
            seg_z_max = slices[e_i]["world_z"]
            seg_height = round(seg_z_max - seg_z_min, 4)
            seg_r_bot = slices[s_i]["radius_mean"]
            seg_r_top = slices[e_i]["radius_mean"]
            seg_r_mean = float(np.mean(radii[s_i:e_i + 1]))

            taper_angle_deg = round(math.degrees(math.atan2(abs(seg_r_top - seg_r_bot), max(1e-4, seg_height))), 1)

            final_segments.append({
                "segment_id": f"segment_{idx + 1:02d}",
                "primitive_type": seg["type"],
                "z_range_world": [round(seg_z_min, 4), round(seg_z_max, 4)],
                "rel_z_span": [round(slices[s_i]["rel_z"], 3), round(slices[e_i]["rel_z"], 3)],
                "height": seg_height,
                "radius_bottom": round(seg_r_bot, 4),
                "radius_top": round(seg_r_top, 4),
                "radius_mean": round(seg_r_mean, 4),
                "taper_angle_deg": taper_angle_deg,
                "slice_indices": [s_i, e_i]
            })

        return final_segments

    def _segment_material_zones(
        self,
        segments: List[Dict[str, Any]],
        slice_mats: List[Dict[str, Any]],
        slots: List[str]
    ) -> List[Dict[str, Any]]:
        """Map primitive segments to assigned Blender material slots."""
        zones: List[Dict[str, Any]] = []

        for seg in segments:
            s_i, e_i = seg["slice_indices"]
            mats_in_seg = [slice_mats[k]["material_name"] for k in range(s_i, min(len(slice_mats), e_i + 1))]
            dom_mat = max(set(mats_in_seg), key=mats_in_seg.count) if mats_in_seg else (slots[0] if slots else "DefaultMaterial")
            slot_idx = slots.index(dom_mat) if dom_mat in slots else 0

            seg["material_slot_index"] = slot_idx
            seg["material_name"] = dom_mat

            zones.append({
                "segment_id": seg["segment_id"],
                "material_name": dom_mat,
                "material_slot_index": slot_idx,
                "z_range": seg["z_range_world"]
            })

        return zones

    def _classify_category(
        self,
        bbox: Dict[str, Any],
        symmetry: Dict[str, Any],
        segments: List[Dict[str, Any]],
        category_hint: Optional[str] = None
    ) -> str:
        """Classify object into category: bottle, box, can, mug, or generic."""
        if category_hint and category_hint.strip().lower() not in ["", "none", "generic"]:
            return category_hint.strip().lower()

        aspect_ratio = bbox["height"] / max(1e-4, bbox["width"])
        is_rot = symmetry["rotational_z"]["is_symmetric"]

        if is_rot:
            if aspect_ratio >= 2.2 and len(segments) >= 2:
                return "bottle"
            elif 0.9 <= aspect_ratio <= 1.8:
                return "can"
            elif aspect_ratio < 0.7:
                return "bowl"
        else:
            if symmetry["mirror_xz"]["is_symmetric"] and symmetry["mirror_yz"]["is_symmetric"]:
                if abs(bbox["width"] - bbox["depth"]) / max(1e-4, bbox["width"]) < 0.3:
                    return "box"

        return "generic"

    def _match_schema(self, category: str, segments: List[Dict[str, Any]], bbox: Dict[str, Any]) -> Dict[str, Any]:
        """Score detected segments against expected structural schema if non-generic."""
        if category == "generic" or not segments:
            return {
                "category": category,
                "schema_matched": False,
                "schema_score": 1.0,
                "components": {}
            }

        schemas = {
            "bottle": ["base_section", "body_cylinder", "shoulder_dome", "cap"],
            "can": ["bottom_rim", "cylindrical_body", "top_rim"],
            "box": ["bottom_base", "vertical_walls", "top_cover"]
        }

        expected_parts = schemas.get(category, [])
        if not expected_parts:
            return {"category": category, "schema_matched": False, "schema_score": 1.0, "components": {}}

        matched_components: Dict[str, Any] = {}
        for idx, seg in enumerate(segments):
            rel_mid = (seg["rel_z_span"][0] + seg["rel_z_span"][1]) / 2.0
            part_name = expected_parts[min(len(expected_parts) - 1, int(rel_mid * len(expected_parts)))]
            matched_components[part_name] = {
                "segment_id": seg["segment_id"],
                "primitive_type": seg["primitive_type"],
                "rel_z_span": seg["rel_z_span"],
                "radius_ratio": round(seg["radius_mean"] / max(1e-4, bbox["width"] / 2.0), 3)
            }

        coverage = len(matched_components) / max(1, len(expected_parts))
        schema_score = round(min(1.0, coverage * 0.95), 2)

        return {
            "category": category,
            "schema_matched": schema_score >= 0.70,
            "schema_score": schema_score,
            "expected_components": expected_parts,
            "detected_components": matched_components
        }

    def render_markdown_report(self, report: Dict[str, Any]) -> str:
        """Render clean, human-readable diagnostic Markdown report."""
        t1 = report["tier1"]
        t2 = report["tier2"]
        cm = report["category_matched"]
        bb = t1["bounding_box"]

        md = []
        md.append(f"# Structural Geometry Analysis Report: {t1['object_name']}")
        md.append(f"\n**Target Category**: `{t1['dominant_category'].upper()}` | **Aspect Ratio**: `{t1['aspect_ratio_height_to_width']}:1`\n")

        md.append("## 1. Tier-1 Global Envelope & Symmetry")
        md.append(f"- **Dimensions**: `{bb['width']} x {bb['depth']} x {bb['height']}` (W x D x H)")
        md.append(f"- **Rotational Symmetry (Z)**: `{'YES' if t1['symmetry']['rotational_z']['is_symmetric'] else 'NO'}` (Confidence: `{t1['symmetry']['rotational_z']['confidence'] * 100:.1f}%`, Mean Circularity: `{t1['symmetry']['rotational_z']['mean_circularity']}`)")
        md.append(f"- **Mirror Symmetry**: XZ Plane: `{t1['symmetry']['mirror_xz']['confidence'] * 100:.1f}%` | YZ Plane: `{t1['symmetry']['mirror_yz']['confidence'] * 100:.1f}%`")
        md.append(f"- **Structural Segments**: `{t1['primitive_segment_count']}` detected across `{t2['slice_count']}` Z-slices\n")

        md.append("## 2. Tier-2 Primitive Segments & Material Zones")
        md.append("| Segment | Primitive Type | Rel Z Span | Height | Radius (Bot -> Top) | Material Slot |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for s in t2["primitive_segments"]:
            md.append(f"| `{s['segment_id']}` | **{s['primitive_type']}** | `{s['rel_z_span'][0]:.2f} - {s['rel_z_span'][1]:.2f}` | `{s['height']}` | `{s['radius_bottom']:.2f} -> {s['radius_top']:.2f}` | `{s.get('material_name', 'Slot 0')}` |")

        md.append("\n## 3. Morphological Schema Matching")
        md.append(f"- **Category Schema**: `{cm['category']}` (Matched: `{'YES' if cm['schema_matched'] else 'NO'}`, Alignment Score: `{cm['schema_score'] * 100:.1f}%`)")
        for comp, details in cm.get("detected_components", {}).items():
            md.append(f"  - **{comp}**: `{details['primitive_type']}` at rel Z `{details['rel_z_span']}` (Radius: `{details['radius_ratio'] * 100:.1f}%`)")

        if report.get("warnings"):
            md.append("\n## ⚠️ Diagnostics & Warnings")
            for w in report["warnings"]:
                md.append(f"- {w}")

        md.append("\n---\n*Generated by Blender MCP Structural Geometry Analyzer (v4.0.0)*\n")
        return "\n".join(md)
