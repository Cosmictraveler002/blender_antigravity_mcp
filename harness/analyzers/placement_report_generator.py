"""
Generalized 3D Element Placement & Viewpoint Report Generator (v4.0.0)
======================================================================
Object-agnostic spatial placement synthesizer.
Translates 2D image analysis into calibrated 3D transforms, bounding volumes,
and multi-viewpoint alignment constraints for procedural Blender scene construction.
"""

import os
import sys
import json
from typing import Dict, Any, List


class PlacementReportGenerator:
    """
    Generates high-precision, multi-viewpoint spatial placement and offset reports
    for 3D reconstruction based on the geometric and material analysis docs.
    Outputs:
    - placement_report.json (machine-readable 3D transforms and constraints)
    - placement_report.md (human-readable comprehensive specification)
    """

    def __init__(self, geom_json_path: str = "geometry_design_doc.json", color_json_path: str = "color_texture_design_doc.json"):
        self.geom_path = os.path.abspath(geom_json_path)
        self.color_path = os.path.abspath(color_json_path)

        with open(self.geom_path, "r", encoding="utf-8") as f:
            self.geom = json.load(f)

        self.color = {}
        if os.path.exists(self.color_path):
            with open(self.color_path, "r", encoding="utf-8") as f:
                self.color = json.load(f)

    def generate_report(self, output_json: str = "placement_report.json", output_md: str = "placement_report.md") -> Dict[str, Any]:
        dims = self.geom.get("overall_dimensions", {})
        components = self.geom.get("components", {})

        total_h = max(1, dims.get("height_px", 100))
        body_diam = max(1, dims.get("body_diameter_px", 100))
        body_rad = max(1.0, dims.get("body_radius_px", 50.0))
        aspect_ratio = dims.get("aspect_ratio_height_to_width", round(float(total_h / body_diam), 3))
        bx, by, bw, bh = dims.get("bbox_pixels", [0, 0, body_diam, total_h])
        cx = dims.get("center_x_px", bx + bw / 2.0)
        cy = dims.get("center_y_px", by + bh / 2.0)

        # Scale factor: set main width/diameter to 2.0 Blender units
        unit_scale = 2.0 / body_diam
        total_h_bu = round(total_h * unit_scale, 3)

        front_sections = []
        blender_transforms = {}

        for cid, comp in components.items():
            py_top = comp.get("pixel_y_top", by)
            py_bot = comp.get("pixel_y_bottom", by + bh)
            comp_h = max(1, py_bot - py_top)

            # Invert Y for 3D Z coordinates (bottom = 0.0, top = 1.0)
            z_end = round(max(0.0, min(1.0, (by + bh - py_top) / float(total_h))), 3)
            z_start = round(max(0.0, min(1.0, (by + bh - py_bot) / float(total_h))), 3)
            if z_start > z_end:
                z_start, z_end = z_end, z_start

            z_mid = round((z_start + z_end) / 2.0, 3)
            z_bu = round(z_mid * total_h_bu, 3)
            h_bu = round(max(0.05, (z_end - z_start) * total_h_bu), 3)

            comp_cx = comp.get("center_x_px", cx)
            offset_x_ratio = round((comp_cx - cx) / float(body_rad), 3)
            offset_x_bu = round(offset_x_ratio * 1.0, 3)

            comp_w = comp.get("pixel_bbox", [0, 0, bw, comp_h])[2]
            w_bu = round(max(0.05, comp_w * unit_scale), 3)

            front_sections.append({
                "component_id": cid,
                "name": comp.get("display_name", cid),
                "category": comp.get("category", "generic"),
                "z_range_normalized": [z_start, z_end],
                "z_center_bu": z_bu,
                "height_bu": h_bu,
                "width_bu": w_bu,
                "center_offset_x_bu": offset_x_bu,
                "color_hex": comp.get("color_hex", "#808080"),
                "notes": comp.get("visual_description", comp.get("description", ""))
            })

            blender_transforms[cid] = {
                "display_name": comp.get("display_name", cid),
                "location": [offset_x_bu, 0.0, z_bu],
                "scale": [round(w_bu / 2.0, 3), round(w_bu / 2.0, 3), round(h_bu / 2.0, 3)],
                "dimensions_bu": [w_bu, w_bu, h_bu],
                "rotation_euler_deg": [0.0, 0.0, 0.0]
            }

        report = {
            "meta": {
                "source_image": self.geom.get("image_metadata", {}).get("source_path", "reference.png"),
                "framework_version": "4.0.0",
                "coordinate_system": {
                    "origin": "BASE_CENTER_ON_GROUND (Z=0.0)",
                    "axis_orientation": "X=RIGHT, Y=FORWARD (camera facing), Z=VERTICAL UP",
                    "unit_normalization": f"WIDTH=2.0 BU, HEIGHT={total_h_bu} BU, SCALE_FACTOR={unit_scale:.5f}"
                }
            },
            "overall_bounds_bu": {
                "width_bu": 2.0,
                "depth_bu": 2.0,
                "height_bu": total_h_bu,
                "aspect_ratio": aspect_ratio
            },
            "front_elevation": {
                "camera_recommended": {
                    "angle_deg": 0.0,
                    "target_z_bu": round(total_h_bu * 0.5, 2),
                    "distance_bu": round(total_h_bu * 2.2, 2)
                },
                "sections": front_sections
            },
            "blender_target_transforms": blender_transforms
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        os.makedirs(os.path.dirname(os.path.abspath(output_md)), exist_ok=True)
        md_content = self._build_markdown(report)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"[PlacementReportGenerator] Saved machine-readable report -> {output_json}")
        print(f"[PlacementReportGenerator] Saved human-readable report -> {output_md}")
        return report

    def _build_markdown(self, r: Dict[str, Any]) -> str:
        meta = r["meta"]
        bounds = r["overall_bounds_bu"]
        sections = r["front_elevation"]["sections"]
        transforms = r["blender_target_transforms"]

        lines = [
            "# 3D Element Placement & Viewpoint Specification",
            f"**Source Target:** `{meta['source_image']}` | **Pipeline Version:** `{meta['framework_version']}`",
            "",
            "## 1. Global Coordinate System & Bounds",
            f"- **Coordinate Origin:** {meta['coordinate_system']['origin']}",
            f"- **Axes:** {meta['coordinate_system']['axis_orientation']}",
            f"- **Dimensions:** {bounds['width_bu']} x {bounds['depth_bu']} x {bounds['height_bu']} BU (Aspect Ratio: {bounds['aspect_ratio']:.2f})",
            "",
            "## 2. Component Layout & Z-Elevations",
            "| Component | Category | Z Range (Norm) | Height (BU) | Center (X, Z) BU | Visual Description |",
            "|:---|:---|:---|:---|:---|:---|"
        ]

        for s in sections:
            z_str = f"[{s['z_range_normalized'][0]:.3f} - {s['z_range_normalized'][1]:.3f}]"
            lines.append(f"| **{s['name']}** | `{s['category']}` | `{z_str}` | `{s['height_bu']:.2f}` | `({s['center_offset_x_bu']:.2f}, {s['z_center_bu']:.2f})` | {s['notes']} |")

        lines.extend([
            "",
            "## 3. Calibrated Blender 3D Transform Table",
            "| Object Name | Location [X, Y, Z] | Scale / Half-Extents | Dimensions [W, D, H] |",
            "|:---|:---|:---|:---|"
        ])

        for cid, t in transforms.items():
            loc = f"[{t['location'][0]:.2f}, {t['location'][1]:.2f}, {t['location'][2]:.2f}]"
            scale = f"[{t['scale'][0]:.2f}, {t['scale'][1]:.2f}, {t['scale'][2]:.2f}]"
            dims = f"[{t['dimensions_bu'][0]:.2f}, {t['dimensions_bu'][1]:.2f}, {t['dimensions_bu'][2]:.2f}]"
            lines.append(f"| `{t['display_name']}` | `{loc}` | `{scale}` | `{dims}` |")

        lines.append("")
        return "\n".join(lines)


if __name__ == "__main__":
    gen = PlacementReportGenerator()
    gen.generate_report()
