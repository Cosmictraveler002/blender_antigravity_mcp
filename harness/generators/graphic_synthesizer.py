"""
Parametric 2D Graphic Surface & Packaging Art Synthesizer (v5.0.0)
===================================================================
Implements Stage 2.1 (PIPELINE_SOLUTIONS_SPEC.md § 6):
  - Standardized 5-step procedural workflow:
      Step 1: Graphical Motif Segmentation & Alpha Extraction
      Step 2: Parametric UV Aspect Ratio Canvas Initialization (2*pi*r*S_res x h*S_res, u=0.50 center)
      Step 3: Resolution-Scaled Vector Typography & Layout
      Step 4: Multi-Channel Material Map Baking (Diffuse, Roughness Delta, Embossed Normal)
      Step 5: Blender Shader Node Graph Integration Metadata
"""

import os
import sys
import json
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple, Optional


class GraphicArtSynthesizer:
    """
    Parametric 2D graphic packaging and labeling synthesizer for manufactured 3D assets.
    """

    def __init__(self, target_resolution_scale: int = 512):
        self.scale_res = target_resolution_scale

    def synthesize_packaging(
        self,
        component_id: str,
        geometry_spec: Dict[str, Any],
        typography_manifest: List[Dict[str, Any]],
        output_dir: str,
        reference_image_path: Optional[str] = None,
        substrate_color_hex: str = "#F0F0F0",
        substrate_roughness: float = 0.45,
        ink_roughness: float = 0.18,
        is_cylindrical: bool = True,
        emboss_height: float = 0.04
    ) -> Dict[str, Any]:
        """
        Executes the 5-step procedural packaging synthesis workflow.
        """
        os.makedirs(output_dir, exist_ok=True)

        # -------------------------------------------------------------
        # Step 2: Parametric UV Aspect Ratio Canvas Initialization (§ 6.3)
        # -------------------------------------------------------------
        # Physical dimensions from geometry spec or defaults
        dims_bu = geometry_spec.get("dimensions_bu", {})
        width_bu = dims_bu.get("width", 2.0)
        height_bu = dims_bu.get("height", 3.0)
        radius_bu = width_bu / 2.0

        if is_cylindrical:
            # W_canvas = round(2 * pi * r_body * S_res), H_canvas = round(h_comp * S_res)
            circumference_bu = 2.0 * math.pi * radius_bu
            w_canvas = max(1024, int(round(circumference_bu * self.scale_res)))
            h_canvas = max(512, int(round(height_bu * self.scale_res)))
        else:
            w_canvas = max(1024, int(round(width_bu * self.scale_res)))
            h_canvas = max(512, int(round(height_bu * self.scale_res)))

        # Substrate base color
        sub_rgb = self._parse_hex_color(substrate_color_hex)
        canvas_rgba = Image.new("RGBA", (w_canvas, h_canvas), (sub_rgb[0], sub_rgb[1], sub_rgb[2], 255))
        alpha_mask = Image.new("L", (w_canvas, h_canvas), 0)
        draw = ImageDraw.Draw(canvas_rgba)
        draw_alpha = ImageDraw.Draw(alpha_mask)

        # -------------------------------------------------------------
        # Step 1: Graphical Motif Segmentation & Alpha Extraction (§ 6.3)
        # -------------------------------------------------------------
        if reference_image_path and os.path.isfile(reference_image_path):
            self._composite_reference_motifs(
                canvas_rgba, alpha_mask, reference_image_path, geometry_spec, w_canvas, h_canvas
            )

        # -------------------------------------------------------------
        # Step 3: Resolution-Scaled Vector Typography & Layout (§ 6.3)
        # -------------------------------------------------------------
        rendered_elements = []
        for elem in typography_manifest:
            # Check if this text belongs to this component or is global
            target_comp = elem.get("target_component", "")
            if target_comp and target_comp != component_id and target_comp != "main_body" and target_comp != "all":
                continue

            text = elem.get("text", "")
            if not text:
                continue

            text_color = self._parse_hex_color(elem.get("color_hex", "#111111"))
            font_style = elem.get("font_style", "sans-serif").lower()
            orientation = elem.get("orientation", "horizontal").lower()
            rel_size = float(elem.get("relative_size", 0.08))

            # Resolution-scaled font point size
            font_size = max(14, int(h_canvas * rel_size))
            font = self._resolve_font(font_style, font_size)

            # Determine anchor placement (u, v)
            # Default zero-meridian center alignment: u=0.50 (Front camera)
            u_norm = float(elem.get("u_coord", 0.50))
            v_norm = float(elem.get("v_coord", 0.50))

            x_pos = int(u_norm * w_canvas)
            y_pos = int((1.0 - v_norm) * h_canvas)

            # Compute bounding box using textbbox
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            if orientation == "vertical":
                # Render vertically onto temporary image and rotate
                temp_img = Image.new("RGBA", (text_w + 20, text_h + 20), (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(temp_img)
                temp_draw.text((10, 10), text, font=font, fill=(text_color[0], text_color[1], text_color[2], 255))
                rot_img = temp_img.rotate(90, expand=True)

                paste_x = x_pos - rot_img.width // 2
                paste_y = y_pos - rot_img.height // 2
                canvas_rgba.paste(rot_img, (paste_x, paste_y), rot_img)
                # Composite onto alpha mask
                rot_alpha = rot_img.split()[3]
                alpha_mask.paste(rot_alpha, (paste_x, paste_y), rot_alpha)
            else:
                paste_x = x_pos - text_w // 2
                paste_y = y_pos - text_h // 2
                draw.text((paste_x, paste_y), text, font=font, fill=(text_color[0], text_color[1], text_color[2], 255))
                draw_alpha.text((paste_x, paste_y), text, font=font, fill=255)

            rendered_elements.append({
                "text": text,
                "anchor_uv": [round(u_norm, 3), round(v_norm, 3)],
                "pixel_bounds": [paste_x, paste_y, text_w, text_h],
                "font_size_pt": font_size,
                "orientation": orientation
            })

        # -------------------------------------------------------------
        # Step 4: Multi-Channel Material Map Baking (§ 6.3)
        # -------------------------------------------------------------
        # 1. Diffuse / Albedo Map
        diffuse_img = canvas_rgba.convert("RGB")
        diffuse_path = os.path.join(output_dir, "label_diffuse.png")
        diffuse_img.save(diffuse_path)

        # 2. Roughness Delta Map
        # R(x, y) = (1 - alpha) * R_substrate + alpha * R_ink
        alpha_arr = np.array(alpha_mask, dtype=np.float32) / 255.0
        rough_arr = (1.0 - alpha_arr) * substrate_roughness + alpha_arr * ink_roughness
        rough_arr = np.clip(rough_arr * 255.0, 0, 255).astype(np.uint8)
        roughness_img = Image.fromarray(rough_arr)
        roughness_path = os.path.join(output_dir, "label_roughness.png")
        roughness_img.save(roughness_path)

        # 3. Normal / Embossing Map (Optional height field gradients)
        normal_path = os.path.join(output_dir, "label_normal.png")
        normal_img = self._bake_embossed_normals(alpha_arr, strength=emboss_height)
        normal_img.save(normal_path)

        # -------------------------------------------------------------
        # Step 5: Manifest & Shader Node Graph Parameters
        # -------------------------------------------------------------
        manifest = {
            "component_id": component_id,
            "canvas_dimensions": {
                "width_px": w_canvas,
                "height_px": h_canvas,
                "aspect_ratio": round(float(w_canvas / float(max(1, h_canvas))), 3)
            },
            "substrate_material": {
                "color_hex": substrate_color_hex,
                "roughness": substrate_roughness
            },
            "ink_material": {
                "roughness": ink_roughness
            },
            "rendered_typography": rendered_elements,
            "texture_maps": {
                "diffuse": diffuse_path.replace("\\", "/"),
                "roughness": roughness_path.replace("\\", "/"),
                "normal": normal_path.replace("\\", "/")
            }
        }

        manifest_path = os.path.join(output_dir, "label_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[GraphicSynthesizer] Baked packaging art maps for '{component_id}' -> {output_dir}")
        return manifest

    def _bake_embossed_normals(self, alpha_norm: np.ndarray, strength: float = 0.04) -> Image.Image:
        """Computes tangent-space normal map from the alpha height field gradient."""
        # Blur slightly for smooth vector edges
        blurred_alpha = cv2.GaussianBlur(alpha_norm, (5, 5), 1.0)
        gx = cv2.Sobel(blurred_alpha, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred_alpha, cv2.CV_32F, 0, 1, ksize=3)

        scale = 5.0
        nx = -gx * scale
        ny = -gy * scale
        nz = np.ones_like(nx)
        length = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6

        nx_norm = nx / length
        ny_norm = ny / length
        nz_norm = nz / length

        h, w = alpha_norm.shape
        norm_map = np.zeros((h, w, 3), dtype=np.uint8)
        norm_map[:, :, 0] = np.clip(128.0 + 127.0 * nx_norm, 0, 255).astype(np.uint8)
        norm_map[:, :, 1] = np.clip(128.0 + 127.0 * ny_norm, 0, 255).astype(np.uint8)
        norm_map[:, :, 2] = np.clip(128.0 + 127.0 * nz_norm, 0, 255).astype(np.uint8)

        return Image.fromarray(norm_map)

    def _composite_reference_motifs(
        self,
        canvas: Image.Image,
        alpha_mask: Image.Image,
        ref_path: str,
        geom_spec: Dict[str, Any],
        w_canvas: int,
        h_canvas: int
    ):
        """Extracts high-contrast central graphic motifs from reference image."""
        try:
            ref_bgr = cv2.imread(ref_path)
            if ref_bgr is None:
                return

            bounds = geom_spec.get("snapped_pixel_y_bounds")
            bbox = geom_spec.get("pixel_bbox")
            if bounds and bbox:
                y1, y2 = bounds
                bx, bw = bbox[0], bbox[2]
                crop_bgr = ref_bgr[y1:y2, bx:bx + bw]
            elif bbox:
                bx, by, bw, bh = bbox
                crop_bgr = ref_bgr[by:by + bh, bx:bx + bw]
            else:
                crop_bgr = ref_bgr

            if crop_bgr.size == 0 or crop_bgr.shape[0] < 20 or crop_bgr.shape[1] < 20:
                return

            # Center 60% ROI of the component crop
            ch, cw = crop_bgr.shape[:2]
            roi = crop_bgr[int(ch * 0.2):int(ch * 0.8), int(cw * 0.2):int(cw * 0.8)]
            if roi.size == 0:
                return

            # Alpha matting via luminance thresholding
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Find ink features differing from background
            med_lum = np.median(gray_roi)
            diff = np.abs(gray_roi.astype(np.float32) - med_lum)
            alpha = np.clip((diff - 25.0) * 4.0, 0, 255).astype(np.uint8)

            if np.sum(alpha > 64) > 100:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                rgba_motif = np.dstack((roi_rgb, alpha))
                motif_img = Image.fromarray(rgba_motif)

                # Scale motif to fit central front camera axis (u = 0.50)
                target_w = int(w_canvas * 0.25)
                target_h = int(motif_img.height * (target_w / float(max(1, motif_img.width))))
                target_h = min(int(h_canvas * 0.6), target_h)
                motif_resized = motif_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

                # Paste centered at u = 0.50 (X = w_canvas // 2)
                cx = w_canvas // 2
                cy = h_canvas // 2
                px = cx - target_w // 2
                py = cy - target_h // 2

                canvas.paste(motif_resized, (px, py), motif_resized)
                motif_alpha = motif_resized.split()[3]
                alpha_mask.paste(motif_alpha, (px, py), motif_alpha)
        except Exception as e:
            print(f"[GraphicSynthesizer] Reference motif extraction notice: {e}")

    def _resolve_font(self, font_style: str, font_size: int) -> ImageFont.ImageFont:
        """Resolves system font or falls back cleanly."""
        font_candidates = [
            "arial.ttf", "segoeui.ttf", "calibri.ttf", "tahoma.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        for cand in font_candidates:
            try:
                return ImageFont.truetype(cand, font_size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _parse_hex_color(self, hex_code: Optional[str]) -> Tuple[int, int, int]:
        """Safely parse hex color string to (R, G, B)."""
        if hex_code and hex_code.startswith("#") and len(hex_code) == 7:
            try:
                r = int(hex_code[1:3], 16)
                g = int(hex_code[3:5], 16)
                b = int(hex_code[5:7], 16)
                return (r, g, b)
            except ValueError:
                pass
        return (128, 128, 128)
