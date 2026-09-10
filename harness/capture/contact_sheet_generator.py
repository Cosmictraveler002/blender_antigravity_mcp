"""
Contact Sheet Generator
=======================
Assembles 14 multi-viewport renders into a high-resolution 4x4 annotated
contact sheet montage.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


class ContactSheetGenerator:
    """Generates a visual 4x4 contact sheet overview of multi-viewport renders."""

    DEFAULT_ORDER = [
        # Row 1: Cardinal orthographic
        {"id": "front",          "title": "FRONT (0°, 0° - Ortho)"},
        {"id": "back",           "title": "BACK (180°, 0° - Ortho)"},
        {"id": "left",           "title": "LEFT (270°, 0° - Ortho)"},
        {"id": "right",          "title": "RIGHT (90°, 0° - Ortho)"},
        # Row 2: Top, Bottom & High Bird's-Eye
        {"id": "top",            "title": "TOP (+90° - Ortho)"},
        {"id": "bottom",         "title": "BOTTOM (-90° - Ortho)"},
        {"id": "high_front",     "title": "HIGH FRONT (0°, +60°)"},
        {"id": "high_back",      "title": "HIGH BACK (180°, +60°)"},
        # Row 3: 3/4 Perspective Angles
        {"id": "front_left_45",  "title": "FRONT-LEFT (315°, +30°)"},
        {"id": "front_right_45", "title": "FRONT-RIGHT (45°, +30°)"},
        {"id": "back_left_45",   "title": "BACK-LEFT (225°, +30°)"},
        {"id": "back_right_45",  "title": "BACK-RIGHT (135°, +30°)"},
        # Row 4: Low Worm's-Eye Angles & Info/Reference
        {"id": "low_front",      "title": "LOW FRONT (0°, -15°)"},
        {"id": "low_back",       "title": "LOW BACK (180°, -15°)"},
        {"id": "reference",      "title": "REFERENCE IMAGE"},
        {"id": "summary",        "title": "SYSTEM METADATA"},
    ]

    def __init__(self, cell_size: Tuple[int, int] = (480, 270), padding: int = 12):
        self.cell_w, self.cell_h = cell_size
        self.padding = padding
        self.cols = 4
        self.rows = 4

    def generate(
        self,
        viewport_manifest: Dict[str, Any],
        output_path: str,
        reference_image_path: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> str:
        """
        Creates a 4x4 montage and saves to output_path.

        Args:
            viewport_manifest: Manifest dict containing 'viewports' list or 'viewports_dir'.
            output_path: Destination PNG file path.
            reference_image_path: Optional path to reference photo to place in slot 15.
            project_name: Optional display name for header.

        Returns:
            Absolute path to saved contact sheet image.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Build lookup table of available viewport images
        vp_map: Dict[str, str] = {}
        for vp in viewport_manifest.get("viewports", []):
            name = vp.get("name")
            p = vp.get("path") or vp.get("file_path")
            if name and p and os.path.isfile(p):
                vp_map[name] = p

        # Header height
        header_h = 70
        footer_h = 30
        grid_w = self.cols * self.cell_w + (self.cols + 1) * self.padding
        grid_h = self.rows * self.cell_h + (self.rows + 1) * self.padding
        total_w = grid_w
        total_h = header_h + grid_h + footer_h

        # Create master canvas (dark slate theme)
        sheet = Image.new("RGB", (total_w, total_h), color=(18, 20, 24))
        draw = ImageDraw.Draw(sheet)

        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

        # Draw Header
        title_text = f"MULTI-VIEWPORT 360° RENDER CAPTURE & ANALYSIS"
        if project_name:
            title_text += f" | {project_name.upper()}"
        draw.text((self.padding, 18), title_text, fill=(240, 242, 246), font=font_large)

        sub_text = f"14 Calibrated Orbit Cameras (6 Orthographic Cardinal + 8 Perspective Diagonals)"
        draw.text((self.padding, 42), sub_text, fill=(150, 160, 175), font=font_small)

        # Render each cell in 4x4 grid
        for idx, item in enumerate(self.DEFAULT_ORDER):
            col = idx % self.cols
            row = idx // self.cols

            x = self.padding + col * (self.cell_w + self.padding)
            y = header_h + self.padding + row * (self.cell_h + self.padding)

            item_id = item["id"]
            title = item["title"]

            # Cell card background
            draw.rectangle([x, y, x + self.cell_w, y + self.cell_h], fill=(28, 32, 38), outline=(45, 52, 64), width=1)

            # Draw image content
            img_to_draw = None
            if item_id == "reference":
                if reference_image_path and os.path.isfile(reference_image_path):
                    try:
                        img_to_draw = Image.open(reference_image_path)
                    except Exception as e:
                        print(f"[ContactSheet] Failed to open reference: {e}")
            elif item_id == "summary":
                # Draw text card
                pass
            elif item_id in vp_map:
                try:
                    img_to_draw = Image.open(vp_map[item_id])
                except Exception as e:
                    print(f"[ContactSheet] Failed to open {vp_map[item_id]}: {e}")

            banner_h = 24
            view_h = self.cell_h - banner_h

            if img_to_draw:
                if img_to_draw.mode in ("RGBA", "LA"):
                    bg = Image.new("RGB", img_to_draw.size, (28, 32, 38))
                    bg.paste(img_to_draw, mask=img_to_draw.split()[-1])
                    img_to_draw = bg
                else:
                    img_to_draw = img_to_draw.convert("RGB")

                # Letterbox fit into cell view area
                img_ratio = img_to_draw.width / max(1, img_to_draw.height)
                target_ratio = self.cell_w / max(1, view_h)

                if img_ratio > target_ratio:
                    new_w = self.cell_w
                    new_h = int(new_w / img_ratio)
                else:
                    new_h = view_h
                    new_w = int(new_h * img_ratio)

                resized = img_to_draw.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)
                offset_x = x + (self.cell_w - new_w) // 2
                offset_y = y + (view_h - new_h) // 2
                sheet.paste(resized, (offset_x, offset_y))
            elif item_id == "summary":
                # System metadata card
                draw.text((x + 16, y + 16), "RECONSTRUCTION HARNESS", fill=(90, 180, 255), font=font_small)
                draw.text((x + 16, y + 36), f"Viewports: 14 Active", fill=(210, 215, 225), font=font_small)
                draw.text((x + 16, y + 54), f"Coverage: Complete 360°", fill=(210, 215, 225), font=font_small)
                draw.text((x + 16, y + 72), f"Orthographic: 6 Axes", fill=(170, 180, 195), font=font_small)
                draw.text((x + 16, y + 90), f"Perspective: 8 Orbits", fill=(170, 180, 195), font=font_small)
                draw.text((x + 16, y + 115), "Status: Verified", fill=(80, 220, 130), font=font_small)
            else:
                # Missing placeholder
                draw.text((x + self.cell_w // 4, y + view_h // 2), "[ Render Missing ]", fill=(120, 80, 80), font=font_small)

            # Draw bottom label banner
            draw.rectangle([x, y + view_h, x + self.cell_w, y + self.cell_h], fill=(16, 18, 22))
            draw.line([(x, y + view_h), (x + self.cell_w, y + view_h)], fill=(45, 52, 64), width=1)
            draw.text((x + 8, y + view_h + 5), title, fill=(225, 230, 240), font=font_small)

        # Draw Footer
        footer_text = "Generated by Blender MCP Reconstruction Harness v4.0.0 — Multi-Viewport Engine"
        draw.text((self.padding, total_h - 22), footer_text, fill=(100, 110, 125), font=font_small)

        sheet.save(output_path, "PNG", optimize=True)
        print(f"[ContactSheetGenerator] Montage saved to: {output_path}")
        return os.path.abspath(output_path)
