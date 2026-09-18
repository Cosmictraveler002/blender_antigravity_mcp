"""
Decal and Surface Texture Generator: NEXUS Mirrorless Camera
===========================================================
Generates crisp, high-resolution branding decals, UI screens, and normal maps.
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

output_dir = r"C:\Users\PC\myapps\Blender works\projects\nexus_camera\outputs\textures\decals"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. NEXUS Pentaprism Logo Decal (512x256)
# -------------------------------------------------------------
def make_nexus_logo():
    w, h = 512, 256
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try standard system fonts or draw high-contrast geometric typography
    try:
        font_large = ImageFont.truetype("arialbd.ttf", 64)
        font_small = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw "NEXUS" bold uppercase in crisp silver/white
    text_main = "NEXUS"
    bbox_main = draw.textbbox((0, 0), text_main, font=font_large)
    tw_main = bbox_main[2] - bbox_main[0]
    th_main = bbox_main[3] - bbox_main[1]
    x_main = (w - tw_main) // 2
    y_main = (h // 2) - th_main - 8

    # Slight glow/shadow for 3D realism
    draw.text((x_main, y_main), text_main, fill=(245, 246, 250, 255), font=font_large)

    # Draw subtitle "FX-N7 II" below
    text_sub = "FX-N7 II"
    bbox_sub = draw.textbbox((0, 0), text_sub, font=font_small)
    tw_sub = bbox_sub[2] - bbox_sub[0]
    x_sub = (w - tw_sub) // 2
    y_sub = (h // 2) + 14

    draw.text((x_sub, y_sub), text_sub, fill=(210, 212, 218, 220), font=font_small)

    out_p = os.path.join(output_dir, "nexus_logo_decal.png")
    img.save(out_p, "PNG")
    print(f"[Decal] Saved {out_p}")

# -------------------------------------------------------------
# 2. Greek Letter Alpha (α) Emblem (256x256)
# -------------------------------------------------------------
def make_alpha_emblem():
    w, h = 256, 256
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("georgiab.ttf", 160)
    except Exception:
        try:
            font = ImageFont.truetype("timesbd.ttf", 160)
        except Exception:
            font = ImageFont.load_default()

    text = "α"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = (h - th) // 2 - 10

    # Draw bright chrome/silver alpha symbol
    draw.text((x, y), text, fill=(240, 242, 248, 255), font=font)

    out_p = os.path.join(output_dir, "alpha_emblem.png")
    img.save(out_p, "PNG")
    print(f"[Decal] Saved {out_p}")

# -------------------------------------------------------------
# 3. Lens Bezel Text Ring (1024x1024 circular ring texture)
# -------------------------------------------------------------
def make_lens_ring():
    size = 1024
    img = Image.new("RGBA", (size, size), (22, 22, 24, 255))
    draw = ImageDraw.Draw(img)

    # Concentric metallic bevel rings
    cx, cy = size // 2, size // 2
    for r in range(480, 495):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(40, 40, 42, 255), width=1)

    # Circular text "35mm F/1.8" and "Ø55"
    try:
        font = ImageFont.truetype("arialbd.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    # Draw text top and bottom
    text_top = "35mm F/1.8"
    bbox_t = draw.textbbox((0, 0), text_top, font=font)
    draw.text((cx - (bbox_t[2] - bbox_t[0]) // 2, 70), text_top, fill=(235, 238, 242, 255), font=font)

    text_bot = "Ø55  NEXUS OPTICS"
    bbox_b = draw.textbbox((0, 0), text_bot, font=font)
    draw.text((cx - (bbox_b[2] - bbox_b[0]) // 2, size - 110), text_bot, fill=(200, 202, 208, 230), font=font)

    # Center aperture circle (inner opening)
    inner_r = 380
    draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r], fill=(8, 8, 10, 255))

    out_p = os.path.join(output_dir, "lens_ring_decal.png")
    img.save(out_p, "PNG")
    print(f"[Decal] Saved {out_p}")

# -------------------------------------------------------------
# 4. Pebbled Leatherette Normal Map (512x512)
# -------------------------------------------------------------
def make_leatherette_normal():
    w, h = 512, 512
    # Generate seamless synthetic Voronoi / cellular bumps
    np.random.seed(42)
    # Layer 1: fine grain
    fine = np.random.randn(h, w).astype(np.float32)
    import cv2
    fine = cv2.GaussianBlur(fine, (5, 5), 1.2)

    # Layer 2: medium pebbled bumps
    med = np.random.randn(h, w).astype(np.float32)
    med = cv2.GaussianBlur(med, (11, 11), 3.0)

    height = fine * 0.45 + med * 0.55
    gx = cv2.Scharr(height, cv2.CV_32F, 1, 0) * 1.8
    gy = cv2.Scharr(height, cv2.CV_32F, 0, 1) * 1.8
    gz = np.ones_like(gx)
    mag = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-6
    nx = gx / mag
    ny = gy / mag
    nz = gz / mag

    normal_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    normal_rgb[:, :, 0] = np.clip(128.0 + 127.0 * nx, 0, 255).astype(np.uint8)
    normal_rgb[:, :, 1] = np.clip(128.0 + 127.0 * ny, 0, 255).astype(np.uint8)
    normal_rgb[:, :, 2] = np.clip(128.0 + 127.0 * nz, 0, 255).astype(np.uint8)

    out_p = os.path.join(output_dir, "leatherette_normal.png")
    Image.fromarray(normal_rgb).save(out_p)
    print(f"[Decal] Saved {out_p}")

if __name__ == "__main__":
    make_nexus_logo()
    make_alpha_emblem()
    make_lens_ring()
    make_leatherette_normal()
